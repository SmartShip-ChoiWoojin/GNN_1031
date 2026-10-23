"""재생성 모델 채점 — 테스트 셋(10%)을 **처음이자 한 번** 연다.

`regen_train.py` 는 학습 셋 안에서만 임계값과 모델을 정했다. 여기서 그 모델로 테스트 셋
(52쌍, 학습 셋과 같은 결함을 공유하지 않는다)의 원본을 다시 만들고 사람의 결과와 비교한다.

지표 (주 지표가 맨 위):
  정확 일치      재생성 결과가 사람 결과와 요소·1D·좌표까지 똑같은가 (절점 번호 무관)
  바뀐 요소 IoU  지운 요소 + 만든 요소 집합이 사람과 얼마나 겹치나 (부분 점수)
  지운 요소 / 만든 요소 / 사라진 자리 — 각각 정밀도·재현율
  불변식         meshedit.check(regen=True) 위반 수 (1D 이탈, 비매니폴드 증가, 법선 반전 …)
  심판 확률      분류기(deploy_10_14)가 본 비정상 확률 — 원본 / 재생성 / 사람 (보조 지표)

함께 적는 기준선:
  변경 없음      원본을 그대로 내는 것 — 고칠 게 없다고 답하는 경우
  상한           정답 대본을 그대로 적용한 것 (`regen_oracle.py`) — 이 연산 체계의 천장

    python src/regen_eval.py --run runs/regen_final                # 테스트 셋
    python src/regen_eval.py --run runs/regen_final --split cv     # 학습 셋 out-of-fold (정직)
    python src/regen_eval.py --run runs/regen_final --split train  # 학습에 쓴 메쉬 (낙관적)
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch

import gae_paths
import features as F
from bdf_io import load_raw_mesh
from meshdiff import compare, signature
from meshedit import EditableMesh, check, replay_ops
from regen_data import load_samples, load_split
from regen_model import DEFAULT_ENCODER_RUN, RegenModel, RegenResult, cached_collapse_cands
from regen_oracle import parse_collapses, parse_edges
from regen_train import edges_str, ops_str


def judge_fn(device, run: str = DEFAULT_ENCODER_RUN):
    """메쉬 → 심판의 비정상 확률 함수를 만들어 돌려준다. `judge_pairs.py` 와 같은 계산.

    심판은 `deploy_10_14` — 수정 쌍 508개의 원본을 학습에서 뺀 분류기다 (train_judge.py).
    """
    import dataset as DS
    import graphs
    from explain import load_full_model
    from meshrep import MeshRep

    judge_model, normalizer, classifier_args, platt = load_full_model(run, device)
    graph_options = {"corner_edges": not classifier_args.no_corner_edges,
                     "include_lines": not classifier_args.no_lines}

    @torch.no_grad()
    def abnormal_probability(mesh) -> float:
        if hasattr(mesh, "to_raw"):
            raw_mesh = mesh.to_raw()
        else:
            raw_mesh = mesh
        graph_arrays = graphs.build(MeshRep(raw_mesh), classifier_args.graph, **graph_options)
        data = DS.to_pyg(graph_arrays, 0, raw_mesh.name, normalizer).to(device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        logit = float(judge_model(data).item())
        # Platt 보정: 1 / (1 + exp(-(a * logit + b)))
        return 1.0 / (1.0 + math.exp(-(platt[0] * logit + platt[1])))

    return abnormal_probability


def quality_counts(mesh):
    """(삼각형 수, 불량각 요소 수)."""
    triangle_count = 0
    bad_count = 0
    for element_id, element_type, node_ids in mesh.shells:
        point_list = []
        for node_id in node_ids:
            point_list.append(mesh.coords[node_id])
        quality = F.shell_quality(np.array(point_list, dtype=float))
        if not quality.is_quad:
            triangle_count = triangle_count + 1
        if F.is_bad_shell(quality):
            bad_count = bad_count + 1
    return triangle_count, bad_count


def wilson(success_count: int, total: int, z: float = 1.96):
    """이항 비율의 95% 신뢰구간 (Wilson). 테스트가 52쌍뿐이라 구간이 넓다는 걸 숫자로 보이려고."""
    if total == 0:
        return 0.0, 0.0
    rate = success_count / total
    center = (rate + z * z / (2 * total)) / (1 + z * z / total)
    half_width = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, center - half_width), min(1.0, center + half_width)


def evaluate(model: RegenModel | None, names: list, split_table: pd.DataFrame, device,
             out_dir: str | None = None, verbose: bool = True, recorded: pd.DataFrame | None = None) -> pd.DataFrame:
    """메쉬마다 재생성해 채점한다.

    ``recorded`` 를 주면 모델을 돌리지 않고 그 표의 연산(collapses/merges 열)을 원본에
    다시 적용한다 — 학습 셋의 out-of-fold 결과(cv_pred.csv)를 테스트와 같은 지표로 재는 데 쓴다.
    """
    samples = load_samples(names, verbose=False)
    first_candidates = None
    recorded_by_name = None
    if recorded is None:
        names_and_meshes = []
        for sample in samples:
            names_and_meshes.append((sample.name, EditableMesh(sample.pair.before)))
        first_candidates = cached_collapse_cands(names_and_meshes, verbose=verbose)
    else:
        recorded_by_name = recorded.set_index("name")

    judge = judge_fn(device)
    info_by_name = split_table.set_index("name")
    rows = []
    count = 0
    for sample in samples:
        count = count + 1
        bad_mesh = EditableMesh(sample.pair.before)
        if recorded is None:
            regen_result = model.regenerate(bad_mesh, first_candidates[sample.name])
        else:
            collapses = parse_collapses(recorded_by_name.loc[sample.name, "collapses"])
            merges = parse_edges(recorded_by_name.loc[sample.name, "merges"])
            replayed_mesh = replay_ops(bad_mesh, collapses, merges)
            regen_result = RegenResult(replayed_mesh, None, collapses, merges, {}, [])

        row = compare(sample.pair.before, regen_result.mesh, sample.pair.after)
        # 화면(app_regen)은 모델을 돌리지 않고 이 기록을 다시 적용해 그린다 — 같은지 확인
        replayed_again = replay_ops(bad_mesh, regen_result.collapses, regen_result.merges)
        replay_ok = signature(replayed_again) == signature(regen_result.mesh)
        check_report = check(regen_result.mesh, bad_mesh, regen=True)
        triangles_bad, bad_angles_bad = quality_counts(sample.pair.before)
        triangles_regen, bad_angles_regen = quality_counts(regen_result.mesh)
        triangles_good, bad_angles_good = quality_counts(sample.pair.after)

        margin_texts = []
        for margin in regen_result.margins:
            margin_texts.append(f"{margin:.2f}")
        row.update(
            name=sample.name, kind=info_by_name.loc[sample.name, "kind"],
            oracle=info_by_name.loc[sample.name, "oracle"],
            group=int(info_by_name.loc[sample.name, "group"]),
            n_collapse=len(regen_result.collapses), n_merge=len(regen_result.merges),
            n_rejected=len(regen_result.rejected), collapses=ops_str(regen_result.collapses),
            merges=edges_str(regen_result.merges), replay_ok=replay_ok,
            margins=";".join(margin_texts),
            true_collapses=ops_str(sample.collapses), true_merges=edges_str(sample.merges),
            n_viol=len(check_report.violations), viol=" | ".join(check_report.violations),
            n_warn=len(check_report.warnings), warn=" | ".join(check_report.warnings),
            p_bad=judge(sample.pair.before), p_regen=judge(regen_result.mesh), p_good=judge(sample.pair.after),
            tri_bad=triangles_bad, tri_regen=triangles_regen, tri_good=triangles_good,
            badang_bad=bad_angles_bad, badang_regen=bad_angles_regen, badang_good=bad_angles_good,
        )

        if out_dir:
            bdf_path = os.path.join(out_dir, f"{sample.name}_regen.bdf")
            json_path = os.path.join(out_dir, f"{sample.name}_regen.json")
            regen_result.mesh.write_bdf(bdf_path)
            regen_result.mesh.write_json(json_path)
            # 쓴 파일을 프로젝트 파서로 다시 읽어 같은지 확인한다 (현업 .bdf 로 나가는 출구라서)
            read_back = load_raw_mesh(bdf_path, json_path)
            read_back_ext = {}
            for node_id, is_ext in read_back.ext.items():
                read_back_ext[node_id] = bool(is_ext)
            same_shape = signature(read_back) == signature(regen_result.mesh)
            row["file_ok"] = same_shape and read_back_ext == regen_result.mesh.ext
        rows.append(row)
        if verbose and count % 10 == 0:
            print(f"  ... {count}/{len(samples)}")
    return pd.DataFrame(rows)


def cluster_ci(table: pd.DataFrame, column: str = "exact", n_boot: int = 4000, seed: int = 0):
    """같은 결함 묶음을 통째로 다시 뽑는 bootstrap 95% 구간.

    테스트 52쌍은 독립이 아니다 — 같은 결함을 고친 쌍(최대 9개)이 한꺼번에 맞거나 틀린다.
    실측 테스트는 24묶음뿐이라, 쌍을 독립으로 보는 Wilson 구간은 너무 좁다.
    """
    if "group" not in table:
        return float("nan"), float("nan")
    random_generator = np.random.default_rng(seed)
    per_group = table.groupby("group")[column].agg(["sum", "size"])
    group_sums = per_group["sum"].to_numpy(float)
    group_sizes = per_group["size"].to_numpy(float)
    sampled_groups = random_generator.integers(0, len(per_group), size=(n_boot, len(per_group)))
    rates = group_sums[sampled_groups].sum(1) / group_sizes[sampled_groups].sum(1)
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def summarize(table: pd.DataFrame) -> dict:
    pair_count = len(table)
    exact_count = int(table.exact.sum())
    wilson_low, wilson_high = wilson(exact_count, pair_count)
    cluster_low, cluster_high = cluster_ci(table)
    reproducible = table[table.oracle == "재현됨"]
    no_change_needed = table.n_rm_true.eq(0) & table.n_cr_true.eq(0) & table.n_del_true.eq(0)

    if "group" in table:
        group_count = int(table.group.nunique())
    else:
        group_count = None
    if len(reproducible) > 0:
        exact_reproducible = float(reproducible.exact.mean())
    else:
        exact_reproducible = None
    if "file_ok" in table:
        file_ok = float(table.file_ok.mean())
    else:
        file_ok = None

    summary = {
        "n": pair_count,
        "exact": exact_count / pair_count, "exact_ci95": [wilson_low, wilson_high], "exact_k": exact_count,
        "exact_ci95_cluster": [cluster_low, cluster_high], "n_groups": group_count,
        "exact_shells": float(table.exact_shells.mean()),
        "exact_reproducible": exact_reproducible,
        "n_reproducible": len(reproducible),
        "change_iou": float(table.change_iou.mean()),
    }
    for measure in ("rm", "cr", "del"):
        for precision_or_recall in ("prec", "rec"):
            column = f"{measure}_{precision_or_recall}"
            summary[column] = float(table[column].mean())

    summary["did_nothing"] = float(table.did_nothing.mean())
    summary["replay_ok"] = float(table.replay_ok.mean())
    summary["file_ok"] = file_ok
    summary["violations_mesh_frac"] = float((table.n_viol > 0).mean())
    summary["warnings_mesh_frac"] = float((table.n_warn > 0).mean())
    summary["baseline_no_change_exact"] = float(no_change_needed.mean())
    summary["ceiling_oracle_exact"] = float((table.oracle == "재현됨").mean())

    judge_median = {}
    judge_abnormal = {}
    for column in ("p_bad", "p_regen", "p_good"):
        judge_median[column] = float(table[column].median())
        judge_abnormal[column] = float((table[column] > 0.5).mean())
    summary["judge_median"] = judge_median
    summary["judge_abnormal_frac"] = judge_abnormal

    bad_angle_sums = {}
    for column in ("badang_bad", "badang_regen", "badang_good"):
        bad_angle_sums[column] = int(table[column].sum())
    summary["badang_sum"] = bad_angle_sums
    triangle_sums = {}
    for column in ("tri_bad", "tri_regen", "tri_good"):
        triangle_sums[column] = int(table[column].sum())
    summary["tri_sum"] = triangle_sums

    by_kind = {}
    for kind, kind_table in table.groupby("kind"):
        by_kind[kind] = {"n": int(len(kind_table)), "exact": float(kind_table.exact.mean()),
                         "iou": float(kind_table.change_iou.mean())}
    summary["by_kind"] = by_kind
    return summary


def print_summary(summary: dict, tag: str, split_name: str) -> None:
    print(f"\n══ {tag} · {split_name} {summary['n']}쌍 ══")
    print(f"  정확 일치            {summary['exact']:.1%}  ({summary['exact_k']}/{summary['n']}, "
          f"95% 구간 {summary['exact_ci95'][0]:.1%}~{summary['exact_ci95'][1]:.1%})")
    if summary.get("n_groups"):
        print(f"    결함 묶음 {summary['n_groups']}개 단위 bootstrap 95% 구간 "
              f"{summary['exact_ci95_cluster'][0]:.1%}~{summary['exact_ci95_cluster'][1]:.1%}  "
              f"(같은 결함의 쌍은 같이 맞거나 틀린다)")
    if summary["exact_reproducible"] is not None:
        print(f"    재현 가능한 {summary['n_reproducible']}쌍 중  {summary['exact_reproducible']:.1%}")
    print(f"  요소만 일치          {summary['exact_shells']:.1%}")
    print(f"  바뀐 요소 IoU        {summary['change_iou']:.3f}")
    print(f"  지운 요소   정밀 {summary['rm_prec']:.3f} / 재현 {summary['rm_rec']:.3f}")
    print(f"  만든 요소   정밀 {summary['cr_prec']:.3f} / 재현 {summary['cr_rec']:.3f}")
    print(f"  사라진 자리 정밀 {summary['del_prec']:.3f} / 재현 {summary['del_rec']:.3f}")
    print(f"  아무것도 안 바꾼 메쉬 {summary['did_nothing']:.1%}")
    print(f"  기록 재적용 = 결과     {summary['replay_ok']:.1%}  (화면이 채점된 결과를 그대로 보여주는가)")
    if summary.get("file_ok") is not None:
        print(f"  .bdf 다시 읽기 = 결과  {summary['file_ok']:.1%}  (내보낸 파일이 결과와 같은가)")
    print(f"  불변식 위반 메쉬      {summary['violations_mesh_frac']:.1%}  (경고만 있는 메쉬 "
          f"{summary['warnings_mesh_frac']:.1%})")
    print(f"  기준선: 변경 없음 {summary['baseline_no_change_exact']:.1%} / 상한(정답 대본) "
          f"{summary['ceiling_oracle_exact']:.1%}")
    median = summary["judge_median"]
    abnormal = summary["judge_abnormal_frac"]
    print(f"  심판 p 중앙값  원본 {median['p_bad']:.3f} → 재생성 {median['p_regen']:.3f} (사람 {median['p_good']:.3f})")
    print(f"  비정상 판정    원본 {abnormal['p_bad']:.1%} → 재생성 {abnormal['p_regen']:.1%} (사람 {abnormal['p_good']:.1%})")
    bad_angles = summary["badang_sum"]
    triangles = summary["tri_sum"]
    print(f"  불량각 요소 합 원본 {bad_angles['badang_bad']} → 재생성 {bad_angles['badang_regen']} "
          f"(사람 {bad_angles['badang_good']})")
    print(f"  삼각형 합      원본 {triangles['tri_bad']} → 재생성 {triangles['tri_regen']} (사람 {triangles['tri_good']})")
    print("  종류별:")

    def pair_count_of(kind_and_values):
        return -kind_and_values[1]["n"]

    for kind, values in sorted(summary["by_kind"].items(), key=pair_count_of):
        print(f"    {kind:22s} {values['n']:3d}쌍  정확 {values['exact']:.1%}  IoU {values['iou']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="재생성 모델 채점")
    parser.add_argument("--run", required=True, help="runs/regen_<tag>")
    parser.add_argument("--split", choices=["test", "cv", "train", "all"], default="test",
                        help="cv = 학습 셋의 out-of-fold 결과(cv_pred.csv)를 다시 적용해 같은 지표로 잰다")
    parser.add_argument("--no-bdf", action="store_true", help="재생성 메쉬를 .bdf 로 쓰지 않는다")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if os.path.isabs(args.run):
        run_dir = args.run
    else:
        run_dir = os.path.join(gae_paths.ROOT, args.run)
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    split_table = load_split()

    recorded = None
    write_bdf = not args.no_bdf
    if args.split == "cv":
        # 학습 셋의 out-of-fold 결과 — 각 메쉬는 자기를 학습에 안 쓴 fold 모델이 만든 것이다
        recorded = pd.read_csv(os.path.join(run_dir, "cv_pred.csv"), encoding="utf-8-sig", keep_default_na=False)
        names = sorted(recorded.name)
        model = None
        write_bdf = False
    else:
        model = RegenModel.load(run_dir, device)
        if args.split == "all":
            names = sorted(split_table.name)
        else:
            names = sorted(split_table[split_table.split == args.split].name)
    if args.split in ("train", "all"):
        print("[주의] 학습에 쓴 메쉬가 섞여 있습니다 — 이 숫자는 낙관적입니다. "
              "학습 셋의 정직한 숫자는 --split cv (out-of-fold) 로 보세요.")

    tag = os.path.basename(run_dir)
    out_dir = None
    if write_bdf:
        out_dir = os.path.join(gae_paths.OUT_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)
    if model is not None:
        print(f"[regen_eval] {tag} · {args.split} {len(names)}쌍 · 1단계 {model.decode} "
              f"· 임계값 {model.stage1_threshold:.2f}/{model.stage2_threshold:.2f}")
    else:
        print(f"[regen_eval] {tag} · 학습 셋 out-of-fold {len(names)}쌍 (기록된 연산을 다시 적용)")

    table = evaluate(model, names, split_table, device, out_dir, recorded=recorded)
    table.to_csv(os.path.join(run_dir, f"eval_{args.split}.csv"), index=False, encoding="utf-8-sig")
    summary = summarize(table)
    with open(os.path.join(run_dir, f"eval_{args.split}.json"), "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)
    print_summary(summary, tag, args.split)
    print(f"\n저장: {os.path.join(run_dir, f'eval_{args.split}.csv')}")
    if out_dir:
        print(f"재생성 메쉬: {out_dir}")


if __name__ == "__main__":
    main()

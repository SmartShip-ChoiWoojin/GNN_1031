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

    python src/regen_eval.py --run runs/regen_gnn
    python src/regen_eval.py --run runs/regen_gnn --split cv      # 학습 셋 out-of-fold (정직)
    python src/regen_eval.py --run runs/regen_gnn --split train   # 학습에 쓴 메쉬 (낙관적)
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch

import gae_paths as gp
import features as F
from bdf_io import load_raw_mesh
from meshdiff import compare, signature
from meshedit import EditableMesh, check, replay_ops
from regen_data import load_samples, load_split
from regen_model import DEFAULT_ENCODER_RUN, RegenModel, cached_collapse_cands
from regen_train import edges_str, ops_str


def judge_fn(device, run: str = DEFAULT_ENCODER_RUN):
    """메쉬 → 심판의 비정상 확률. `judge_pairs.prob` 과 같은 계산.

    심판은 `deploy_10_14` — 수정 쌍 508개의 원본을 학습에서 뺀 분류기다 (train_judge.py).
    """
    import dataset as DS
    import graphs
    from explain import load_full_model
    from meshrep import MeshRep

    model, norm, args, platt = load_full_model(run, device)
    opts = {"corner_edges": not args.no_corner_edges, "include_lines": not args.no_lines}

    @torch.no_grad()
    def prob(mesh) -> float:
        raw = mesh.to_raw() if hasattr(mesh, "to_raw") else mesh
        ga = graphs.build(MeshRep(raw), args.graph, **opts)
        d = DS.to_pyg(ga, 0, raw.name, norm).to(device)
        d.batch = torch.zeros(d.x.size(0), dtype=torch.long, device=device)
        logit = float(model(d).item())
        return 1.0 / (1.0 + math.exp(-(platt[0] * logit + platt[1])))

    return prob


def quality(mesh) -> tuple[int, int]:
    """(삼각형 수, 불량각 요소 수)."""
    n_tri = n_bad = 0
    for _, _, ns in mesh.shells:
        q = F.shell_quality(np.array([mesh.coords[n] for n in ns], dtype=float))
        n_tri += not q.is_quad
        n_bad += F.is_bad_shell(q)
    return n_tri, n_bad


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 95% 신뢰구간 (Wilson). 테스트가 52쌍뿐이라 구간이 넓다는 걸 숫자로 보이려고."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def evaluate(model: RegenModel | None, names: list[str], split_df: pd.DataFrame, device,
             out_dir: str | None = None, verbose: bool = True,
             recorded: pd.DataFrame | None = None) -> pd.DataFrame:
    """메쉬마다 재생성해 채점한다.

    ``recorded`` 를 주면 모델을 돌리지 않고 그 표의 연산(collapses/merges 열)을 원본에
    다시 적용한다 — 학습 셋의 out-of-fold 결과(cv_pred.csv)를 테스트와 같은 지표로 재는 데 쓴다.
    """
    from types import SimpleNamespace

    samples = load_samples(names, verbose=False)
    if recorded is None:
        cands = cached_collapse_cands([(s.name, EditableMesh(s.pair.before)) for s in samples],
                                      verbose=verbose)
    else:
        rec = recorded.set_index("name")
    judge = judge_fn(device)
    info = split_df.set_index("name")
    rows = []
    for i, s in enumerate(samples, 1):
        bad = EditableMesh(s.pair.before)
        if recorded is None:
            res = model.regenerate(bad, cands[s.name])
        else:
            c = [tuple(int(x) for x in t.split(">")) for t in str(rec.loc[s.name, "collapses"]).split(";") if t]
            e = [frozenset(int(x) for x in t.split("-")) for t in str(rec.loc[s.name, "merges"]).split(";") if t]
            res = SimpleNamespace(mesh=replay_ops(bad, c, e), collapses=c, merges=e, rejected={}, margins=[])
        r = compare(s.pair.before, res.mesh, s.pair.after)
        # 화면(app_regen)은 모델을 돌리지 않고 이 기록을 다시 적용해 그린다 — 같은지 확인
        replay_ok = signature(replay_ops(bad, res.collapses, res.merges)) == signature(res.mesh)
        ck = check(res.mesh, bad, regen=True)
        tb, bb = quality(s.pair.before)
        tr, br = quality(res.mesh)
        tg, bg = quality(s.pair.after)
        r.update(
            name=s.name, kind=info.loc[s.name, "kind"], oracle=info.loc[s.name, "oracle"],
            group=int(info.loc[s.name, "group"]),
            n_collapse=len(res.collapses), n_merge=len(res.merges),
            n_rejected=len(res.rejected), collapses=ops_str(res.collapses),
            merges=edges_str(res.merges), replay_ok=replay_ok,
            margins=";".join(f"{x:.2f}" for x in res.margins),
            true_collapses=ops_str(s.collapses), true_merges=edges_str(s.merges),
            n_viol=len(ck.violations), viol=" | ".join(ck.violations),
            n_warn=len(ck.warnings), warn=" | ".join(ck.warnings),
            p_bad=judge(s.pair.before), p_regen=judge(res.mesh), p_good=judge(s.pair.after),
            tri_bad=tb, tri_regen=tr, tri_good=tg, badang_bad=bb, badang_regen=br, badang_good=bg,
        )
        if out_dir:
            fb = os.path.join(out_dir, f"{s.name}_regen.bdf")
            fj = os.path.join(out_dir, f"{s.name}_regen.json")
            res.mesh.write_bdf(fb)
            res.mesh.write_json(fj)
            # 쓴 파일을 프로젝트 파서로 다시 읽어 같은지 확인한다 (현업 .bdf 로 나가는 출구라서)
            back = load_raw_mesh(fb, fj)
            r["file_ok"] = (signature(back) == signature(res.mesh)
                            and {n: bool(v) for n, v in back.ext.items()} == res.mesh.ext)
        rows.append(r)
        if verbose and i % 10 == 0:
            print(f"  ... {i}/{len(samples)}")
    return pd.DataFrame(rows)


def cluster_ci(d: pd.DataFrame, col: str = "exact", n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """같은 결함 묶음을 통째로 다시 뽑는 bootstrap 95% 구간.

    테스트 52쌍은 독립이 아니다 — 같은 결함을 고친 쌍(최대 9개)이 한꺼번에 맞거나 틀린다.
    실측 테스트는 24묶음뿐이라, 쌍을 독립으로 보는 Wilson 구간은 너무 좁다.
    """
    if "group" not in d:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    g = d.groupby("group")[col].agg(["sum", "size"])
    s_, n_ = g["sum"].to_numpy(float), g["size"].to_numpy(float)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    r = s_[idx].sum(1) / n_[idx].sum(1)
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def summarize(d: pd.DataFrame) -> dict:
    n = len(d)
    k = int(d.exact.sum())
    lo, hi = wilson(k, n)
    clo, chi = cluster_ci(d)
    rep = d[d.oracle == "재현됨"]
    no_change = d.n_rm_true.eq(0) & d.n_cr_true.eq(0) & d.n_del_true.eq(0)
    out = {
        "n": n,
        "exact": k / n, "exact_ci95": [lo, hi], "exact_k": k,
        "exact_ci95_cluster": [clo, chi], "n_groups": int(d.group.nunique()) if "group" in d else None,
        "exact_shells": float(d.exact_shells.mean()),
        "exact_reproducible": float(rep.exact.mean()) if len(rep) else None,
        "n_reproducible": len(rep),
        "change_iou": float(d.change_iou.mean()),
        **{f"{m}_{pr}": float(d[f"{m}_{pr}"].mean())
           for m in ("rm", "cr", "del") for pr in ("prec", "rec")},
        "did_nothing": float(d.did_nothing.mean()),
        "replay_ok": float(d.replay_ok.mean()),
        "file_ok": float(d.file_ok.mean()) if "file_ok" in d else None,
        "violations_mesh_frac": float((d.n_viol > 0).mean()),
        "warnings_mesh_frac": float((d.n_warn > 0).mean()),
        "baseline_no_change_exact": float(no_change.mean()),
        "ceiling_oracle_exact": float((d.oracle == "재현됨").mean()),
        "judge_median": {c: float(d[c].median()) for c in ("p_bad", "p_regen", "p_good")},
        "judge_abnormal_frac": {c: float((d[c] > 0.5).mean()) for c in ("p_bad", "p_regen", "p_good")},
        "badang_sum": {c: int(d[c].sum()) for c in ("badang_bad", "badang_regen", "badang_good")},
        "tri_sum": {c: int(d[c].sum()) for c in ("tri_bad", "tri_regen", "tri_good")},
        "by_kind": {kd: {"n": int(len(g)), "exact": float(g.exact.mean()),
                         "iou": float(g.change_iou.mean())}
                    for kd, g in d.groupby("kind")},
    }
    return out


def print_summary(sm: dict, tag: str, split: str) -> None:
    print(f"\n══ {tag} · {split} {sm['n']}쌍 ══")
    print(f"  정확 일치            {sm['exact']:.1%}  ({sm['exact_k']}/{sm['n']}, "
          f"95% 구간 {sm['exact_ci95'][0]:.1%}~{sm['exact_ci95'][1]:.1%})")
    if sm.get("n_groups"):
        print(f"    결함 묶음 {sm['n_groups']}개 단위 bootstrap 95% 구간 "
              f"{sm['exact_ci95_cluster'][0]:.1%}~{sm['exact_ci95_cluster'][1]:.1%}  "
              f"(같은 결함의 쌍은 같이 맞거나 틀린다)")
    if sm["exact_reproducible"] is not None:
        print(f"    재현 가능한 {sm['n_reproducible']}쌍 중  {sm['exact_reproducible']:.1%}")
    print(f"  요소만 일치          {sm['exact_shells']:.1%}")
    print(f"  바뀐 요소 IoU        {sm['change_iou']:.3f}")
    print(f"  지운 요소   정밀 {sm['rm_prec']:.3f} / 재현 {sm['rm_rec']:.3f}")
    print(f"  만든 요소   정밀 {sm['cr_prec']:.3f} / 재현 {sm['cr_rec']:.3f}")
    print(f"  사라진 자리 정밀 {sm['del_prec']:.3f} / 재현 {sm['del_rec']:.3f}")
    print(f"  아무것도 안 바꾼 메쉬 {sm['did_nothing']:.1%}")
    print(f"  기록 재적용 = 결과     {sm['replay_ok']:.1%}  (화면이 채점된 결과를 그대로 보여주는가)")
    if sm.get("file_ok") is not None:
        print(f"  .bdf 다시 읽기 = 결과  {sm['file_ok']:.1%}  (내보낸 파일이 결과와 같은가)")
    print(f"  불변식 위반 메쉬      {sm['violations_mesh_frac']:.1%}  (경고만 있는 메쉬 "
          f"{sm['warnings_mesh_frac']:.1%})")
    print(f"  기준선: 변경 없음 {sm['baseline_no_change_exact']:.1%} / 상한(정답 대본) "
          f"{sm['ceiling_oracle_exact']:.1%}")
    j, f = sm["judge_median"], sm["judge_abnormal_frac"]
    print(f"  심판 p 중앙값  원본 {j['p_bad']:.3f} → 재생성 {j['p_regen']:.3f} (사람 {j['p_good']:.3f})")
    print(f"  비정상 판정    원본 {f['p_bad']:.1%} → 재생성 {f['p_regen']:.1%} (사람 {f['p_good']:.1%})")
    b, t = sm["badang_sum"], sm["tri_sum"]
    print(f"  불량각 요소 합 원본 {b['badang_bad']} → 재생성 {b['badang_regen']} (사람 {b['badang_good']})")
    print(f"  삼각형 합      원본 {t['tri_bad']} → 재생성 {t['tri_regen']} (사람 {t['tri_good']})")
    print("  종류별:")
    for kd, v in sorted(sm["by_kind"].items(), key=lambda x: -x[1]["n"]):
        print(f"    {kd:22s} {v['n']:3d}쌍  정확 {v['exact']:.1%}  IoU {v['iou']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="재생성 모델 채점")
    ap.add_argument("--run", required=True, help="runs/regen_<tag>")
    ap.add_argument("--split", choices=["test", "cv", "train", "all"], default="test",
                    help="cv = 학습 셋의 out-of-fold 결과(cv_pred.csv)를 다시 적용해 같은 지표로 잰다")
    ap.add_argument("--no-bdf", action="store_true", help="재생성 메쉬를 .bdf 로 쓰지 않는다")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    run = a.run if os.path.isabs(a.run) else os.path.join(gp.ROOT, a.run)
    device = torch.device("cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu")
    split = load_split()
    recorded = None
    if a.split == "cv":
        # 학습 셋의 out-of-fold 결과 — 각 메쉬는 자기를 학습에 안 쓴 fold 모델이 만든 것이다
        recorded = pd.read_csv(os.path.join(run, "cv_pred.csv"), encoding="utf-8-sig", keep_default_na=False)
        names = sorted(recorded.name)
        model = None
        a.no_bdf = True
    else:
        model = RegenModel.load(run, device)
        names = sorted(split.name) if a.split == "all" else sorted(split[split.split == a.split].name)
    if a.split in ("train", "all"):
        print("[주의] 학습에 쓴 메쉬가 섞여 있습니다 — 이 숫자는 낙관적입니다. "
              "학습 셋의 정직한 숫자는 --split cv (out-of-fold) 로 보세요.")
    tag = os.path.basename(run)
    out_dir = None
    if not a.no_bdf:
        out_dir = os.path.join(gp.OUT_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)
    if model is not None:
        print(f"[regen_eval] {tag} · {a.split} {len(names)}쌍 · 1단계 {model.decode} "
              f"· 임계값 {model.thr1:.2f}/{model.thr2:.2f}")
    else:
        print(f"[regen_eval] {tag} · 학습 셋 out-of-fold {len(names)}쌍 (기록된 연산을 다시 적용)")
    d = evaluate(model, names, split, device, out_dir, recorded=recorded)
    d.to_csv(os.path.join(run, f"eval_{a.split}.csv"), index=False, encoding="utf-8-sig")
    sm = summarize(d)
    with open(os.path.join(run, f"eval_{a.split}.json"), "w", encoding="utf-8") as fh:
        json.dump(sm, fh, ensure_ascii=False, indent=2)
    print_summary(sm, tag, a.split)
    print(f"\n저장: {os.path.join(run, f'eval_{a.split}.csv')}")
    if out_dir:
        print(f"재생성 메쉬: {out_dir}")


if __name__ == "__main__":
    main()

"""요소 지목 — 어텐션 게이트값으로 "어느 요소가 문제인지" 보여준다.

readout 의 어텐션 pooling 게이트(alpha)가 곧 요소별 기여도다.
alpha 는 그래프 안에서 합이 1 이므로, ``alpha × 노드수`` 로 보면 1.0 = 평균 중요도다.

    # 한 모델 시각화 (해당 모델이 test 였던 fold 의 모델을 자동 선택)
    python src/explain.py --run runs/elem_gatv2 --name SN2592_FWD_Clean_0001

    # 비정상 확신도 상위 6개
    python src/explain.py --run runs/elem_gatv2 --worst 6

    # 오분류 사례
    python src/explain.py --run runs/elem_gatv2 --errors 6

    # 데이터셋 전체 어텐션 검증 (지목이 실제 불량 요소와 맞는지)
    python src/explain.py --run runs/elem_gatv2 --audit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import dataset as DS
import features as F
import graph_elem
import graphs
import paths
from bdf_io import load_raw_mesh
from data_index import list_samples
from meshrep import MeshRep
from models import build_model


# ──────────────────────────────────────────────────────────────────────────
# 실행 결과 로딩
# ──────────────────────────────────────────────────────────────────────────
def load_result(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "result.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def fold_assignment(names: list[str], labels: list[int], args: SimpleNamespace) -> np.ndarray:
    """각 샘플이 어느 fold 의 test 였는지 복원한다.

    train.py 와 같은 splitter·seed 를 쓰므로 결정적으로 재현된다. 덕분에 항상
    **해당 샘플을 학습에 쓰지 않은 모델**로 설명할 수 있다 (정직한 지목).
    """
    y = np.array(labels)
    if args.split == "group":
        from data_index import spatial_groups

        by_name = {s.name: s for s in list_samples(verbose=False)}
        g = spatial_groups([by_name[n] for n in names])
        splitter = StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed)
        folds = splitter.split(np.zeros(len(y)), y, g)
    else:
        splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        folds = splitter.split(np.zeros(len(y)), y)

    out = np.full(len(y), -1, dtype=int)
    for k, (_, te) in enumerate(folds, start=1):
        out[te] = k
    assert (out > 0).all()
    return out


def load_fold_model(run_dir: str, fold: int, device: torch.device):
    """(model, normalizer, args, platt) 를 복원한다."""
    ck = torch.load(os.path.join(run_dir, f"fold{fold}.pt"), map_location=device,
                    weights_only=False)
    args = SimpleNamespace(**ck["args"])
    model = build_model(
        ck["d_node"], ck["d_edge"], ck["d_graph"], args,
        ext_mask_index=(
            None if args.ext_mask != "hard"
            else graphs.specs(args.graph)[0].index("all_ext" if args.graph == "elem" else "ext")
        ),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    norm = DS.NormalizerSet.from_state_dict(ck["normalizer"])
    return model, norm, args, tuple(ck.get("platt", (1.0, 0.0)))


# ──────────────────────────────────────────────────────────────────────────
# 어텐션 계산
# ──────────────────────────────────────────────────────────────────────────
def attention_for(ga, name: str, label: int, model, norm, device) -> tuple[float, np.ndarray]:
    """(비정상 logit, 노드별 alpha)"""
    data = DS.to_pyg(ga, label, name, norm).to(device)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        logit, alpha = model(data, return_alpha=True)
    return float(logit.item()), alpha.cpu().numpy()


def relative_importance(alpha: np.ndarray) -> np.ndarray:
    """alpha 를 '평균 대비 배수'로 바꾼다. 1.0 = 평균 중요도."""
    return alpha * len(alpha)


# ──────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────
def setup_korean_font() -> None:
    """그림 제목에 한글이 들어가므로 한글 폰트를 지정한다 (없으면 조용히 넘어간다)."""
    import matplotlib
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"):
        if cand in have:
            matplotlib.rcParams["font.family"] = cand
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


def render(
    rep: MeshRep,
    ga,
    imp: np.ndarray,
    title: str,
    ax=None,
    vmax: float | None = None,
):
    """쉘은 면 색, 1D 는 선 색으로 중요도를 칠한다.

    검은 테두리 = 불량 각도 요소(최소내각<30° 또는 최대내각>150°).
    빈 원 = 모델 외부와 연결된 절점(json ext=true). 그 주변의 이상함은 감점 대상이 아니다.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("inferno")
    vmax = vmax if vmax is not None else max(float(imp.max()), 1.5)

    n_shell = len(rep.shells)
    polys, colors, edge_colors, edge_widths = [], [], [], []
    for si, sh in enumerate(rep.shells):
        P = np.array([rep.raw.coords[n] for n in sh.nodes])
        polys.append(P)
        colors.append(cmap(min(imp[si] / vmax, 1.0)))
        bad = F.is_bad_shell(sh.quality)
        edge_colors.append("black" if bad else "0.55")
        edge_widths.append(1.6 if bad else 0.4)

    pc = Poly3DCollection(polys, facecolors=colors, edgecolors=edge_colors,
                          linewidths=edge_widths, alpha=0.92)
    ax.add_collection3d(pc)

    # 1D 요소 — 그래프에 노드로 들어간 경우에만 중요도가 있다
    has_line_nodes = ga.n_node > n_shell
    segs, lcolors, lwidths = [], [], []
    for li, ln in enumerate(rep.lines):
        a, b = ln.nodes
        segs.append([rep.raw.coords[a], rep.raw.coords[b]])
        if has_line_nodes and n_shell + li < len(imp):
            v = imp[n_shell + li]
            lcolors.append(cmap(min(v / vmax, 1.0)))
            lwidths.append(1.5 + 3.5 * min(v / vmax, 1.0))
        else:
            lcolors.append("tab:cyan")
            lwidths.append(1.5)
    if segs:
        ax.add_collection3d(
            Line3DCollection(segs, colors=lcolors, linewidths=lwidths)
        )

    ext_pts = np.array(
        [rep.raw.coords[n] for n in rep.node_ids if rep.ext.get(n, False)]
    )
    if len(ext_pts):
        ax.scatter(ext_pts[:, 0], ext_pts[:, 1], ext_pts[:, 2],
                   s=14, facecolors="none", edgecolors="tab:cyan", linewidths=0.8)

    P = rep.points
    ctr = P.mean(axis=0)
    r = max(float(np.abs(P - ctr).max()), 1.0) * 1.1
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=9)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.grid(False)
    return ax


def top_elements(rep: MeshRep, ga, imp: np.ndarray, k: int = 5) -> list[str]:
    """중요도 상위 **쉘** 요소를 사람이 읽을 문자열로.

    1D 요소는 여기서 뺀다 — 1D 는 형상 결함(각도/찌그러짐) 판정 대상이 아니라
    보강재 위상 정보일 뿐인데, 종합 순위에 섞으면 어텐션이 큰 1D 가 "왜 비정상인지"
    설명을 가로채는 경우가 있다(1D 노드는 shell 노드와 피처 인코더를 공유해서,
    타입 자체가 어텐션 게이트에 지름길로 작용할 가능성이 있다). 1D 별도 순위는
    :func:`top_line_elements` 로 확인한다.
    """
    n_shell = len(rep.shells)
    order = np.argsort(-imp[:n_shell])[:k]
    out = []
    for i in order:
        sh = rep.shells[i]
        q = sh.quality
        ext = "ext" if all(rep.ext.get(n, False) for n in sh.nodes) else "내부"
        out.append(
            f"{imp[i]:5.2f}x  {sh.etype} {sh.eid:>6d}  "
            f"내각 {q.angle_min:5.1f}~{q.angle_max:5.1f}°  "
            f"aspect {q.aspect:4.2f}  {'불량' if F.is_bad_shell(q) else '정상'}  {ext}"
        )
    return out


def top_line_elements(rep: MeshRep, imp: np.ndarray, k: int = 5) -> list[str]:
    """중요도 상위 **1D** 요소 — 참고용. 결함 판정과는 무관하다 (top_elements 참고)."""
    n_shell = len(rep.shells)
    if len(imp) <= n_shell or not rep.lines:
        return []
    order = n_shell + np.argsort(-imp[n_shell:])[:k]
    out = []
    for i in order:
        ln = rep.lines[i - n_shell]
        out.append(f"{imp[i]:5.2f}x  {ln.etype} {ln.eid:>6d}  길이 {ln.length:7.1f}")
    return out


# ──────────────────────────────────────────────────────────────────────────
# 전체 검증
# ──────────────────────────────────────────────────────────────────────────
def audit(run_dir: str, device: torch.device) -> None:
    """어텐션이 실제로 '문제 요소'를 짚는지 데이터셋 전체에서 검증한다.

    확인 항목
      1. 그래프 안에서 alpha 가 불량각도 요소를 골라내는가 (그래프별 AUC 평균)
      2. ``all_ext`` 요소(모델 절단면)가 상대적으로 낮은 중요도를 받는가
         — 사용자의 도메인 규칙("외부 연결된 곳의 이상함은 감점하지 말 것")을 학습했는지
      3. 비정상으로 예측한 그래프에서 최상위 요소가 실제 불량 요소인 비율
    """
    res = load_result(run_dir)
    args = SimpleNamespace(**res["args"])
    opts = {}
    if args.graph == "elem":
        opts = {"corner_edges": not args.no_corner_edges,
                "include_lines": not args.no_lines}
    names, labels, arrays = DS.build_cache(args.graph, opts=opts, verbose=False)
    fa = fold_assignment(names, labels, args)

    by_name = {s.name: s for s in list_samples(verbose=False)}

    auc_narrow, auc_wide = [], []
    hits = {1: [0, 0], 3: [0, 0]}   # k -> [적중, 전체]
    imp_ext, imp_int = [], []
    imp_bad, imp_ok = [], []

    for fold in range(1, args.folds + 1):
        model, norm, _, platt = load_fold_model(run_dir, fold, device)
        idxs = np.where(fa == fold)[0]
        for i in idxs:
            rep = MeshRep(load_raw_mesh(by_name[names[i]].bdf, by_name[names[i]].json))
            logit, alpha = attention_for(arrays[i], names[i], labels[i], model, norm, device)
            imp = relative_importance(alpha)
            prob = 1.0 / (1.0 + np.exp(-(platt[0] * logit + platt[1])))

            n_shell = len(rep.shells)
            s_imp = imp[:n_shell]
            bad = np.array([F.is_bad_shell(s.quality) for s in rep.shells])
            # 넓은 정의 — 각도 외에 형상비/왜곡도 '문제 요소'로 본다. 좁은 정의(각도만)로는
            # 심하게 찌그러졌지만 30~150° 안에 있는 사각형이 빠진다.
            wide = np.array([
                F.is_bad_shell(s.quality) or s.quality.aspect > 4.0 or s.quality.skew > 45.0
                for s in rep.shells
            ])
            allext = np.array(
                [all(rep.ext.get(n, False) for n in s.nodes) for s in rep.shells]
            )

            if bad.any() and not bad.all():
                auc_narrow.append(roc_auc_score(bad, s_imp))
            if wide.any() and not wide.all():
                auc_wide.append(roc_auc_score(wide, s_imp))
            imp_bad += list(s_imp[bad])
            imp_ok += list(s_imp[~bad])
            imp_ext += list(s_imp[allext])
            imp_int += list(s_imp[~allext])

            if prob > 0.5 and wide.any():
                order = np.argsort(-s_imp)
                for k in (1, 3):
                    hits[k][1] += 1
                    hits[k][0] += int(wide[order[:k]].any())

    print("=" * 72)
    print(f"어텐션 검증: {run_dir}")
    print("=" * 72)
    print("1) 그래프 안에서 어텐션이 문제 요소를 골라내는가 (0.5=무작위)")
    print(f"     좁은 정의(내각<30° 또는 >150°) AUC {np.mean(auc_narrow):.4f}"
          f"   그래프 {len(auc_narrow)}개")
    print(f"     넓은 정의(+aspect>4 또는 skew>45°) AUC {np.mean(auc_wide):.4f}"
          f"   그래프 {len(auc_wide)}개")
    print(f"     불량 요소 평균 중요도 {np.mean(imp_bad):.3f}x  "
          f"vs 나머지 {np.mean(imp_ok):.3f}x  "
          f"({np.mean(imp_bad) / max(np.mean(imp_ok), 1e-9):.2f}배)")
    print("     참고: 어텐션은 상위 몇 개에 강하게 집중되므로, 순위 전체보다 최상위를")
    print("           보는 용도가 맞다. 아래 top-k 적중률이 실사용에 더 가까운 지표다.")
    print()
    print("2) 절단면(all_ext) 요소 vs 내부 요소 평균 중요도")
    print(f"     all_ext {np.mean(imp_ext):.3f}x  ({len(imp_ext)}개)")
    print(f"     내부    {np.mean(imp_int):.3f}x  ({len(imp_int)}개)")
    ratio = np.mean(imp_ext) / max(np.mean(imp_int), 1e-9)
    verdict = "낮게 봄 (도메인 규칙 학습됨)" if ratio < 0.95 else (
        "차이 거의 없음" if ratio < 1.05 else "오히려 높게 봄")
    print(f"     비율 {ratio:.3f} → {verdict}")
    print()
    print("3) 비정상 예측 그래프에서 상위 k개 안에 실제 문제 요소가 있는 비율 (넓은 정의)")
    for k in (1, 3):
        h, t = hits[k]
        print(f"     top-{k}: {h}/{t} = {h / max(t, 1):.3f}")


# ──────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="어텐션 기반 요소 지목")
    ap.add_argument("--run", required=True, help="실행 디렉터리 (예: runs/elem_gatv2)")
    ap.add_argument("--name", default=None, help="모델 이름 하나")
    ap.add_argument("--worst", type=int, default=0, help="비정상 확신도 상위 N개")
    ap.add_argument("--errors", type=int, default=0, help="오분류 상위 N개")
    ap.add_argument("--audit", action="store_true", help="전체 어텐션 검증")
    ap.add_argument("--out", default=None, help="png 저장 경로")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu")
    run_dir = a.run if os.path.isabs(a.run) else os.path.join(paths.ROOT, a.run)

    if a.audit:
        audit(run_dir, device)
        return 0

    res = load_result(run_dir)
    args = SimpleNamespace(**res["args"])
    opts = {}
    if args.graph == "elem":
        opts = {"corner_edges": not args.no_corner_edges,
                "include_lines": not args.no_lines}
    names, labels, arrays = DS.build_cache(args.graph, opts=opts, verbose=False)
    fa = fold_assignment(names, labels, args)
    oof = np.loadtxt(os.path.join(run_dir, "oof.csv"), delimiter=",", skiprows=1)
    prob = oof[:, 2]
    y = np.array(labels)

    # 무엇을 볼지 결정
    if a.name:
        picks = [names.index(a.name)]
    elif a.errors:
        wrong = np.where((prob > 0.5) != (y == 1))[0]
        picks = list(wrong[np.argsort(-np.abs(prob[wrong] - 0.5))][: a.errors])
    elif a.worst:
        picks = list(np.argsort(-prob)[: a.worst])
    else:
        picks = [0]

    import matplotlib
    matplotlib.use("Agg")
    setup_korean_font()
    import matplotlib.pyplot as plt

    n = len(picks)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6.0 * cols, 5.2 * rows))

    by_name = {s.name: s for s in list_samples(verbose=False)}
    cache: dict[int, tuple] = {}

    for k, i in enumerate(picks, start=1):
        fold = int(fa[i])
        if fold not in cache:
            cache[fold] = load_fold_model(run_dir, fold, device)
        model, norm, _, platt = cache[fold]

        rep = MeshRep(load_raw_mesh(by_name[names[i]].bdf, by_name[names[i]].json))
        logit, alpha = attention_for(arrays[i], names[i], labels[i], model, norm, device)
        imp = relative_importance(alpha)
        p = 1.0 / (1.0 + np.exp(-(platt[0] * logit + platt[1])))

        truth = "비정상" if y[i] == 1 else "정상"
        pred = "비정상" if p > 0.5 else "정상"
        mark = "O" if (p > 0.5) == (y[i] == 1) else "X"
        title = (f"{names[i]}  [fold {fold}]\n"
                 f"실제 {truth} / 예측 {pred} (p={p:.3f}) {mark}")

        ax = fig.add_subplot(rows, cols, k, projection="3d")
        render(rep, arrays[i], imp, title, ax=ax)

        print(f"\n{'=' * 68}")
        print(f"{names[i]}  실제 {truth} / 예측 {pred} p={p:.3f}  (fold {fold} 모델)")
        print(f"  요소 {len(rep.shells)}쉘 + {len(rep.lines)}개 1D, "
              f"불량각도 요소 {sum(F.is_bad_shell(s.quality) for s in rep.shells)}개")
        print(f"  중요도 상위 쉘 (1.00x = 평균, 형상 결함 후보):")
        for line in top_elements(rep, arrays[i], imp, k=6):
            print("    " + line)
        line_top = top_line_elements(rep, imp, k=3)
        if line_top:
            print(f"  중요도 상위 1D (참고용 - 결함 판정 아님):")
            for line in line_top:
                print("    " + line)

    fig.tight_layout()
    out = a.out or os.path.join(run_dir, "explain.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    # 사내 PC 는 라벨 엑셀·데이터 파일에 암호가 걸리므로 파일을 열기 전에 해제한다.
    # (app.py 에서 import 해 쓸 때는 app.py 가 이미 해제했으므로 여기서 하지 않는다.)
    from fasoo_drm import setting_fasoo

    setting_fasoo()
    sys.exit(main())

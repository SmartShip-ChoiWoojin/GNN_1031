"""Tabular 기준선 — GNN 이 넘어야 할 선을 확정한다.

MeshRep.graph_features() 의 집계 피처 + HistGradientBoosting.
GNN 결과는 항상 이 숫자와 나란히 보고한다.

    python src/baseline.py
    python src/baseline.py --split group      # 공간블록 그룹 CV (참고용)
    python src/baseline.py --ablation         # 피처 그룹별 기여도
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

import paths
from data_index import list_samples, spatial_groups
from meshrep import GRAPH_FEATURE_NAMES, MeshRep, graph_feature_vector
from bdf_io import load_raw_mesh

CACHE_CSV = os.path.join(paths.CACHE_DIR, "graph_features.csv")

# 피처 그룹 — ablation 에서 "1D 정보와 json 정보가 집계 수준에서 얼마나 보태는가"를 본다
GROUP_SHELL = (
    "n_node", "n_shell", "n_quad", "n_tri", "tri_frac",
    "aspect_max", "aspect_mean", "aspect_p90", "skew_max", "skew_mean",
    "warp_max", "warp_mean", "taper_min", "angle_min", "angle_max",
    "n_bad_angle", "njac_min", "njac_mean", "det_ratio_min", "area_cv", "area_ratio",
    "n_orient", "dih_max", "dih_mean", "n_free_edge", "n_nonmanifold_edge",
    "deg_shell_max", "n_irregular_node",
)
GROUP_1D = ("n_bar", "n_beam", "deg_line_max", "n_dangling_all")
GROUP_JSON = ("ext_frac", "n_bad_internal", "n_bad_ext_only",
              "n_dangling_internal", "n_free_edge_cut")


def build_feature_table(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """전 학습대상 모델의 집계 피처를 계산(또는 캐시에서 로드)한다."""
    paths.ensure_dirs()
    if os.path.exists(CACHE_CSV) and not force:
        df = pd.read_csv(CACHE_CSV)
        if list(df.columns) == ["name", "cls", *GRAPH_FEATURE_NAMES]:
            # 피처는 bdf/json 에서만 나오므로 재사용하되, 라벨(cls)과 대상 목록은
            # 항상 엑셀에서 새로 읽는다. 예전에는 캐시의 옛 라벨을 그대로 써서,
            # 라벨 엑셀을 갈아끼워도 기준선이 옛 정답으로 계산됐다.
            cur = {s.name: s.cls for s in list_samples(verbose=False)}
            df = df[df["name"].isin(cur)].copy()
            df["cls"] = df["name"].map(cur).astype(int)
            missing = set(cur) - set(df["name"])
            if not missing:
                if verbose:
                    print(f"[baseline] 캐시 사용: {CACHE_CSV} ({len(df)}행, 라벨은 엑셀 기준)")
                return df.reset_index(drop=True)
            if verbose:
                print(f"[baseline] 캐시에 없는 모델 {len(missing)}개 → 재계산")
        if verbose:
            print("[baseline] 캐시 컬럼이 바뀜 → 재계산")

    samples = list_samples(verbose=verbose)
    t0 = time.time()
    rows = []
    for i, s in enumerate(samples, start=1):
        rep = MeshRep(load_raw_mesh(s.bdf, s.json))
        rows.append([s.name, s.cls, *graph_feature_vector(rep).tolist()])
        if verbose and i % 400 == 0:
            print(f"  ... {i}/{len(samples)}  ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows, columns=["name", "cls", *GRAPH_FEATURE_NAMES])
    df.to_csv(CACHE_CSV, index=False)
    if verbose:
        print(f"[baseline] 피처 계산 완료 {len(df)}행, {time.time() - t0:.1f}s → {CACHE_CSV}")
    return df


def metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob > 0.5).astype(int)
    return dict(
        auc=float(roc_auc_score(y, prob)),
        pr_auc=float(average_precision_score(y, prob)),
        acc=float(accuracy_score(y, pred)),
        f1=float(f1_score(y, pred, zero_division=0)),
    )


def report(
    y: np.ndarray, prob: np.ndarray, per_fold: list[dict], title: str
) -> dict[str, float]:
    """fold 평균과 out-of-fold 종합을 함께 출력한다.

    두 숫자가 다른 이유: fold 별 모델의 확률 스케일(캘리브레이션)이 서로 달라서,
    그것을 한데 모아 계산한 AUC 는 fold 별 AUC 평균보다 낮게 나온다.
    **모델의 판별력을 보려면 fold 평균**을, 혼동행렬·오분류 목록을 보려면 종합을 쓴다.
    """
    m = metrics(y, prob)
    cm = confusion_matrix(y, (prob > 0.5).astype(int))
    print(f"\n{title}")
    for k in ("auc", "acc", "f1"):
        v = np.array([f[k] for f in per_fold])
        print(f"  {k:6s} fold평균 {v.mean():.4f} ± {v.std():.4f}   종합 {m[k]:.4f}")
    print(f"  pr_auc fold평균 "
          f"{np.mean([f['pr_auc'] for f in per_fold]):.4f}   종합 {m['pr_auc']:.4f}")
    print("  혼동행렬 (행=실제 정상/비정상, 열=예측 정상/비정상)")
    print(f"    정상   {cm[0, 0]:5d} {cm[0, 1]:5d}")
    print(f"    비정상 {cm[1, 0]:5d} {cm[1, 1]:5d}")
    out = {f"{k}_fold": float(np.mean([f[k] for f in per_fold])) for k in m}
    out.update({f"{k}_oof": v for k, v in m.items()})
    return out


def make_cv(split: str, folds: int, y: np.ndarray, groups: np.ndarray | None):
    if split == "group":
        return StratifiedGroupKFold(folds, shuffle=True, random_state=0), groups
    return StratifiedKFold(folds, shuffle=True, random_state=0), None


def cv_predict(
    df: pd.DataFrame, cols, split: str, folds: int, groups
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(정답, out-of-fold 확률, fold 별 지표). train.py 와 같은 프로토콜로 계산한다."""
    X = df[list(cols)].to_numpy(dtype=np.float64)
    y = df["cls"].to_numpy()
    cv, g = make_cv(split, folds, y, groups)

    oof = np.full(len(y), np.nan)
    per_fold = []
    for tr, te in cv.split(X, y, groups=g):
        clf = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof[te] = p
        per_fold.append(metrics(y[te], p))
    assert not np.isnan(oof).any()
    return y, oof, per_fold


def run(df: pd.DataFrame, cols, split: str, folds: int, groups, title: str) -> dict[str, float]:
    y, oof, per_fold = cv_predict(df, cols, split, folds, groups)
    return report(y, oof, per_fold, f"{title}  (피처 {len(cols)}개, {split} {folds}-fold)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="집계 피처 + HistGradientBoosting 기준선")
    ap.add_argument("--split", choices=["random", "group"], default="random",
                    help="random=층화 랜덤(기본), group=공간블록 그룹(참고)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ablation", action="store_true", help="피처 그룹별 기여도도 출력")
    ap.add_argument("--force", action="store_true", help="피처 캐시 무시하고 재계산")
    ap.add_argument("--save-oof", default=None,
                    help="out-of-fold 확률을 csv 로 저장할 경로")
    args = ap.parse_args(argv)

    df = build_feature_table(force=args.force)

    groups = None
    if args.split == "group":
        print("[baseline] 공간 그룹 계산 중...")
        samples = list_samples(verbose=False)
        by_name = {s.name: s for s in samples}
        groups = spatial_groups([by_name[n] for n in df["name"]])
        print(f"  그룹 수: {len(set(groups))}")

    print("=" * 70)
    print("기준선: 전체 집계 피처")
    print("=" * 70)
    full = run(df, GRAPH_FEATURE_NAMES, args.split, args.folds, groups, "전체")

    if args.save_oof:
        y, oof, _ = cv_predict(df, GRAPH_FEATURE_NAMES, args.split, args.folds, groups)
        pd.DataFrame({"name": df["name"], "cls": y, "prob": oof}).to_csv(
            args.save_oof, index=False
        )
        print(f"\n[baseline] out-of-fold 확률 저장: {args.save_oof}")

    if args.ablation:
        print()
        print("=" * 70)
        print("피처 그룹 ablation")
        print("=" * 70)
        run(df, ["tri_frac"], args.split, args.folds, groups, "삼각형 비율 단독")
        run(df, GROUP_SHELL, args.split, args.folds, groups, "쉘 품질만")
        run(df, GROUP_SHELL + GROUP_1D, args.split, args.folds, groups, "쉘 + 1D")
        run(df, GROUP_1D + GROUP_JSON, args.split, args.folds, groups, "1D + json 만")

        print()
        print("=" * 70)
        print("라벨과의 상관 상위 15개")
        print("=" * 70)
        c = df[list(GRAPH_FEATURE_NAMES) + ["cls"]].corr()["cls"].drop("cls")
        for k in c.abs().sort_values(ascending=False).index[:15]:
            print(f"  {k:22s} {c[k]:+.4f}")

    print()
    print("=" * 70)
    print("기준선 확정 - train.py 의 BASELINE 상수에 이 값을 넣는다")
    print(f"  fold 평균: AUC {full['auc_fold']:.4f} / ACC {full['acc_fold']:.4f}"
          f" / F1 {full['f1_fold']:.4f}")
    print(f"  종합(OOF): AUC {full['auc_oof']:.4f} / ACC {full['acc_oof']:.4f}"
          f" / F1 {full['f1_oof']:.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    # 사내 PC 는 라벨 엑셀·데이터 파일에 암호가 걸리므로 파일을 열기 전에 해제한다.
    from fasoo_drm import setting_fasoo

    setting_fasoo()
    sys.exit(main())

"""runs/ 아래 모든 실행 결과를 한 표로 비교한다.

    python src/compare.py
    python src/compare.py --sort acc
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

import paths
from train import BASELINE


def describe(args: dict) -> str:
    """설정을 짧은 태그로."""
    bits = [args.get("graph", "?"), args.get("conv", "?")]
    if args.get("no_graph_features"):
        bits.append("집계피처off")
    if args.get("no_lines"):
        bits.append("1D없음")
    if args.get("separate_encoder"):
        bits.append("인코더분리")
    if args.get("no_corner_edges"):
        bits.append("꼭짓점엣지off")
    if args.get("ext_mask") == "hard":
        bits.append("ext하드")
    if args.get("split") == "group":
        bits.append("그룹CV")
    if args.get("seed", 0):
        bits.append(f"seed{args['seed']}")
    return " ".join(bits)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="실행 결과 비교표")
    ap.add_argument("--sort", choices=("auc", "acc", "f1"), default="auc")
    ap.add_argument("--runs", default=None, help="실행 디렉터리 glob (기본: runs/*)")
    a = ap.parse_args(argv)

    pattern = a.runs or os.path.join(paths.RUNS_DIR, "*")
    rows = []
    for d in sorted(glob.glob(pattern)):
        rj = os.path.join(d, "result.json")
        if not os.path.isfile(rj):
            continue
        with open(rj, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        fm, fs = r.get("fold_mean"), r.get("fold_std")
        if fm is None:  # 구버전 result.json
            fm = {k: float(np.mean([p[k] for p in r["per_fold"]]))
                  for k in ("auc", "acc", "f1")}
            fs = {k: float(np.std([p[k] for p in r["per_fold"]]))
                  for k in ("auc", "acc", "f1")}
        rows.append({
            "tag": r["tag"],
            "desc": describe(r["args"]),
            "auc": fm["auc"], "auc_sd": fs["auc"],
            "acc": fm["acc"], "acc_sd": fs["acc"],
            "f1": fm["f1"], "f1_sd": fs["f1"],
            "cm": r["confusion_matrix"],
        })

    if not rows:
        print(f"결과 없음: {pattern}")
        return 1

    rows.sort(key=lambda r: -r[a.sort])

    w = max(len(r["tag"]) for r in rows)
    print("=" * (w + 78))
    print(f"{'실행':<{w}}  {'AUC':>15}  {'ACC':>15}  {'F1':>15}   설정")
    print("-" * (w + 78))
    print(f"{'기준선(tabular)':<{w}}  {BASELINE['auc_fold']:>7.4f}"
          f"{'':>8}  {BASELINE['acc_fold']:>7.4f}{'':>8}  "
          f"{BASELINE['f1_fold']:>7.4f}{'':>8}   집계피처 37개 + HistGradientBoosting")
    print("-" * (w + 78))
    for r in rows:
        print(f"{r['tag']:<{w}}  "
              f"{r['auc']:.4f}±{r['auc_sd']:.4f}  "
              f"{r['acc']:.4f}±{r['acc_sd']:.4f}  "
              f"{r['f1']:.4f}±{r['f1_sd']:.4f}   {r['desc']}")
    print("=" * (w + 78))
    print("숫자는 5-fold 평균 ± 표준편차. 기준선과 같은 프로토콜(층화 랜덤 5-fold).")

    best = rows[0]
    cm = np.array(best["cm"])
    print(f"\n최고 ({a.sort} 기준): {best['tag']}  -  {best['desc']}")
    print("  혼동행렬 (out-of-fold, 행=실제, 열=예측)")
    print(f"    정상   {cm[0, 0]:5d} {cm[0, 1]:5d}   (오탐 {cm[0, 1]})")
    print(f"    비정상 {cm[1, 0]:5d} {cm[1, 1]:5d}   (미탐 {cm[1, 0]})")
    return 0


if __name__ == "__main__":
    # runs/*/result.json 도 DRM 활성 상태에서 만들어졌으면 암호가 걸려 있다.
    from fasoo_drm import setting_fasoo

    setting_fasoo()
    sys.exit(main())

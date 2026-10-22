"""여러 실행의 학습 셋 out-of-fold(CV) 결과를 나란히 — 어디서 나아졌나/나빠졌나.

    python src/rg_compare.py base aux05 aux10 --ref regen_final

각 실행에 ``rg_eval.py --split cv`` 가 먼저 돌아 있어야 한다 (eval_cv.csv).
``--ref`` 는 기존 GNN_GAE/runs 의 실행 (예: regen_final).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import rg_paths

S = lambda v: set() if (v is None or (isinstance(v, float) and np.isnan(v)) or v == "") else set(str(v).split(";"))


def stage_cat(p: set, t: set) -> str:
    if p == t:
        return "ok"
    if not p:
        return "아예 안 함"
    if p < t:
        return "덜 함"
    if p > t:
        return "더 함"
    return "위치 틀림"


def summarize(path: str) -> dict:
    e = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    r = e[e.oracle == "재현됨"]
    s1 = [stage_cat(S(a), S(b)) for a, b in zip(r.collapses, r.true_collapses)]
    need = np.array([len(S(b)) > 0 for b in r.true_collapses])
    s1 = np.array(s1)
    out = {
        "CV 정확일치": e.exact.astype(bool).mean(),
        "재현가능 정확일치": r.exact.astype(bool).mean(),
        "바뀐요소 IoU": e.change_iou.astype(float).mean(),
        "병합필요 중 1단계 정답": (s1[need] == "ok").mean(),
        "병합필요 중 아예 안 함": (s1[need] == "아예 안 함").mean(),
        "병합불필요 중 더 함": (s1[~need] == "더 함").mean(),
        "심판 비정상률": (e.p_regen.astype(float) > 0.5).mean() if "p_regen" in e else np.nan,
    }
    for k in ("연결만 변경", "절점삭제"):
        out[f"{k} 정확일치"] = e[e.kind == k].exact.astype(bool).mean()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--ref", nargs="*", default=["regen_final"])
    a = ap.parse_args()
    cols = {}
    for t in a.ref:
        cols[f"{t} (기존)"] = summarize(os.path.join(rg_paths.GAE_RUNS_DIR, t, "eval_cv.csv"))
    for t in a.tags:
        cols[t] = summarize(os.path.join(rg_paths.RUNS_DIR, f"regen_{t}", "eval_cv.csv"))
    df = pd.DataFrame(cols)
    pd.set_option("display.width", 200)
    print(df.map(lambda v: f"{v:.1%}" if isinstance(v, float) and abs(v) <= 1 else v).to_string())
    df.to_csv(os.path.join(rg_paths.RUNS_DIR, "compare_cv.csv"), encoding="utf-8-sig")


if __name__ == "__main__":
    main()

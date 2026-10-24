"""VGAE 출력 → 편집 연산 → 재생성 메쉬 (.bdf / .json).

VGAE 는 "Good 메쉬에서 어떤 절점 쌍이 이어져 있을까" (변 확률 P)와 "어떤 절점이 남을까"
(절점 확률 q)를 낸다. 이것을 그대로 .bdf 로 바꿀 수는 없어서, **예측한 Good 연결에 가장
가까워지는 편집을 하나씩 고른다.**

  점수(메쉬 상태) = Σ_{지금 메쉬의 변 (i,j)} logit P_ij  +  Σ_{남아 있는 절점 i} logit q_i

이 점수는 "지금 메쉬의 연결이 VGAE 예측과 얼마나 맞나" 의 로그우도와 같다 (상수 차이).
매 단계 가능한 편집 전부에 대해 점수가 얼마나 오르는지 계산하고, 가장 많이 오르는
편집부터 실제로 적용해 본다. 검사를 통과한 첫 편집을 적용하고, 점수를 올리는 편집이
없으면 멈춘다. 임계값은 따로 없다 (확률 0.5 가 자연스러운 기준이 된다).

  편집 1  절점 병합 r→k       r 의 변이 모두 사라지고, r 의 이웃이 k 에 붙는다. r 은 없어진다.
  편집 2  삼각형 쌍 병합      두 삼각형이 공유하던 변 하나가 사라진다 (= 엣지 제거).

실제 메쉬 수정과 검사는 GNN_GAE/src/meshedit.py 가 한다. 그래서 결과는 항상 유효하다.

``--run oracle`` 은 VGAE 대신 **진짜 Good 연결**을 넣는다. 예측이 완벽할 때 이 변환으로
몇 쌍을 정확히 재현하는지, 곧 이 방식의 상한이다.

    python src/vg_decode.py --run oracle
    python src/vg_decode.py --run vgae
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import vg_paths as vp
from meshdiff import compare
from meshedit import EditableMesh, EditError, _compact, check, collapse_pairs, merge_candidates
from vg_data import sides

MAX_STEPS = 15
EPS = 1e-6
ORACLE_LOGIT = 8.0


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p) - np.log1p(-p)


def logit_tables(d: dict, ps: np.ndarray, pk: np.ndarray):
    """(절점쌍 logit 행렬, 절점 logit) — 행·열은 d['ids'] 순서."""
    n = len(d["ids"])
    iu, ju = np.triu_indices(n, k=1)
    L = np.zeros((n, n))
    L[iu, ju] = _logit(ps)
    L = L + L.T
    return L, _logit(pk)


def oracle_tables(d: dict):
    n = len(d["ids"])
    L = np.full((n, n), -ORACLE_LOGIT)
    for i, j in d["adj_good"]:
        L[i, j] = L[j, i] = ORACLE_LOGIT
    Lq = np.where(d["keep"] > 0, ORACLE_LOGIT, -ORACLE_LOGIT)
    return L, Lq


def _collapsed_shells(shells, r: int, k: int):
    """r→k 를 적용했을 때 남는 쉘의 절점 목록 (검사 없이 위상만)."""
    out, seen = [], set()
    for s in shells:
        ns = s[2]
        if r in ns:
            ns = _compact([k if x == r else x for x in ns])
            if ns is None:
                continue
        key = frozenset(ns)
        if key in seen:
            continue
        seen.add(key)
        out.append((None, None, ns))
    return out


def regenerate(bad: EditableMesh, ids: list[int], L: np.ndarray, Lq: np.ndarray):
    """(결과 메쉬, 적용한 편집 목록, 단계별 점수 증가)."""
    idx = {n: i for i, n in enumerate(ids)}
    lv = lambda e: L[idx[e[0]], idx[e[1]]]
    m = bad.clone()
    m.log.clear()
    ops, gains = [], []
    for _ in range(MAX_STEPS):
        cur = sides(m.shells)
        cands = []
        for c in merge_candidates(m, drop_lines=True):
            if not c.ok:
                continue
            e = tuple(sorted(set(m.shells[c.si][2]) & set(m.shells[c.sj][2])))
            cands.append((-lv(e), "merge", e))
        for r, k, _ in collapse_pairs(m):
            new = sides(_collapsed_shells(m.shells, r, k))
            g = sum(lv(e) for e in new - cur) - sum(lv(e) for e in cur - new) - Lq[idx[r]]
            cands.append((g, "collapse", (r, k)))
        cands.sort(key=lambda t: -t[0])

        applied = None
        for g, kind, key in cands:
            if g <= EPS:
                break
            t = m.clone()
            try:
                if kind == "collapse":
                    t.collapse_edge(*key)
                else:
                    hit = [c for c in merge_candidates(t, drop_lines=True)
                           if tuple(sorted(set(t.shells[c.si][2]) & set(t.shells[c.sj][2]))) == key]
                    t.merge_tris(hit[0].si, hit[0].sj, drop_lines=True)
            except (EditError, IndexError):
                continue
            if not check(t, bad, regen=True).ok:     # 누적 면적 변화 등 — 원본 대비로 본다
                continue
            applied = (g, kind, key, t)
            break
        if applied is None:
            break
        g, kind, key, m = applied
        ops.append(f"{kind}:{key[0]}>{key[1]}" if kind == "collapse" else f"merge:{key[0]}-{key[1]}")
        gains.append(round(float(g), 3))
    return m, ops, gains


def main() -> None:
    import pickle

    import torch
    from torch_geometric.loader import DataLoader  # noqa: F401  (predict 가 쓴다)

    from vg_model import to_data
    from vg_train import load_model, predict

    ap = argparse.ArgumentParser(description="VGAE → 편집 → 재생성 (508쌍 전부, 학습 데이터)")
    ap.add_argument("--run", default="vgae", help="runs/<이름> 또는 oracle")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no_bdf", action="store_true", help=".bdf/.json 을 쓰지 않는다")
    a = ap.parse_args()

    with open(vp.DATA_PATH, "rb") as fh:
        D = pickle.load(fh)
    if a.limit:
        D = D[: a.limit]

    if a.run == "oracle":
        tables = [oracle_tables(d) for d in D]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, nx, ne = load_model(os.path.join(vp.RUNS_DIR, a.run), device)
        preds = predict(model, [to_data(d, nx, ne) for d in D], device)
        tables = [logit_tables(d, ps, pk) for d, (ps, pk) in zip(D, preds)]

    out_dir = os.path.join(vp.OUT_DIR, a.run)
    run_dir = os.path.join(vp.RUNS_DIR, a.run)
    os.makedirs(run_dir, exist_ok=True)
    if not a.no_bdf:
        os.makedirs(out_dir, exist_ok=True)
    rows = []
    t0 = time.time()
    for k, (d, (L, Lq)) in enumerate(zip(D, tables), 1):
        bad = EditableMesh(d["before"])
        pred, ops, gains = regenerate(bad, d["ids"], L, Lq)
        r = compare(bad, pred, d["after"])
        r.update(name=d["name"], n_ops=len(ops), n_collapse=sum(o.startswith("collapse") for o in ops),
                 n_merge=sum(o.startswith("merge") for o in ops), ops=";".join(ops),
                 gains=";".join(map(str, gains)), valid=check(pred, bad, regen=True).ok)
        rows.append(r)
        if not a.no_bdf:
            pred.write_bdf(os.path.join(out_dir, f"{d['name']}.bdf"))
            pred.write_json(os.path.join(out_dir, f"{d['name']}.json"))
        if k % 100 == 0:
            print(f"  ... {k}/{len(D)}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    cols = ["name", "exact", "exact_shells", "did_nothing", "valid", "n_ops", "n_collapse",
            "n_merge", "change_iou", "rm_prec", "rm_rec", "ops", "gains"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    # --limit 으로 일부만 돌릴 때 전체 결과(decode.csv)를 덮어쓰지 않는다
    csv_name = f"decode_limit{a.limit}.csv" if a.limit else "decode.csv"
    df.to_csv(os.path.join(run_dir, csv_name), index=False, encoding="utf-8-sig")

    print(f"\n[{a.run}] {len(df)}쌍 = 학습 데이터 (테스트 아님)")
    print(f"  정확 일치 (요소+1D+좌표)   {int(df.exact.sum())}/{len(df)} ({df.exact.mean():.1%})")
    print(f"  요소만 일치               {int(df.exact_shells.sum())}/{len(df)} ({df.exact_shells.mean():.1%})")
    print(f"  아무것도 안 함            {int(df.did_nothing.sum())}")
    print(f"  바뀐 요소 IoU 평균        {df.change_iou.mean():.3f}")
    print(f"  지운 요소 정밀도/재현율   {df.rm_prec.mean():.3f} / {df.rm_rec.mean():.3f}")
    print(f"  편집 수 평균              병합 {df.n_collapse.mean():.2f} / 삼각형 쌍 {df.n_merge.mean():.2f}")
    print(f"  검사 위반                 {int((~df.valid).sum())}")
    orc = os.path.join(vp.RUNS_DIR, "oracle", "decode.csv")
    if a.run != "oracle" and os.path.exists(orc):
        ok = set(pd.read_csv(orc).query("exact").name)
        sub = df[df.name.isin(ok)]
        print(f"  오라클이 재현하는 {len(sub)}쌍 중 정확 일치  {int(sub.exact.sum())} ({sub.exact.mean():.1%})")
    print(f"저장: {run_dir}\\{csv_name}" +("" if a.no_bdf else f", {out_dir}\\*.bdf"))


if __name__ == "__main__":
    main()

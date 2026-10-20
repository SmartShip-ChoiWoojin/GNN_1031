"""배포 방식 검증 — '다시 학습(refit)' 과 'fold 모델 평균(ensemble)' 을 학습 셋 안에서만 비교한다.

왜 필요했나: 처음 배포 모델은 CV 로 임계값을 고른 뒤 학습 셋 전부로 **다시 학습**했다.
테스트에서 절점삭제 메쉬의 병합을 평균 0.70개만 골랐는데(CV 의 fold 모델은 1.14개, 정답
1.63개), 다시 학습한 모델의 점수 눈금이 달라져 임계값이 어긋났다고 의심했다.

테스트 셋을 이미 한 번 열었으므로, 이 의심을 **테스트를 다시 쓰지 않고** 확인한다:

    학습 셋의 CV fold 하나(k)를 '가짜 테스트' 로 떼어 놓는다
    나머지 4 fold 로  (1) 안쪽 CV → 임계값 선택
                      (2-a) 4 fold 전부로 다시 학습 → k 채점        (refit)
                      (2-b) 안쪽 CV 의 fold 모델 4개 평균 → k 채점  (ensemble)
    k = 1..5 를 다 돌면 학습 셋 456쌍 모두가 한 번씩 가짜 테스트가 된다

두 방식이 같은 임계값을 쓰므로, 차이는 오직 '점수 눈금이 임계값과 맞는가' 에서 온다.

    python src/regen_nested.py --outer 1      # fold 하나 (병렬로 5개를 띄운다)
    python src/regen_nested.py --summary      # 다 끝난 뒤 합친다
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch

import gae_paths as gp
from meshdiff import compare
from meshedit import EditableMesh
from regen_data import load_samples, load_split
from regen_model import SCORERS, EnsembleScorer, RegenModel, TrainCfg, load_encoder
from regen_train import build_items, grid_eval

OUT = os.path.join(gp.RUNS_DIR, "regen_nested")


def run_outer(k: int, a) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    log = lambda m: print(f"[outer {k} {time.time() - t0:5.0f}s] {m}", flush=True)
    split = load_split()
    tr = split[split.split == "train"]
    samples = load_samples(sorted(tr.name), verbose=False)
    fold = tr.set_index("name").cv_fold
    _, norm, gopts, _ = load_encoder(TrainCfg().encoder_run, device)
    it1, it2, cands0 = build_items(samples, norm, gopts, False, states=True, verbose=False)

    fo = np.array([int(fold[s.name]) for s in samples])
    inner_ids = [j for j in sorted(set(fo)) if j != k]
    cfg1 = TrainCfg(epochs=a.epochs1, loss=a.loss)
    cfg2 = TrainCfg(epochs=a.epochs2)
    S1, S2 = SCORERS[a.scorer], SCORERS[a.scorer2]

    grid, members = {}, []
    for j in inner_ids:
        fit = [i for i in range(len(samples)) if fo[i] not in (k, j)]
        te = [i for i in range(len(samples)) if fo[i] == j]
        s1 = S1(1, cfg1, device).fit([it for i in fit for it in it1[i]])
        s2 = S2(2, cfg2, device).fit([it2[i] for i in fit])
        g = grid_eval([samples[i] for i in te], cands0, s1, s2, norm, gopts, False, "iter")
        for key, v in g.items():
            grid.setdefault(key, []).extend(v)
        members.append((s1, s2))
        log(f"안쪽 fold {j} 끝")
    best = max(grid, key=lambda x: (np.mean([r["exact"] for r in grid[x]]),
                                    np.mean([r["change_iou"] for r in grid[x]])))
    log(f"안쪽 CV 로 고른 임계값 {best}  (안쪽 OOF 정확일치 {np.mean([r['exact'] for r in grid[best]]):.1%})")

    fit = [i for i in range(len(samples)) if fo[i] != k]
    refit = RegenModel(S1(1, cfg1, device).fit([it for i in fit for it in it1[i]]),
                       S2(2, cfg2, device).fit([it2[i] for i in fit]),
                       best[0], best[1], norm, gopts, {"decode": "iter"})
    ens = RegenModel(EnsembleScorer([m[0] for m in members]), EnsembleScorer([m[1] for m in members]),
                     best[0], best[1], norm, gopts, {"decode": "iter"})
    rows = []
    for i in [i for i in range(len(samples)) if fo[i] == k]:
        s = samples[i]
        bad = EditableMesh(s.pair.before)
        for nm, model in (("refit", refit), ("ensemble", ens)):
            res = model.regenerate(bad, cands0[s.name])
            r = compare(s.pair.before, res.mesh, s.pair.after)
            r.update(name=s.name, outer=k, deploy=nm, t1=best[0], t2=best[1],
                     n_collapse=len(res.collapses), n_collapse_true=len(s.collapses),
                     n_merge=len(res.merges), n_merge_true=len(s.merges))
            rows.append(r)
    os.makedirs(OUT, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, f"outer{k}.csv"), index=False, encoding="utf-8-sig")
    for nm, g in d.groupby("deploy"):
        log(f"{nm:8s} 정확일치 {g.exact.mean():.1%}  병합 {g.n_collapse.mean():.2f}개/메쉬 "
            f"(정답 {g.n_collapse_true.mean():.2f})")


def summary() -> None:
    files = sorted(glob.glob(os.path.join(OUT, "outer*.csv")))
    d = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files])
    split = load_split().set_index("name")
    d["kind"] = d.name.map(split.kind)
    print(f"가짜 테스트 {d.name.nunique()}쌍 (outer fold {sorted(d.outer.unique())})")
    t = d.groupby("deploy").agg(exact=("exact", "mean"), iou=("change_iou", "mean"),
                                pred_collapse=("n_collapse", "mean"),
                                true_collapse=("n_collapse_true", "mean"),
                                pred_merge=("n_merge", "mean"), true_merge=("n_merge_true", "mean"))
    print(t.round(3).to_string())
    print("\n절점삭제 메쉬만:")
    x = d[d.kind == "절점삭제"]
    print(x.groupby("deploy").agg(n=("name", "size"), exact=("exact", "mean"),
                                  pred_collapse=("n_collapse", "mean"),
                                  true_collapse=("n_collapse_true", "mean")).round(3).to_string())
    p = d.pivot_table(index="name", columns="deploy", values="exact")
    both = p.dropna()
    print(f"\n같은 메쉬끼리: ensemble 만 맞힘 {int(((both.ensemble == 1) & (both.refit == 0)).sum())} / "
          f"refit 만 맞힘 {int(((both.ensemble == 0) & (both.refit == 1)).sum())}")
    d.to_csv(os.path.join(OUT, "all.csv"), index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description="refit vs ensemble — 학습 셋 안 중첩 검증")
    ap.add_argument("--outer", type=int)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--scorer", default="gnn")
    ap.add_argument("--scorer2", default="gbm")
    ap.add_argument("--loss", default="list")
    ap.add_argument("--epochs1", type=int, default=60)
    ap.add_argument("--epochs2", type=int, default=100)
    a = ap.parse_args()
    if a.summary:
        summary()
    elif a.outer:
        run_outer(a.outer, a)
    else:
        ap.error("--outer k 또는 --summary")


if __name__ == "__main__":
    main()

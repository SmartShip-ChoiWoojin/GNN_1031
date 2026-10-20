"""재생성 모델 학습 — 학습 셋(90%) 안에서만 모든 결정을 내린다.

    1. 학습 셋을 5-fold 로 나눠, 각 메쉬를 **자기를 학습에 안 쓴 모델**로 재생성한다
       (out-of-fold). fold 는 분할 파일에 묶음 단위로 미리 적혀 있다. 이걸로
         - 임계값 두 개(1단계·2단계)를 고르고
         - 학습 셋 안에서의 정직한 성능(CV)을 잰다.
    2. 배포 모델은 **그 fold 모델 5개의 평균**이다 (``--deploy ensemble``, 기본). 학습 셋
       전부로 다시 학습하면(``--deploy refit``) 점수 눈금이 달라져 CV 에서 고른 임계값이
       어긋난다 — 실제로 그래서 테스트에서 병합을 너무 적게 골랐다 (regen_nested.py 로 확인).
    3. 테스트 셋은 여기서 **한 번도 열지 않는다** → `regen_eval.py` 가 처음으로 연다.

임계값을 왜 CV 로 고르나: 실제로 쓸 때는 "몇 개를 고쳐야 하는지" 를 모른다. 개수까지
모델이 정해야 하는데, 테스트 셋에서 임계값을 고르면 그 조건이 무너져 숫자가 낙관적이 된다.

1단계 선택 방식 (``--decode``):
  iter     '다음에 합칠 병합 하나 또는 멈춤' 을 반복 (기본). 학습에는 원본 말고도
           **사람의 병합을 일부 적용한 중간 상태**를 함께 쓴다 — 남은 병합이 정답이고,
           다 적용한 상태의 정답은 '멈춤' 이다. 그래야 두 번째·세 번째 병합과 멈출 때를 배운다.
  oneshot  원본에서 후보 전부에 한 번에 점수를 매겨 임계값 이상을 묶어 적용 (처음 방식).

2단계는 학습할 때 **정답 1단계를 적용한 메쉬**로 배운다 (teacher forcing). 쓸 때는 모델이
만든 1단계 결과 위에서 돈다 — 그 차이까지 포함해 채점되도록 CV 평가는 실제 파이프라인을
그대로 돌린다.

    python src/regen_train.py --tag gnn_list --loss list         # 반복형 + 멈춤 (권장)
    python src/regen_train.py --tag gnn_iter                     # 반복형, 후보별 이진분류
    python src/regen_train.py --tag gbm_iter --scorer gbm        # 피처만 쓰는 기준선
    python src/regen_train.py --tag gnn --decode oneshot         # 처음 방식
"""

from __future__ import annotations

import argparse
import os
import time
import zlib
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

import gae_paths as gp
from meshdiff import compare
from meshedit import EditableMesh, EditError, apply_collapse_seq, apply_merges, collapse_greedy
from regen_data import load_samples, load_split
from regen_model import (SCORERS, EnsembleScorer, RegenModel, TrainCfg, cached_collapse_cands, decode_collapses,
                         load_encoder, pick_collapses, pick_merges, stage1_item, stage2_item)

#: oneshot 의 1단계 임계값 = 확률. iter 의 1단계 임계값 = 여유(최고 후보 점수 - 멈춤 점수).
GRID1 = [round(float(x), 2) for x in np.arange(0.15, 0.96, 0.05)]
GRID1_ITER = [round(float(x), 2) for x in np.arange(-2.0, 4.01, 0.25)]
GRID2 = [round(float(x), 2) for x in np.arange(0.05, 0.91, 0.05)]


def ops_str(collapses) -> str:
    """적용한 절점 병합을 csv 한 칸으로 — 화면이 이 기록만으로 결과를 재현한다."""
    return ";".join(f"{r}>{k}" for r, k in collapses)


def edges_str(edges) -> str:
    return ";".join("-".join(map(str, sorted(e))) for e in edges)


# ──────────────────────────────────────────────────────────────────────────
# 학습 표본
# ──────────────────────────────────────────────────────────────────────────
def make_states(s, n_orders: int = 2) -> list[tuple[str, EditableMesh, list]]:
    """``[(캐시 키, 메쉬, 남은 정답 병합)]`` — 원본 + 사람의 병합을 일부 적용한 중간 상태.

    사람이 병합한 순서는 알 수 없으므로 무작위 순서 ``n_orders`` 개를 따라가며 앞부분을
    적용한 상태를 모은다 (같은 집합이면 한 번만 — 적용 순서가 달라도 메쉬는 같다).
    다 적용한 상태는 남은 정답이 없으므로 '멈춤' 을 가르친다.
    """
    T = [tuple(p) for p in s.collapses]
    bad = EditableMesh(s.pair.before)
    out = [(s.name, bad, T)]
    if not T:
        return out
    # hash() 는 실행마다 바뀐다 — 재현되도록 이름의 crc 를 씨앗으로 쓴다
    rng = np.random.default_rng(zlib.crc32(s.name.encode()))
    seen = {()}
    for _ in range(n_orders):
        m = bad.clone()
        applied: list = []
        for i in rng.permutation(len(T)):
            try:
                m.collapse_edge(*T[i])
            except EditError:
                continue
            applied.append(T[i])
            key = tuple(sorted(applied))
            if key in seen:
                continue
            seen.add(key)
            tag = ".".join(f"{a}>{b}" for a, b in key)
            out.append((f"{s.name}|{tag}", m.clone(), [q for q in T if q not in applied]))
    return out


def build_items(samples, norm, gopts, ctx: bool, states: bool, verbose: bool = True):
    """(1단계 표본 [메쉬별 목록], 2단계 표본, 원본 1단계 후보). 1단계는 메쉬마다 상태 여러 개."""
    per_mesh = [make_states(s) if states else [(s.name, EditableMesh(s.pair.before), list(s.collapses))]
                for s in samples]
    flat = [(k, m) for st in per_mesh for k, m, _ in st]
    cands = cached_collapse_cands(flat, verbose=verbose)
    it1 = []
    for st in per_mesh:
        lst = []
        for k, m, remain in st:
            it = stage1_item(k, m, norm, gopts, cands[k], remain, ctx=ctx)
            # 남은 정답이 있는데 후보에 하나도 없으면(두 칸 떨어진 병합 등) '멈춤' 으로
            # 잘못 가르치게 되므로 뺀다
            if remain and it.y.sum() == 0:
                continue
            lst.append(it)
        it1.append(lst)
    it2 = [stage2_item(s.name, s.m1, norm, gopts, s.merges, ctx=ctx) for s in samples]
    return it1, it2, {s.name: cands[s.name] for s in samples}


# ──────────────────────────────────────────────────────────────────────────
# OOF 채점 — 모든 임계값 조합
# ──────────────────────────────────────────────────────────────────────────
def _finish(s, m1, done, it2, p2) -> dict:
    """1단계 결과 m1 에 2단계 임계값별로 병합을 적용해 사람과 비교한다. {t2: 결과}"""
    rows, seen = {}, {}
    for t2 in GRID2:
        edges = pick_merges(it2.cands, p2, t2)
        k2 = tuple(sorted(tuple(sorted(e)) for e in edges))
        if k2 not in seen:
            t = m1.clone()
            merged, _ = apply_merges(t, edges)
            r = compare(s.pair.before, t, s.pair.after)
            r.update(name=s.name, n_collapse=len(done), n_merge=len(merged),
                     collapse_exact=set(done) == {tuple(x) for x in s.collapses},
                     collapses=ops_str(done), merges=edges_str(merged))
            seen[k2] = r
        rows[t2] = seen[k2]
    return rows


def grid_eval(samples, cands0, s1, s2, norm, gopts, ctx: bool, decode: str) -> dict:
    """OOF 메쉬마다 모든 (임계값1, 임계값2) 조합으로 재생성하고 사람 결과와 비교한다.

    반복형은 낮은 임계값으로 한 번 끝까지 풀고 임계값마다 그 앞부분을 쓴다. 같은 1단계
    결과는 한 번만 2단계를 돌린다. 반환: {(t1, t2): [메쉬별 compare 결과]}
    """
    g1 = GRID1_ITER if decode == "iter" else GRID1
    out: dict = {(a, b): [] for a in g1 for b in GRID2}
    for s in samples:
        bad = EditableMesh(s.pair.before)
        if decode == "iter":
            seq = decode_collapses(s1, s.name, bad, norm, gopts, cands0[s.name], min_margin=g1[0])
            choice = {}
            for t1 in g1:
                k = 0
                while k < len(seq) and seq[k][1] >= t1:
                    k += 1
                choice[t1] = [p for p, _ in seq[:k]]
        else:
            it = stage1_item(s.name, bad, norm, gopts, cands0[s.name], ctx=ctx)
            p1 = s1.predict([it])[0]
            choice = {t1: pick_collapses(it.cands, p1, t1) for t1 in g1}
        memo: dict = {}
        for t1 in g1:
            key = tuple(choice[t1])
            if key not in memo:
                if decode == "iter":
                    m1, done = apply_collapse_seq(bad, choice[t1]), choice[t1]
                else:
                    m1 = bad.clone()
                    done, _ = collapse_greedy(m1, choice[t1])
                it2 = stage2_item(s.name, m1, norm, gopts, ctx=ctx)
                p2 = s2.predict([it2])[0]
                memo[key] = _finish(s, m1, done, it2, p2)
            for t2, r in memo[key].items():
                out[(t1, t2)].append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="재생성 모델 학습 (CV + 최종)")
    ap.add_argument("--tag", required=True, help="실행 이름 → runs/regen_<tag>/")
    ap.add_argument("--scorer", choices=["gbm", "gnn"], default="gnn", help="1단계 채점기")
    ap.add_argument("--scorer2", choices=["gbm", "gnn"], default=None,
                    help="2단계 채점기 (기본: --scorer 와 같음). 2단계는 후보가 학습 셋 전체에 ~950개뿐이라 "
                         "gbm 이 GNN 보다 낫게 나왔다")
    ap.add_argument("--decode", choices=["iter", "oneshot"], default="iter")
    ap.add_argument("--loss", choices=["bce", "list"], default="bce",
                    help="1단계 손실. list = 메쉬마다 [후보들, 멈춤] softmax (gnn 만)")
    ap.add_argument("--epochs1", type=int, default=60, help="1단계(절점 병합) epoch")
    ap.add_argument("--epochs2", type=int, default=100, help="2단계(삼각형 병합) epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=4.0)
    ap.add_argument("--no-pretrain", action="store_true", help="인코더를 처음부터 학습")
    ap.add_argument("--ctx", action="store_true",
                    help="문맥 피처를 붙인다 — 반대 방향 결과와의 비교, 메쉬 안 순위, 메쉬 상태")
    ap.add_argument("--train-on", choices=["all", "reproducible"], default="all",
                    help="reproducible = 정답 대본이 사람 결과를 정확히 재현하는 쌍만으로 학습 "
                         "(채점은 언제나 전부로 한다)")
    ap.add_argument("--deploy", choices=["ensemble", "refit"], default="ensemble",
                    help="배포할 모델. ensemble = CV fold 모델 5개의 평균 (임계값을 고른 바로 그 모델들). "
                         "refit = 학습 셋 전부로 다시 학습 — 점수 눈금이 달라져 임계값이 어긋날 수 있다")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="쉼표로 여러 씨앗 (예: 0,1,2). fold 마다 씨앗별로 학습해 평균한다 — 같은 설정을 "
                         "두 번 돌려도 CV 가 ±2%p 흔들려서(GPU 비결정성) 그 잡음을 평균으로 줄인다")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.loss == "list" and (a.scorer != "gnn" or a.decode != "iter"):
        ap.error("--loss list 는 --scorer gnn --decode iter 에서만 쓴다")

    device = torch.device("cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu")
    run_dir = os.path.join(gp.RUNS_DIR, f"regen_{a.tag}")
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    split = load_split()
    names = sorted(split[split.split == "train"].name)
    log(f"학습 셋 {len(names)}쌍 (테스트 {int((split.split == 'test').sum())}쌍은 열지 않음) · "
        f"채점기={a.scorer}/{a.scorer2 or a.scorer} · 1단계={a.decode}/{a.loss} · 문맥피처={a.ctx} · device={device}")

    samples = load_samples(names, verbose=False)
    _, norm, gopts, _ = load_encoder(TrainCfg().encoder_run, device)
    it1, it2, cands0 = build_items(samples, norm, gopts, a.ctx, states=a.decode == "iter")
    n_state = sum(len(x) for x in it1)
    log(f"1단계 표본 {n_state}개 (원본 {len(samples)} + 중간 상태 {n_state - len(samples)}) · "
        f"후보 {sum(it.n for x in it1 for it in x)}개 (정답 {int(sum(it.y.sum() for x in it1 for it in x))}) · "
        f"2단계 후보 {sum(it.n for it in it2)}개 (정답 {int(sum(it.y.sum() for it in it2))})")

    fit_mask = np.array([a.train_on == "all" or s.oracle_exact for s in samples])
    mk = lambda ep: TrainCfg(epochs=ep, lr=a.lr, pos_weight=a.pos_weight, seed=a.seed,
                             pretrained=not a.no_pretrain, ctx=a.ctx)
    cfg1, cfg2 = mk(a.epochs1), mk(a.epochs2)
    cfg1.loss = a.loss
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [a.seed]
    Scorer, Scorer2 = SCORERS[a.scorer], SCORERS[a.scorer2 or a.scorer]

    # ── 1. 학습 셋 안에서 5-fold ────────────────────────────────────────
    # fold 는 분할 파일에 적힌 것을 쓴다 — 같은 결함을 고친 쌍이 학습과 채점에 갈라지지
    # 않도록 묶음 단위로 미리 정해 둔 것이다 (regen_data.defect_groups).
    fold_col = split.set_index("name").cv_fold
    fold_of = np.array([int(fold_col[s.name]) for s in samples])
    n_folds = int(fold_of.max())
    grid: dict = {}
    fold_models: list = []
    for fi in range(1, n_folds + 1):
        te = np.where(fold_of == fi)[0]
        tr = [i for i in range(len(samples)) if fold_of[i] != fi and fit_mask[i]]
        log(f"fold {fi}/{n_folds}: 학습 {len(tr)} / 채점 {len(te)}")
        mods = []
        for sd in seeds:
            c1, c2 = replace(cfg1, seed=sd), replace(cfg2, seed=sd)
            mods.append((Scorer(1, c1, device).fit([it for i in tr for it in it1[i]], log),
                         Scorer2(2, c2, device).fit([it2[i] for i in tr], log)))
        if len(mods) == 1:
            s1, s2 = mods[0]
        else:
            s1 = EnsembleScorer([m[0] for m in mods])
            s2 = EnsembleScorer([m[1] for m in mods])
        g = grid_eval([samples[i] for i in te], cands0, s1, s2, norm, gopts, a.ctx, a.decode)
        for k, v in g.items():
            grid.setdefault(k, []).extend(v)
        best = max(g, key=lambda k: np.mean([r["exact"] for r in g[k]]))
        log(f"  이 fold 최고 정확일치 {np.mean([r['exact'] for r in g[best]]):.1%} @ {best}")
        fold_models.extend(mods)

    # ── 임계값 선택 — CV 정확일치 최대, 동률이면 바뀐 요소 IoU ──────────────
    table = []
    for (t1, t2), rs in grid.items():
        table.append(dict(t1=t1, t2=t2, exact=np.mean([r["exact"] for r in rs]),
                          iou=np.mean([r["change_iou"] for r in rs]),
                          collapse_exact=np.mean([r["collapse_exact"] for r in rs])))
    tab = pd.DataFrame(table).sort_values(["exact", "iou"], ascending=False)
    t1, t2 = float(tab.iloc[0].t1), float(tab.iloc[0].t2)
    cv = pd.DataFrame(grid[(t1, t2)])
    cv = cv.merge(split[["name", "cv_fold", "group", "kind", "oracle"]], on="name")
    cv.to_csv(os.path.join(run_dir, "cv_pred.csv"), index=False, encoding="utf-8-sig")
    tab.to_csv(os.path.join(run_dir, "cv_grid.csv"), index=False, encoding="utf-8-sig")

    reproducible = cv.oracle == "재현됨"
    log(f"선택한 임계값: 1단계 {t1:.2f} / 2단계 {t2:.2f}")
    log(f"CV 정확일치 {cv.exact.mean():.1%} ({int(cv.exact.sum())}/{len(cv)}) · "
        f"재현 가능한 쌍 중 {cv[reproducible].exact.mean():.1%} · "
        f"바뀐 요소 IoU {cv.change_iou.mean():.3f} · 1단계만 정확 {cv.collapse_exact.mean():.1%}")

    # ── 2. 배포 모델 ────────────────────────────────────────────────────
    allfit = [i for i in range(len(samples)) if fit_mask[i]]
    if a.deploy == "ensemble":
        log(f"배포 모델 = CV fold 모델 {len(fold_models)}개의 평균 (fold {n_folds} × 씨앗 {len(seeds)}, "
            f"다시 학습하지 않는다)")
        s1 = EnsembleScorer([m[0] for m in fold_models])
        s2 = EnsembleScorer([m[1] for m in fold_models])
    else:
        log("배포 모델 = 학습 셋 전부로 다시 학습")
        s1 = Scorer(1, cfg1, device).fit([it for i in allfit for it in it1[i]], log)
        s2 = Scorer2(2, cfg2, device).fit([it2[i] for i in allfit], log)
    meta = dict(tag=a.tag, scorer=a.scorer, scorer2=a.scorer2 or a.scorer, deploy=a.deploy,
                decode=a.decode, loss=a.loss,
                encoder_run=cfg1.encoder_run, pretrained=not a.no_pretrain, ctx=a.ctx,
                train_on=a.train_on, folds=n_folds, epochs1=a.epochs1, epochs2=a.epochs2,
                lr=a.lr, pos_weight=a.pos_weight, seed=a.seed, seeds=seeds, n_train=len(samples),
                n_fit=len(allfit), n_state=n_state,
                cv_exact=float(cv.exact.mean()), cv_iou=float(cv.change_iou.mean()),
                cv_exact_reproducible=float(cv[reproducible].exact.mean()),
                split_file=os.path.relpath(os.path.join(gp.RUNS_DIR, "regen_split.csv"), gp.ROOT))
    RegenModel(s1, s2, t1, t2, norm, gopts, meta).save(run_dir)
    log(f"저장: {run_dir}")
    log(f"테스트 셋 채점:  python src/regen_eval.py --run {os.path.relpath(run_dir, gp.ROOT)}")


if __name__ == "__main__":
    main()

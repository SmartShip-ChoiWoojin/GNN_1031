"""5-fold CV 학습 — 요소기반/절점기반 그래프를 동일 조건으로 비교한다.

    python src/train.py --graph elem --conv gine
    python src/train.py --graph node --conv gine
    python src/train.py --graph elem --conv gatv2
    python src/train.py --graph elem --conv gine --split group     # 참고용

검증 프로토콜
  - 층화 랜덤 5-fold. 모든 모델이 out-of-fold 예측을 하나씩 갖는다.
  - **fold 의 test 조각은 절대 건드리지 않는다.** early stopping 은 학습분에서 다시
    떼어낸 내부 val 로만 한다. (기존 `SHI/Ver1/Train.py:198` 은 보고용 val 로
    best 를 골라서 보고 성능이 낙관적으로 나왔다.)
  - **정규화 통계는 fold 의 학습분에서만** 적합한다 (dataset.NormalizerSet).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from torch_geometric.loader import DataLoader

import dataset as DS
import graphs
import paths
from models import build_model, count_params

#: baseline.py 로 실측한 tabular 기준선 (랜덤 5-fold, 집계 피처 37개 + HistGradientBoosting).
#: SN2592_FWD_Clean 전체(3214개 학습대상) 기준. fold 평균과 out-of-fold 종합을 따로 둔다.
BASELINE = {
    "auc_fold": 0.9421, "acc_fold": 0.8653, "f1_fold": 0.8268,
    "auc_oof": 0.9416, "acc_oof": 0.8653, "f1_oof": 0.8269,
}


def ext_index(kind: str) -> int:
    """하드 마스킹에 쓸 외부연결 플래그의 노드 피처 인덱스."""
    node_spec, _ = graphs.specs(kind)
    name = "all_ext" if kind == "elem" else "ext"
    return node_spec.index(name)


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """(정답, logit). 확률은 캘리브레이션 후에 만든다."""
    model.eval()
    ys, ls = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ls.append(model(batch).cpu().numpy())
            ys.append(batch.y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ls)


def fit_platt(logit: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """내부 val 로 Platt 스케일링 (a, b) 을 적합한다: p = sigmoid(a·logit + b).

    왜 필요한가: fold 마다 early stopping 시점이 달라 logit 스케일이 어긋난다. 그러면
    fold 별 AUC 는 좋은데 5개 fold 의 확률을 한데 모아 계산한 out-of-fold AUC 가
    떨어진다(실측: fold평균 0.9707 vs 종합 0.9522). Platt 은 단조변환이라
    **fold 별 AUC 를 바꾸지 않고** 스케일만 맞춰서 종합 지표와 0.5 임계값을 의미있게 만든다.
    val 만 쓰므로 test 누출이 없다.
    """
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(logit.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(logit: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * logit + b)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p > 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def train_one_fold(
    args,
    arrays: list,
    labels: list[int],
    names: list[str],
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    device: torch.device,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """한 fold 학습. Returns (test 정답, test 확률, 지표, 체크포인트)."""
    y_all = np.array(labels)

    # 학습분에서 내부 val 을 떼어낸다 — test 조각은 early stopping 에 쓰지 않는다
    inner_tr, inner_val = train_test_split(
        tr_idx, test_size=args.val_frac, random_state=args.seed + fold,
        stratify=y_all[tr_idx],
    )

    # 정규화 통계는 내부 학습분에서만 적합
    norm = DS.NormalizerSet(args.graph).fit(arrays, inner_tr)

    ds_tr = DS.make_dataset(arrays, labels, names, inner_tr, norm)
    ds_val = DS.make_dataset(arrays, labels, names, inner_val, norm)
    ds_te = DS.make_dataset(arrays, labels, names, te_idx, norm)

    ld_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    ld_val = DataLoader(ds_val, batch_size=args.batch_size * 4)
    ld_te = DataLoader(ds_te, batch_size=args.batch_size * 4)

    d_node, d_edge, d_graph = DS.feature_dims(args.graph)
    model = build_model(
        d_node, d_edge, d_graph, args,
        ext_mask_index=ext_index(args.graph) if args.ext_mask == "hard" else None,
    ).to(device)

    n_pos = int((y_all[inner_tr] == 1).sum())
    n_neg = int((y_all[inner_tr] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # 이진 분류용 표준 손실

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for batch in ld_tr:
            batch = batch.to(device)
            opt.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.detach().item() * batch.num_graphs
        sched.step()

        yv, lv = evaluate(model, ld_val, device)
        auc = roc_auc_score(yv, lv)
        if auc > best_auc:
            best_auc, best_epoch, bad = auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if args.verbose and epoch % 20 == 0:
            print(f"    epoch {epoch:3d}  loss {tot / len(ds_tr):.4f}  "
                  f"val AUC {auc:.4f}  (best {best_auc:.4f} @ {best_epoch})")

        if bad >= args.patience:
            if args.verbose:
                print(f"    early stop @ {epoch} (best val AUC {best_auc:.4f} @ {best_epoch})")
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    # 내부 val 로 Platt 스케일링을 적합해 fold 간 확률 스케일을 맞춘다
    yv, lv = evaluate(model, ld_val, device)
    a, b = fit_platt(lv, yv)

    y_te, l_te = evaluate(model, ld_te, device)
    p_te = apply_platt(l_te, a, b)
    m = metrics(y_te, p_te)
    m["best_val_auc"] = float(best_auc)
    m["best_epoch"] = best_epoch
    m["platt_a"], m["platt_b"] = a, b

    ckpt = {
        "model_state": best_state,
        "normalizer": norm.state_dict(),
        "args": vars(args),
        "fold": fold,
        "metrics": m,
        "platt": [a, b],
        "d_node": d_node, "d_edge": d_edge, "d_graph": d_graph,
    }
    return y_te, p_te, m, ckpt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="메쉬 정상/비정상 GNN 학습 (5-fold CV)")
    ap.add_argument("--graph", choices=graphs.GRAPH_KINDS, default="elem")
    ap.add_argument("--conv", choices=("gine", "gatv2"), default="gine")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fold 학습분에서 내부 val 로 떼어낼 비율")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=("random", "group"), default="random",
                    help="random=층화 랜덤(기본), group=공간블록 그룹(참고)")
    ap.add_argument("--ext-mask", choices=("feature", "hard"), default="feature",
                    help="feature=외부연결을 피처로 주고 학습(기본), hard=readout 에서 제외")
    ap.add_argument("--no-graph-features", action="store_true",
                    help="집계 피처 concat 을 끈다 (ablation)")
    ap.add_argument("--no-corner-edges", action="store_true",
                    help="요소기반에서 꼭짓점 인접 엣지를 뺀다 (ablation)")
    ap.add_argument("--no-lines", action="store_true",
                    help="요소기반에서 1D 요소 노드를 뺀다 (ablation)")
    ap.add_argument("--separate-encoder", action="store_true",
                    help="쉘/1D 노드를 각각 다른 인코더로 임베딩한다 (ablation)")
    ap.add_argument("--tag", default=None, help="실행 이름 (기본: graph_conv)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    ap.add_argument("--force-cache", action="store_true")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    paths.ensure_dirs()

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )
    tag = args.tag or f"{args.graph}_{args.conv}"
    run_dir = os.path.join(paths.RUNS_DIR, tag)
    os.makedirs(run_dir, exist_ok=True)

    opts: dict = {}
    if args.graph == "elem":
        opts = {
            "corner_edges": not args.no_corner_edges,
            "include_lines": not args.no_lines,
        }

    print("=" * 72)
    print(f"실행 {tag}   device={device}")
    print(f"  그래프={args.graph}  conv={args.conv}  hidden={args.hidden}  layers={args.layers}")
    print(f"  분할={args.split}  ext={args.ext_mask}  "
          f"집계피처={'off' if args.no_graph_features else 'on'}  "
          f"인코더분리={'on' if args.separate_encoder else 'off'}  옵션={opts}")
    print("=" * 72)

    names, labels, arrays = DS.build_cache(
        args.graph, opts=opts, force=args.force_cache, verbose=args.verbose
    )
    y = np.array(labels)

    groups = None
    if args.split == "group":
        from data_index import list_samples, spatial_groups
        by_name = {s.name: s for s in list_samples(verbose=False)}
        groups = spatial_groups([by_name[n] for n in names])
        print(f"[train] 공간 그룹 {len(set(groups))}개")

    if args.split == "group":
        splitter = StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed)
        folds = list(splitter.split(np.zeros(len(y)), y, groups))
    else:
        splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        folds = list(splitter.split(np.zeros(len(y)), y))

    d_node, d_edge, d_graph = DS.feature_dims(args.graph)
    probe = build_model(d_node, d_edge, d_graph, args)
    print(f"[train] 피처 {d_node}/{d_edge}/{d_graph}차원, 파라미터 {count_params(probe):,}개")
    del probe

    oof = np.full(len(y), np.nan)
    per_fold: list[dict] = []
    t0 = time.time()

    for fold, (tr_idx, te_idx) in enumerate(folds, start=1):
        print(f"\n── fold {fold}/{args.folds}  학습 {len(tr_idx)} / 평가 {len(te_idx)}")
        y_te, p_te, m, ckpt = train_one_fold(
            args=args,
            arrays=arrays,
            labels=labels,
            names=names,
            tr_idx=tr_idx,
            te_idx=te_idx,
            device=device,
            fold=fold,
        )
        oof[te_idx] = p_te
        per_fold.append(m)
        torch.save(ckpt, os.path.join(run_dir, f"fold{fold}.pt"))
        print(f"  fold {fold}: AUC {m['auc']:.4f}  ACC {m['acc']:.4f}  F1 {m['f1']:.4f}"
              f"  (내부 val AUC {m['best_val_auc']:.4f} @ epoch {m['best_epoch']})")

    # ── 종합 ──────────────────────────────────────────────────────────
    assert not np.isnan(oof).any(), "out-of-fold 예측이 비어 있는 샘플이 있음"
    overall = metrics(y, oof)
    cm = confusion_matrix(y, (oof > 0.5).astype(int))

    fold_mean = {k: float(np.mean([m[k] for m in per_fold])) for k in ("auc", "pr_auc", "acc", "f1")}
    fold_std = {k: float(np.std([m[k] for m in per_fold])) for k in fold_mean}

    print()
    print("=" * 72)
    print(f"{tag}  결과 ({len(y)}개, {time.time() - t0:.0f}s)")
    print("=" * 72)
    print("  지표      fold평균 ± 표준편차     종합(OOF)   기준선(fold)   차이")
    for k in ("auc", "acc", "f1"):
        base = BASELINE[f"{k}_fold"]
        print(f"  {k:8s} {fold_mean[k]:.4f} ± {fold_std[k]:.4f}       "
              f"{overall[k]:.4f}      {base:.4f}     {fold_mean[k] - base:+.4f}")
    print(f"  {'pr_auc':8s} {fold_mean['pr_auc']:.4f} ± {fold_std['pr_auc']:.4f}       "
          f"{overall['pr_auc']:.4f}")
    print("\n  혼동행렬 (out-of-fold, 행=실제, 열=예측)")
    print(f"    정상   {cm[0, 0]:5d} {cm[0, 1]:5d}")
    print(f"    비정상 {cm[1, 0]:5d} {cm[1, 1]:5d}")
    print("\n  fold 별 AUC: " + ", ".join(f"{m['auc']:.4f}" for m in per_fold))

    result = {
        "tag": tag,
        "args": vars(args),
        "overall": overall,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "per_fold": per_fold,
        "confusion_matrix": cm.tolist(),
        "baseline": BASELINE,
        "n_samples": int(len(y)),
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    np.savetxt(
        os.path.join(run_dir, "oof.csv"),
        np.stack([np.arange(len(y)), y, oof], axis=1),
        delimiter=",", header="idx,y,prob", comments="", fmt=["%d", "%d", "%.6f"],
    )
    with open(os.path.join(run_dir, "oof_names.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(names))
    print(f"\n저장: {run_dir}")
    return 0


if __name__ == "__main__":
    # 사내 PC 는 라벨 엑셀·데이터 파일에 암호가 걸리므로 파일을 열기 전에 해제한다.
    from fasoo_drm import setting_fasoo

    setting_fasoo()
    sys.exit(main())

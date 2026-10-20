"""1단계 수리 모델 — "어느 삼각형 쌍을 합칠 것인가" 를 GNN 으로 고른다.

과제 정의는 `task_merge.py` 참고. 요약하면:

    입력   비정상 메쉬 + 합칠 수 있는 삼각형 쌍 후보 (메쉬당 평균 2.8개)
    출력   어느 후보를 합칠 것인가 (1~5개)
    채점   Good 메쉬와 쉘 집합이 정확히 일치하는가
    기준   실제로 골라야 하는 121개에서 무작위 31.4% / 규칙 47.1%

**요소기반 그래프에서 병합 후보는 인접한 두 삼각형 '노드' 다.** 그래서 분류기가 쓰던
그래프를 그대로 쓰고, 노드 임베딩 두 개를 묶어 후보 점수를 낸다.

표본이 191개뿐이라 두 가지를 쓴다:
  - **사전학습 인코더 전이** — 심판 분류기(`deploy_10_14`)의 가중치로 시작한다.
    그 모델은 이 메쉬들을 학습에 쓴 적이 없다.
  - **메쉬 단위 5-fold** — 같은 메쉬가 학습과 채점에 동시에 들어가지 않게 한다.

    python src/merge_gnn.py                    # 5-fold 학습 + 채점
    python src/merge_gnn.py --no-pretrain      # 전이 없이
    python src/merge_gnn.py --epochs 80
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch_geometric.data import Batch

import gae_paths as gp
import dataset as DS
import features as F
import graphs
import paths as parent_paths
from explain import load_full_model
from meshedit import EditableMesh, merge_candidates
from meshrep import Edge, MeshRep
from models import mlp
from task_merge import build as build_samples

#: 후보 자체의 피처 — 그래프에는 없는, "합쳤을 때 어떻게 되는가" 정보
CAND_FEATS = ["dihedral", "q_skew", "q_aspect", "q_warp", "q_angle_min", "q_angle_max",
              "q_taper", "q_area_rel", "lines_on_edge", "tri_skew_max", "tri_aspect_max",
              "tri_angle_min"]


def cand_vector(m: EditableMesh, c, med: float) -> list[float]:
    """후보 하나를 숫자 벡터로. 길이 피처는 메쉬의 중앙 변길이로 상대화한다."""
    q = c.quality
    qi = F.shell_quality(m.corners(c.si))
    qj = F.shell_quality(m.corners(c.sj))
    return [
        c.dihedral, q.skew, q.aspect, q.warp, q.angle_min, q.angle_max, q.taper,
        q.area / (med * med + 1e-9), float(c.lines_on_edge),
        max(qi.skew, qj.skew), max(qi.aspect, qj.aspect), min(qi.angle_min, qj.angle_min),
    ]


# ──────────────────────────────────────────────────────────────────────────
class MergeScorer(nn.Module):
    """메쉬를 인코딩하고, 후보(노드 쌍)마다 점수를 낸다."""

    def __init__(self, encoder, hidden: int, d_cand: int, dropout: float = 0.3):
        super().__init__()
        self.enc = encoder
        # 후보는 **순서 없는 쌍**이다. (i,j) 와 (j,i) 가 같은 점수가 나오도록
        # 합과 절대차만 쓴다 — concat 을 쓰면 순서에 따라 답이 달라진다.
        self.head = mlp(2 * hidden + d_cand, hidden, hidden // 2, 1, dropout=dropout)

    def forward(self, data, pairs: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        x = self.enc.encode(data)
        hi, hj = x[pairs[:, 0]], x[pairs[:, 1]]
        z = torch.cat([hi + hj, (hi - hj).abs(), cand], dim=-1)
        return self.head(z).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────
def prepare(samples, norm, args, device):
    """표본을 (PyG Data, 후보 노드쌍, 후보피처, 정답, 메타) 로 바꾼다."""
    opts = {}
    if args.graph == "elem":
        opts = {"corner_edges": not args.no_corner_edges, "include_lines": not args.no_lines}
    out = []
    for s in samples:
        rep = MeshRep(s.mesh.to_raw())
        ga = graphs.build(rep, args.graph, **opts)
        d = DS.to_pyg(ga, 0, s.name, norm)
        med = rep.median_edge_length
        pairs, cf, y, edges = [], [], [], []
        for c in s.cands:
            # 요소기반 그래프는 쉘을 0..n_shell-1 로 먼저 깔므로 쉘 인덱스 = 노드 인덱스
            pairs.append([c.si, c.sj])
            cf.append(cand_vector(s.mesh, c, med))
            e = s.cand_edge(c)
            edges.append(e)
            y.append(1.0 if e in s.truth else 0.0)
        out.append(dict(
            name=s.name, data=d,
            pairs=torch.tensor(pairs, dtype=torch.long),
            cand=torch.tensor(cf, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            edges=edges, cands=s.cands, truth=s.truth, n_truth=s.n_truth,
            n_choice=math.comb(len(s.cands), s.n_truth) if len(s.cands) >= s.n_truth else 1,
        ))
    return out


def normalize_cand(items, idx_fit):
    """후보 피처를 학습 fold 통계로 표준화한다 (fold 밖 정보가 새지 않도록)."""
    M = torch.cat([items[i]["cand"] for i in idx_fit], dim=0)
    mu, sd = M.mean(0, keepdim=True), M.std(0, keepdim=True).clamp_min(1e-6)
    for it in items:
        it["cand_n"] = (it["cand"] - mu) / sd
    return mu, sd


def pick(scores: np.ndarray, cands, k: int | None = None, thr: float = 0.5):
    """점수 순으로 고르되 **삼각형이 겹치는 후보는 같이 못 쓴다**.

    규칙 베이스라인이 여러 개를 골라야 하는 구간에서 0% 로 무너진 이유가 이것이다 —
    한 삼각형이 여러 후보에 동시에 들어가는데 상위 k개를 그냥 집으면 서로 충돌한다.
    """
    order = np.argsort(-scores)
    used, out = set(), []
    for i in order:
        if k is not None and len(out) >= k:
            break
        if k is None and scores[i] < thr:
            break
        si, sj = cands[i].si, cands[i].sj
        if si in used or sj in used:
            continue
        used |= {si, sj}
        out.append(i)
    return out


def collate(items, idx, device):
    """메쉬 여러 개를 한 배치로 묶는다.

    묶지 않으면 후보가 1개인 메쉬에서 head 의 BatchNorm 이 터진다(배치 1). 속도도
    훨씬 빠르다. PyG 가 노드를 이어 붙이면서 번호를 밀므로, 후보 노드쌍도 그 메쉬의
    시작 위치(ptr)만큼 더해 줘야 한다.
    """
    ds = [items[i]["data"] for i in idx]
    b = Batch.from_data_list(ds).to(device)
    off = b.ptr[:-1]
    pairs = torch.cat([items[i]["pairs"] + int(off[k]) for k, i in enumerate(idx)], 0)
    cand = torch.cat([items[i]["cand_n"] for i in idx], 0)
    y = torch.cat([items[i]["y"] for i in idx], 0)
    sizes = [len(items[i]["y"]) for i in idx]
    return b, pairs.to(device), cand.to(device), y.to(device), sizes


def raw_scores(items, idx, model, device) -> list[np.ndarray]:
    """메쉬별 후보 점수(확률)."""
    model.eval()
    with torch.no_grad():
        b, pairs, cand, _, sizes = collate(items, list(idx), device)
        allv = torch.sigmoid(model(b, pairs, cand)).cpu().numpy()
    out, pos = [], 0
    for n in sizes:
        out.append(allv[pos: pos + n])
        pos += n
    return out


def tune_threshold(items, idx, model, device) -> float:
    """**학습 fold 에서** 임계값을 고른다.

    실제로 쓸 때는 "몇 개를 합칠지" 를 모른다. 그 개수까지 모델이 정해야 하는데,
    임계값을 채점 데이터에서 고르면 그 조건이 무너져 숫자가 낙관적이 된다.
    """
    sc = raw_scores(items, idx, model, device)
    best, best_thr = -1, 0.5
    for thr in np.arange(0.05, 0.96, 0.05):
        ok = 0
        for k, i in enumerate(idx):
            it = items[i]
            sel = pick(sc[k], it["cands"], k=None, thr=float(thr))
            ok += {it["edges"][j] for j in sel} == it["truth"]
        if ok > best:
            best, best_thr = ok, float(thr)
    return best_thr


def evaluate(items, idx, model, device, use_truth_k: bool, thr: float = 0.5) -> dict:
    """채점. ``use_truth_k`` 면 사람이 합친 개수를 알려주고 고르게 한다(낙관적)."""
    ex, hit1, n_real, ex_real, n_pick = 0, 0, 0, 0, 0
    sc = raw_scores(items, idx, model, device)
    for k, i in enumerate(idx):
        it = items[i]
        s = sc[k]
        sel = pick(s, it["cands"], k=it["n_truth"] if use_truth_k else None, thr=thr)
        good = {it["edges"][j] for j in sel} == it["truth"]
        ex += good
        n_pick += len(sel)
        hit1 += it["edges"][int(np.argmax(s))] in it["truth"]
        if it["n_choice"] > 1:
            n_real += 1
            ex_real += good
    return dict(exact=ex / max(1, len(idx)), top1=hit1 / max(1, len(idx)),
                exact_real=ex_real / max(1, n_real), n_real=n_real,
                n_pick=n_pick / max(1, len(idx)), thr=thr)


def main() -> None:
    ap = argparse.ArgumentParser(description="병합 후보 선택 GNN")
    ap.add_argument("--run", default=None, help="전이할 분류기 (기본: deploy_10_14)")
    ap.add_argument("--no-pretrain", action="store_true", help="인코더를 처음부터 학습")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc-lr-scale", type=float, default=0.1,
                    help="사전학습 인코더의 학습률 배율 (너무 크면 전이가 깨진다)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-pred", default=None,
                    help="out-of-fold 예측을 csv 로 저장 (기본: out/merge_pred.csv). "
                         "뷰어(app_merge.py)가 이 파일을 읽는다.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu")
    run = a.run or os.path.join(parent_paths.RUNS_TRAIN, "deploy_10_14")
    base, norm, cargs, _ = load_full_model(run, device)
    print(f"[merge_gnn] 전이 원본: {run} · device={device} · graph={cargs.graph}")

    print("표본을 만드는 중...")
    S = build_samples()
    items = prepare(S, norm, cargs, device)
    # 정답이 후보에 다 들어 있는 메쉬만 학습/채점한다 (열거가 못 담은 것은 과제 밖)
    items = [it for it in items if it["truth"] <= set(it["edges"]) and it["n_truth"]]
    print(f"  메쉬 {len(items)}개 · 후보 {sum(len(it['y']) for it in items)}개 · "
          f"정답 {int(sum(it['y'].sum() for it in items))}개")

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(items))
    folds = np.array_split(order, a.folds)

    agg, pred_rows = [], []
    for fi, te in enumerate(folds, 1):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != fi - 1])
        normalize_cand(items, tr)

        torch.manual_seed(a.seed + fi)
        if a.no_pretrain:
            enc = type(base)(
                d_node=base.node_enc_shell[0].in_features if base.separate_encoder
                else base.node_enc[0].in_features,
                d_edge=base.edge_enc[0].in_features, d_graph=1,
                conv=cargs.conv, hidden=base.hidden, layers=len(base.convs),
                dropout=cargs.dropout, heads=cargs.heads,
                use_graph_features=False, separate_encoder=base.separate_encoder,
            ).to(device)
        else:
            enc = load_full_model(run, device)[0]
        model = MergeScorer(enc, base.hidden, len(CAND_FEATS), cargs.dropout).to(device)

        opt = torch.optim.AdamW([
            {"params": model.enc.parameters(), "lr": a.lr * (1.0 if a.no_pretrain else a.enc_lr_scale)},
            {"params": model.head.parameters(), "lr": a.lr},
        ], weight_decay=1e-4)

        for ep in range(a.epochs):
            model.train()
            rng.shuffle(tr)
            for st in range(0, len(tr), a.batch_size):
                chunk = list(tr[st: st + a.batch_size])
                if len(chunk) < 2:
                    continue   # BatchNorm 은 배치 1 에서 못 돈다
                b, pairs, cand, y, _ = collate(items, chunk, device)
                s = model(b, pairs, cand)
                loss = Fnn.binary_cross_entropy_with_logits(s, y)
                opt.zero_grad(); loss.backward(); opt.step()

        # 임계값은 **학습 fold 에서만** 고른다. 채점 데이터로 고르면 "개수를 스스로
        # 정한다" 는 조건이 무너져 숫자가 낙관적이 된다.
        thr = tune_threshold(items, tr, model, device)
        # 채점 fold 의 후보 점수를 모아 둔다 — 각 메쉬는 자기를 학습에 쓰지 않은
        # 모델이 매긴 점수만 갖게 된다 (out-of-fold).
        for i in te:
            it = items[i]
            sc = raw_scores(items, [i], model, device)[0]
            sel = set(pick(sc, it["cands"], k=None, thr=thr))
            for j, e in enumerate(it["edges"]):
                pred_rows.append(dict(
                    name=it["name"], fold=fi,
                    edge=f"{min(e)}-{max(e)}",
                    si=it["cands"][j].si, sj=it["cands"][j].sj,
                    score=float(sc[j]), picked=int(j in sel),
                    truth=int(e in it["truth"]), thr=thr,
                    n_cand=len(it["edges"]), n_truth=it["n_truth"],
                    n_choice=it["n_choice"]))
        rk = evaluate(items, te, model, device, use_truth_k=True)
        rt = evaluate(items, te, model, device, use_truth_k=False, thr=thr)
        agg.append((rk, rt))
        print(f"  fold {fi}: [개수 알려줌] 전체 {rk['exact']:.1%} 실제선택 {rk['exact_real']:.1%}"
              f"  |  [스스로 정함 thr={thr:.2f}] 전체 {rt['exact']:.1%} "
              f"실제선택 {rt['exact_real']:.1%} (평균 {rt['n_pick']:.2f}개 선택)")

    print(f"\n── {a.folds}-fold 평균 ──")
    print(f"  {'':32s}{'개수 알려줌':>16s}{'스스로 정함':>18s}")
    for k, lab in (("exact", "정확히 일치 (전체)"), ("top1", "상위1 적중"),
                   ("exact_real", "정확히 일치 (실제 선택분)")):
        a1 = np.array([x[0][k] for x in agg])
        a2 = np.array([x[1][k] for x in agg])
        print(f"  {lab:30s} {a1.mean():7.1%} ±{a1.std():4.1%}  {a2.mean():9.1%} ±{a2.std():4.1%}")
    tp = np.array([x[1]["n_pick"] for x in agg])
    tt = float(np.mean([it["n_truth"] for it in items]))
    print(f"  {'메쉬당 선택 개수':30s} {tt:7.2f} (정답)  {tp.mean():9.2f}")
    print(f"\n  비교 기준 — 무작위 31.4% / 규칙 47.1% (실제로 골라야 하는 121개 기준)")

    import pandas as pd

    gp.ensure_dirs()
    out = a.save_pred or os.path.join(gp.OUT_DIR, "merge_pred.csv")
    pd.DataFrame(pred_rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n예측 저장: {out}  (후보 {len(pred_rows)}개)")
    print("  화면으로 보려면:  streamlit run GNN_GAE/src/app_merge.py")


if __name__ == "__main__":
    main()

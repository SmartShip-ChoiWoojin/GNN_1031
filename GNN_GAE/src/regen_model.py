"""재생성 모델 — 후보에 점수를 매기는 두 개의 채점기와, 그것으로 메쉬를 다시 만드는 파이프라인.

    원본(Bad)
      │  1단계: 절점 병합 — "다음에 합칠 것 하나, 또는 멈춤" 을 반복 (iter)
      │         또는 후보 전부에 한 번에 점수를 매겨 임계값 이상을 묶어 적용 (oneshot)
      ▼
    중간 메쉬
      │  2단계: 삼각형 쌍 후보 ~2개에 점수 → 임계값 이상을 겹치지 않게 적용
      ▼
    재생성 결과

1단계를 왜 반복형으로 바꿨나 (oneshot 의 CV 를 쪼개 본 결과):
  - 절점 병합이 2개 이상인 메쉬에서 정답 개수를 못 채웠다 (2개짜리에 평균 1.4개).
    첫 병합을 하고 나면 두 번째 자리의 모양이 바뀌는데, 한 번에 점수를 매기면 그걸 못 본다.
  - "이 메쉬에 병합이 필요한가" 를 임계값 하나로 가르기 어려웠다. 병합이 필요 없는 메쉬의
    최고 점수(상위 25% 가 0.41 이상)와 정답 병합의 점수(하위 25% 가 0.08 이하)가 겹친다.
그래서 반복형은 매 단계 후보들과 **'멈춤'** 을 한 줄에 세워 고르게 하고(listwise), 하나를
적용한 뒤 메쉬를 다시 보고 다음을 고른다.

채점기는 두 종류를 같은 틀로 둔다 — 비교해야 GNN 이 값을 하는지 알 수 있다.

  ``gnn``  부모의 요소기반 그래프 + 심판 분류기(`deploy_10_14`) 인코더를 이어받아 미세조정.
           요소 임베딩 → 절점 임베딩(붙은 요소의 평균·최대) → 후보 = [remove, keep, 피처].
           2단계 후보 = [두 삼각형 임베딩의 합, 차의 절댓값, 피처] — 순서 없는 쌍이라 대칭.
           ``loss="list"`` 면 '멈춤' 점수를 메쉬 임베딩(요소 평균·최대 + 메쉬 상태)으로 낸다.
  ``gbm``  후보 피처만 쓰는 gradient boosting. 그래프 문맥 없이 어디까지 되나의 기준선.

인코더를 심판에서 이어받는 이유: 학습 표본이 456쌍뿐이고, 그 모델은 "어디가 비정상인가" 를
4000개 메쉬로 이미 배웠다. 수정 쌍 508개의 원본은 그 학습에서 빠져 있다(train_judge.py).
단, 그래서 심판으로 채점할 때 완전한 남은 아니다 — 평가(regen_eval.py)의 주 지표는
사람 결과와의 일치이고 심판 확률은 보조로만 쓴다.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

import gae_paths as gp
import paths as parent_paths
from meshedit import (REGEN_AREA_TOL, EditableMesh, EditError, apply_collapse_seq,  # noqa: F401
                      collapse_greedy, replay_ops)
from models import mlp
from meshedit import apply_merges
from regen_feats import (COLLAPSE_CTX_FEATS, COLLAPSE_FEATS, MERGE_CTX_FEATS, MERGE_FEATS,
                         MESH_FEATS, CollapseCands, MergeCands, _mesh_row, collapse_candidates,
                         merge_cands, mesh_graph, with_context)

#: 인코더를 이어받을 분류기. 수정 쌍 원본을 학습에서 뺀 심판이다.
DEFAULT_ENCODER_RUN = os.path.join(parent_paths.RUNS_TRAIN, "deploy_10_14")

#: 메쉬 하나에서 고를 절점 병합의 상한. 사람은 최대 7개였다 (508쌍). 임계값이 낮게
#: 잡혀도 메쉬를 통째로 무너뜨리지 않게 하는 안전장치다.
MAX_COLLAPSES = 10


# ──────────────────────────────────────────────────────────────────────────
# 피처 정규화
# ──────────────────────────────────────────────────────────────────────────
class FeatNorm:
    """부호 보존 log1p 뒤 z-score. 통계는 **학습 데이터에서만** 적합한다.

    aspect·면적비처럼 꼬리가 수십 배로 긴 피처와 각도(0~180)가 섞여 있다. log1p 로 꼬리를
    누르고 나서 표준화하면 한 규칙으로 다 다룰 수 있다.
    """

    def __init__(self, mu=None, sd=None):
        self.mu, self.sd = mu, sd

    @staticmethod
    def _t(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, X: np.ndarray) -> "FeatNorm":
        T = self._t(X)
        self.mu = T.mean(0)
        self.sd = np.maximum(T.std(0), 1e-6)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((self._t(X) - self.mu) / self.sd).astype(np.float32)

    def state(self) -> dict:
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> "FeatNorm":
        return cls(np.array(s["mu"], np.float32), np.array(s["sd"], np.float32))


# ──────────────────────────────────────────────────────────────────────────
# 학습/추론 단위
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Item:
    """메쉬(또는 중간 상태) 하나 × 단계 하나. 후보와 (있으면) 정답."""

    name: str
    stage: int
    g: object                  # regen_feats.MeshGraph
    cands: object              # CollapseCands | MergeCands
    y: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    #: 메쉬 상태 (MESH_FEATS) — '멈춤' 점수의 입력
    gfeat: np.ndarray = field(default_factory=lambda: np.zeros(len(MESH_FEATS), np.float32))

    @property
    def n(self) -> int:
        return len(self.cands.pairs)


def feat_dim(stage: int, ctx: bool) -> int:
    if stage == 1:
        return len(COLLAPSE_FEATS) + (len(COLLAPSE_CTX_FEATS) if ctx else 0)
    return len(MERGE_FEATS) + (len(MERGE_CTX_FEATS) if ctx else 0)


def stage1_item(name: str, mesh: EditableMesh, norm, gopts: dict,
                cands: CollapseCands | None = None, truth=None, ctx: bool = False) -> Item:
    cands = cands if cands is not None else collapse_candidates(mesh)
    gfeat = np.array(_mesh_row(mesh, len(cands.pairs)), np.float32)
    if ctx:
        cands = with_context(mesh, cands, 1)
    g = mesh_graph(mesh, norm, gopts)
    want = {tuple(p) for p in (truth or [])}
    y = np.array([1.0 if p in want else 0.0 for p in cands.pairs], np.float32)
    return Item(name, 1, g, cands, y, gfeat)


def stage2_item(name: str, mesh: EditableMesh, norm, gopts: dict, truth=None,
                ctx: bool = False) -> Item:
    cands = merge_cands(mesh)
    gfeat = np.array(_mesh_row(mesh, len(cands.pairs)), np.float32)
    if ctx:
        cands = with_context(mesh, cands, 2)
    g = mesh_graph(mesh, norm, gopts)
    want = set(truth or [])
    y = np.array([1.0 if e in want else 0.0 for e in cands.edges], np.float32)
    return Item(name, 2, g, cands, y, gfeat)


@dataclass
class _Batch:
    b: object
    inc: torch.Tensor
    n_vertex: int
    cand: torch.Tensor
    feats: torch.Tensor
    y: torch.Tensor
    cand_graph: torch.Tensor     # 후보마다 몇 번째 메쉬인가
    gfeat: torch.Tensor          # (메쉬 수, len(MESH_FEATS)) 정규화 끝


def _collate(items: list[Item], fn: FeatNorm, fng: FeatNorm | None, device) -> _Batch:
    """메쉬 여러 개를 한 배치로. 요소·절점 번호를 메쉬마다 밀어 준다."""
    b = Batch.from_data_list([it.g.data for it in items]).to(device)
    eoff = b.ptr[:-1].tolist()
    voff, v = [], 0
    for it in items:
        voff.append(v)
        v += len(it.g.vertex_ids)
    inc = torch.cat([it.g.inc + torch.tensor([[voff[i]], [eoff[i]]]) for i, it in enumerate(items)], 1)
    feats = torch.from_numpy(np.concatenate([fn(it.cands.feats) for it in items], 0))
    y = torch.from_numpy(np.concatenate([it.y for it in items], 0))
    if items[0].stage == 1:
        cand = []
        for i, it in enumerate(items):
            vix = {vid: j for j, vid in enumerate(it.g.vertex_ids)}
            cand += [[vix[r] + voff[i], vix[k] + voff[i]] for r, k in it.cands.pairs]
    else:
        cand = [[si + eoff[i], sj + eoff[i]] for i, it in enumerate(items) for si, sj in it.cands.pairs]
    cand = torch.tensor(cand, dtype=torch.long).reshape(-1, 2)
    cg = torch.tensor([i for i, it in enumerate(items) for _ in range(it.n)], dtype=torch.long)
    G = np.stack([it.gfeat for it in items], 0)
    gfeat = torch.from_numpy(fng(G) if fng is not None else G.astype(np.float32))
    return _Batch(b, inc.to(device), v, cand.to(device), feats.to(device), y.to(device),
                  cg.to(device), gfeat.to(device))


# ──────────────────────────────────────────────────────────────────────────
# 신경망
# ──────────────────────────────────────────────────────────────────────────
class CandidateNet(nn.Module):
    """요소 임베딩 → 후보 점수 (+ 선택적으로 메쉬마다 '멈춤' 점수).

    1단계(절점 쌍, 방향 있음) / 2단계(삼각형 쌍, 대칭) 공용.
    """

    def __init__(self, enc, hidden: int, stage: int, d_feat: int, dropout: float = 0.3,
                 stop: bool = False):
        super().__init__()
        self.enc, self.stage = enc, stage
        # 1단계: [remove 절점(평균·최대), keep 절점(평균·최대), 피처] — 방향이 결과를 가르므로
        #        remove/keep 자리를 고정해 비대칭을 그대로 둔다.
        # 2단계: [합, 차의 절댓값, 피처] — (i,j) 와 (j,i) 가 같은 점수가 나와야 한다.
        d_in = (4 * hidden if stage == 1 else 2 * hidden) + d_feat
        self.head = mlp(d_in, hidden, hidden // 2, 1, dropout=dropout)
        # '멈춤' — 메쉬 전체를 보고 "더 고칠 게 없다" 의 점수. 후보 점수와 같은 줄에 세운다.
        self.stop_head = mlp(2 * hidden + len(MESH_FEATS), hidden, hidden // 2, 1,
                             dropout=dropout) if stop else None

    def forward(self, bt: _Batch):
        H = self.enc.encode(bt.b)
        if self.stage == 1:
            hv = H[bt.inc[1]]
            V = torch.cat([scatter(hv, bt.inc[0], dim=0, dim_size=bt.n_vertex, reduce="mean"),
                           scatter(hv, bt.inc[0], dim=0, dim_size=bt.n_vertex, reduce="max")], 1)
            z = torch.cat([V[bt.cand[:, 0]], V[bt.cand[:, 1]], bt.feats], 1)
        else:
            hi, hj = H[bt.cand[:, 0]], H[bt.cand[:, 1]]
            z = torch.cat([hi + hj, (hi - hj).abs(), bt.feats], 1)
        s = self.head(z).squeeze(-1)
        if self.stop_head is None:
            return s, None
        B = bt.gfeat.size(0)
        pooled = torch.cat([scatter(H, bt.b.batch, dim=0, dim_size=B, reduce="mean"),
                            scatter(H, bt.b.batch, dim=0, dim_size=B, reduce="max"), bt.gfeat], 1)
        return s, self.stop_head(pooled).squeeze(-1)


def _seg_lse(x: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    """구간별 log-sum-exp."""
    m = scatter(x.detach(), idx, dim=0, dim_size=n, reduce="max")
    s = scatter((x - m[idx]).exp(), idx, dim=0, dim_size=n, reduce="sum")
    return s.log() + m


def listwise_loss(s, stop, y, cg, B: int) -> torch.Tensor:
    """메쉬마다 [후보들, 멈춤] 위의 softmax. 정답 = 남은 사람 병합 중 아무거나(여럿이면 합), 없으면 멈춤.

    -inf 대신 큰 음수로 가린다 — torch.where 의 안 쓰는 쪽에 log(0) 이 있으면 기울기가 NaN 이 된다.
    """
    dev = s.device
    allv = torch.cat([s, stop])
    idx = torch.cat([cg, torch.arange(B, device=dev)])
    has_pos = scatter(y, cg, dim=0, dim_size=B, reduce="sum") > 0
    mask = torch.cat([y > 0, ~has_pos])
    tgt = _seg_lse(torch.where(mask, allv, torch.full_like(allv, -1e4)), idx, B)
    return (_seg_lse(allv, idx, B) - tgt).mean()


def load_encoder(run: str, device, pretrained: bool = True):
    """(인코더, 그래프 정규화기, 그래프 옵션, hidden). ``pretrained=False`` 면 같은 구조를 새로 만든다."""
    from explain import load_full_model

    base, norm, cargs, _ = load_full_model(run, device)
    gopts = {"corner_edges": not cargs.no_corner_edges, "include_lines": not cargs.no_lines}
    if not pretrained:
        from models import MeshGNN

        d_node = (base.node_enc_shell[0] if base.separate_encoder else base.node_enc[0]).in_features
        base = MeshGNN(d_node=d_node, d_edge=base.edge_enc[0].in_features, d_graph=1,
                       conv=cargs.conv, hidden=base.hidden, layers=len(base.convs),
                       dropout=cargs.dropout, heads=cargs.heads, use_graph_features=False,
                       separate_encoder=base.separate_encoder).to(device)
    # 분류 head·pooling 은 쓰지 않는다 — encode() 까지만 쓴다
    return base, norm, gopts, base.hidden


# ──────────────────────────────────────────────────────────────────────────
# 채점기 — 같은 인터페이스
#   fit(items)        학습
#   predict(items)    후보별 확률 (oneshot 선택용)
#   scores(items)     [(후보별 점수, 멈춤 점수)] — 반복형 선택은 '최고 후보 - 멈춤' 의 차로 판단한다.
#                     BCE 로 배운 채점기는 멈춤 점수가 0 이다 (= 확률 0.5 기준의 logit).
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class TrainCfg:
    epochs: int = 60
    lr: float = 1e-3
    enc_lr_scale: float = 0.1
    batch_size: int = 16
    pos_weight: float = 4.0
    dropout: float = 0.3
    weight_decay: float = 1e-4
    seed: int = 0
    pretrained: bool = True
    encoder_run: str = DEFAULT_ENCODER_RUN
    #: 문맥 피처(반대 방향 비교·메쉬 안 순위·메쉬 상태)를 붙이나 (regen_feats.with_context)
    ctx: bool = False
    #: "bce" = 후보마다 독립 이진분류 / "list" = 메쉬마다 [후보들, 멈춤] softmax (1단계용)
    loss: str = "bce"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


class GNNScorer:
    kind = "gnn"

    def __init__(self, stage: int, cfg: TrainCfg, device):
        self.stage, self.cfg, self.device = stage, cfg, device
        self.net: CandidateNet | None = None
        self.fn: FeatNorm | None = None
        self.fng: FeatNorm | None = None

    def _build(self, pretrained: bool) -> None:
        enc, _, _, hidden = load_encoder(self.cfg.encoder_run, self.device, pretrained)
        self.net = CandidateNet(enc, hidden, self.stage, feat_dim(self.stage, self.cfg.ctx),
                                self.cfg.dropout, stop=self.cfg.loss == "list").to(self.device)

    def fit(self, items: list[Item], log=None) -> "GNNScorer":
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self._build(cfg.pretrained)
        items = [it for it in items if it.n > 0]
        self.fn = FeatNorm().fit(np.concatenate([it.cands.feats for it in items], 0))
        self.fng = FeatNorm().fit(np.stack([it.gfeat for it in items], 0))
        enc_lr = cfg.lr * (cfg.enc_lr_scale if cfg.pretrained else 1.0)
        heads = list(self.net.head.parameters()) + (
            list(self.net.stop_head.parameters()) if self.net.stop_head is not None else [])
        opt = torch.optim.AdamW([
            {"params": self.net.enc.parameters(), "lr": enc_lr},
            {"params": heads, "lr": cfg.lr},
        ], weight_decay=cfg.weight_decay)
        pw = torch.tensor(cfg.pos_weight, device=self.device)
        idx = np.arange(len(items))
        for ep in range(cfg.epochs):
            self.net.train()
            rng.shuffle(idx)
            tot = 0.0
            for st in range(0, len(idx), cfg.batch_size):
                chunk = [items[i] for i in idx[st: st + cfg.batch_size]]
                if len(chunk) < 2:
                    continue   # BatchNorm 은 배치 1 에서 못 돈다
                bt = _collate(chunk, self.fn, self.fng, self.device)
                if len(bt.y) < 2:
                    continue
                s, stop = self.net(bt)
                if cfg.loss == "list":
                    loss = listwise_loss(s, stop, bt.y, bt.cand_graph, len(chunk))
                else:
                    loss = Fnn.binary_cross_entropy_with_logits(s, bt.y, pos_weight=pw)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(chunk)
            if log and (ep + 1) % 20 == 0:
                log(f"      [{self.stage}단계] epoch {ep + 1}/{cfg.epochs}  loss {tot / len(items):.4f}")
        return self

    @torch.no_grad()
    def scores(self, items: list[Item]) -> list[tuple[np.ndarray, float]]:
        self.net.eval()
        out: list = [(np.zeros(0, np.float32), 0.0)] * len(items)
        live = [i for i, it in enumerate(items) if it.n > 0]
        for st in range(0, len(live), 32):
            ids = live[st: st + 32]
            chunk = [items[i] for i in ids]
            bt = _collate(chunk, self.fn, self.fng, self.device)
            s, stop = self.net(bt)
            s = s.cpu().numpy()
            stop = stop.cpu().numpy() if stop is not None else np.zeros(len(chunk), np.float32)
            pos = 0
            for j, (i, it) in enumerate(zip(ids, chunk)):
                out[i] = (s[pos: pos + it.n], float(stop[j]))
                pos += it.n
        return out

    def predict(self, items: list[Item]) -> list[np.ndarray]:
        return [1.0 / (1.0 + np.exp(-(s - stop))) for s, stop in self.scores(items)]

    def state(self) -> dict:
        return {"kind": self.kind, "stage": self.stage, "cfg": self.cfg.__dict__,
                "net": self.net.state_dict(), "fn": self.fn.state(),
                "fng": self.fng.state() if self.fng is not None else None}

    @classmethod
    def from_state(cls, s: dict, device) -> "GNNScorer":
        obj = cls(s["stage"], TrainCfg(**s["cfg"]), device)
        obj._build(pretrained=True)
        obj.net.load_state_dict(s["net"])
        obj.net.eval()
        obj.fn = FeatNorm.from_state(s["fn"])
        obj.fng = FeatNorm.from_state(s["fng"]) if s.get("fng") else None
        return obj


class GBMScorer:
    """후보 피처만 쓰는 기준선 (그래프 문맥 없음)."""

    kind = "gbm"

    def __init__(self, stage: int, cfg: TrainCfg, device=None):
        self.stage, self.cfg = stage, cfg
        self.clf = None

    def fit(self, items: list[Item], log=None) -> "GBMScorer":
        from sklearn.ensemble import HistGradientBoostingClassifier

        X = np.concatenate([it.cands.feats for it in items if it.n], 0)
        y = np.concatenate([it.y for it in items if it.n], 0)
        w = np.where(y > 0, self.cfg.pos_weight, 1.0)
        self.clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                  random_state=self.cfg.seed)
        self.clf.fit(X, y, sample_weight=w)
        return self

    def predict(self, items: list[Item]) -> list[np.ndarray]:
        return [self.clf.predict_proba(it.cands.feats)[:, 1].astype(np.float32) if it.n
                else np.zeros(0, np.float32) for it in items]

    def scores(self, items: list[Item]) -> list[tuple[np.ndarray, float]]:
        return [(_logit(p), 0.0) for p in self.predict(items)]

    def state(self) -> dict:
        return {"kind": self.kind, "stage": self.stage, "cfg": self.cfg.__dict__,
                "clf": pickle.dumps(self.clf)}

    @classmethod
    def from_state(cls, s: dict, device=None) -> "GBMScorer":
        obj = cls(s["stage"], TrainCfg(**s["cfg"]))
        obj.clf = pickle.loads(s["clf"])
        return obj


class EnsembleScorer:
    """같은 설정으로 학습한 채점기 여러 개(CV 의 fold 모델들)의 평균.

    왜 이걸 배포하나: 임계값은 **fold 모델들의** out-of-fold 점수로 골랐다. 학습 셋 전부로
    다시 학습한 모델은 점수의 눈금이 달라져서(실측: 테스트의 절점삭제 메쉬에서 병합을 평균
    0.70개만 골랐다 — CV 의 fold 모델은 1.14개, 정답 1.63개) 그 임계값이 맞지 않는다.
    fold 모델들을 그대로 평균하면 임계값을 고른 바로 그 모델들의 눈금이 유지된다.
    점수(후보 점수·멈춤 점수)를 평균하고, 확률은 확률끼리 평균한다.
    """

    kind = "ens"

    def __init__(self, members: list):
        self.members = members
        self.stage = members[0].stage
        self.cfg = members[0].cfg

    def scores(self, items: list[Item]) -> list[tuple[np.ndarray, float]]:
        outs = [m.scores(items) for m in self.members]
        return [(np.mean([o[i][0] for o in outs], axis=0).astype(np.float32),
                 float(np.mean([o[i][1] for o in outs]))) for i in range(len(items))]

    def predict(self, items: list[Item]) -> list[np.ndarray]:
        ps = [m.predict(items) for m in self.members]
        return [np.mean([p[i] for p in ps], axis=0).astype(np.float32) for i in range(len(items))]

    def state(self) -> dict:
        return {"kind": self.kind, "stage": self.stage, "members": [m.state() for m in self.members]}

    @classmethod
    def from_state(cls, s: dict, device=None) -> "EnsembleScorer":
        return cls([SCORERS[ms["kind"]].from_state(ms, device) for ms in s["members"]])


SCORERS = {"gnn": GNNScorer, "gbm": GBMScorer, "ens": EnsembleScorer}


# ──────────────────────────────────────────────────────────────────────────
# 선택 규칙 — 점수를 실제 편집으로
# ──────────────────────────────────────────────────────────────────────────
def pick_collapses(cands: CollapseCands, p: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """(oneshot) 임계값 이상을 점수 순으로 (최대 ``MAX_COLLAPSES``). 남길지는 ``collapse_greedy`` 가 정한다."""
    order = np.argsort(-p, kind="stable")
    return [cands.pairs[i] for i in order if p[i] >= thr][:MAX_COLLAPSES]


def decode_collapses(s1, name: str, bad: EditableMesh, norm, gopts: dict,
                     cands0: CollapseCands | None = None, min_margin: float = -np.inf,
                     max_steps: int = MAX_COLLAPSES) -> list[tuple[tuple[int, int], float]]:
    """(iter) '최고 후보 하나를 합친다 → 메쉬를 다시 본다' 를 반복한다. ``[(병합, 여유)]``.

    여유(margin) = 최고 후보 점수 - 멈춤 점수. ``min_margin`` 밑으로 떨어지면 멈춘다.
    고르는 **순서**는 임계값과 무관하므로, 낮은 임계값으로 한 번 끝까지 풀어 두면 어떤
    임계값의 결과든 그 앞부분(prefix)이다 — CV 에서 임계값을 고를 때 이걸 쓴다.
    후보는 매번 '지금 메쉬에서 적용 가능한 것' 이라 고른 병합은 항상 적용된다.
    """
    m = bad.clone()
    m.log.clear()
    a0 = _total_area(bad)
    seq: list = []
    c = cands0
    for _ in range(max_steps):
        it = stage1_item(name, m, norm, gopts, c, ctx=s1.cfg.ctx)
        if it.n == 0:
            break
        s, stop = s1.scores([it])[0]
        picked = None
        for i in np.argsort(-s, kind="stable"):
            margin = float(s[i] - stop)
            if margin < min_margin:
                break
            t = m.clone()
            t.collapse_edge(*it.cands.pairs[i])
            # 면적 가드는 **원본 대비 누적**으로 본다. 병합 하나하나는 5% 안이어도 여러 번
            # 쌓이면 넘을 수 있다 (실측 `_0231`: 병합 4번에 +6.3%). 넘으면 다음 후보로.
            if abs(_total_area(t) - a0) / a0 > REGEN_AREA_TOL:
                continue
            picked = (tuple(it.cands.pairs[i]), margin, t)
            break
        if picked is None:
            break
        pair, margin, m = picked
        seq.append((pair, margin))
        c = None
    return seq


def _total_area(m: EditableMesh) -> float:
    return float(sum(s.quality.area for s in m.rep.shells))


def pick_merges(cands: MergeCands, p: np.ndarray, thr: float) -> list[frozenset]:
    """임계값 이상을 점수 순으로, **삼각형이 겹치지 않게**. 한 삼각형이 여러 후보에 걸린다."""
    used, out = set(), []
    for i in np.argsort(-p, kind="stable"):
        if p[i] < thr:
            break
        si, sj = cands.pairs[i]
        if si in used or sj in used:
            continue
        used |= {si, sj}
        out.append(cands.edges[i])
    return out


@dataclass
class RegenResult:
    mesh: EditableMesh                       # 재생성 결과
    m1: EditableMesh                         # 1단계 뒤
    collapses: list = field(default_factory=list)    # 적용한 (remove, keep), 적용 순서
    merges: list = field(default_factory=list)       # 적용한 공유 변
    rejected: dict = field(default_factory=dict)     # 고른 병합 중 거부된 것 -> 사유 (oneshot)
    margins: list = field(default_factory=list)      # 병합마다 여유 (iter)
    p2: np.ndarray | None = None
    c2: MergeCands | None = None

    @property
    def absorbed(self) -> dict[int, int]:
        """{사라진 절점: 이어받은 절점} — 화면 설명용."""
        fwd = {r: k for r, k in self.collapses}
        out = {}
        for r in fwd:
            k = fwd[r]
            while k in fwd:
                k = fwd[k]
            out[r] = k
        return out


class RegenModel:
    """두 채점기 + 임계값 + 그래프 정규화. 학습이 끝난 실행 폴더 하나에 대응한다.

    ``decode`` = "iter" 면 ``thr1`` 은 여유(최고 후보 - 멈춤)의 하한이고, "oneshot" 이면 확률이다.
    """

    def __init__(self, s1, s2, thr1: float, thr2: float, norm, gopts: dict, meta: dict | None = None):
        self.s1, self.s2 = s1, s2
        self.thr1, self.thr2 = thr1, thr2
        self.norm, self.gopts = norm, gopts
        self.meta = meta or {}

    @property
    def decode(self) -> str:
        return self.meta.get("decode", "oneshot")

    def stage2(self, name: str, m1: EditableMesh):
        it = stage2_item(name, m1, self.norm, self.gopts, ctx=self.s2.cfg.ctx)
        return it.cands, self.s2.predict([it])[0]

    def regenerate(self, bad: EditableMesh, cands1: CollapseCands | None = None,
                   thr1: float | None = None, thr2: float | None = None) -> RegenResult:
        """원본 하나를 다시 만든다. 원본은 건드리지 않는다."""
        thr1 = self.thr1 if thr1 is None else thr1
        thr2 = self.thr2 if thr2 is None else thr2
        why: dict = {}
        margins: list = []
        if self.decode == "iter":
            seq = decode_collapses(self.s1, bad.name, bad, self.norm, self.gopts, cands1, thr1)
            done = [p for p, _ in seq]
            margins = [mg for _, mg in seq]
            m = apply_collapse_seq(bad, done)
        else:
            it = stage1_item(bad.name, bad, self.norm, self.gopts, cands1, ctx=self.s1.cfg.ctx)
            p1 = self.s1.predict([it])[0]
            m = bad.clone()
            m.log.clear()
            done, why = collapse_greedy(m, pick_collapses(it.cands, p1, thr1))
        m1 = m.clone()
        c2, p2 = self.stage2(bad.name, m1)
        merged, _ = apply_merges(m, pick_merges(c2, p2, thr2))
        return RegenResult(m, m1, done, merged, why, margins, p2, c2)

    # ── 저장 ─────────────────────────────────────────────────────────
    def save(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        torch.save({"s1": self.s1.state(), "s2": self.s2.state()}, os.path.join(run_dir, "model.pt"))
        meta = dict(self.meta, thr1=self.thr1, thr2=self.thr2, gopts=self.gopts)
        with open(os.path.join(run_dir, "model.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, run_dir: str, device=None) -> "RegenModel":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(run_dir, "model.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        st = torch.load(os.path.join(run_dir, "model.pt"), map_location=device, weights_only=False)
        s1 = SCORERS[st["s1"]["kind"]].from_state(st["s1"], device)
        s2 = SCORERS[st["s2"]["kind"]].from_state(st["s2"], device)
        _, norm, gopts, _ = load_encoder(meta.get("encoder_run", DEFAULT_ENCODER_RUN), device)
        return cls(s1, s2, meta["thr1"], meta["thr2"], norm, gopts, meta)


# ──────────────────────────────────────────────────────────────────────────
# 1단계 후보 캐시 — 미리보기를 계산하느라 메쉬당 0.5초 걸린다
# ──────────────────────────────────────────────────────────────────────────
def _feat_version() -> str:
    """1단계 후보·피처를 만드는 **함수들의 코드**가 바뀌면 캐시를 버린다.

    파일 통째로 해시하면 무관한 수정(예: .bdf 내려받기 함수 추가)에도 수백 개를 다시
    계산하느라 5분을 쓴다. 그래서 결과에 영향을 주는 함수의 소스만 본다.
    """
    import hashlib
    import inspect

    import features
    import meshedit
    import regen_feats

    fns = [regen_feats.collapse_candidates, regen_feats._region, regen_feats._vertex_table,
           meshedit.collapse_pairs, meshedit._compact, meshedit._convex_quad, meshedit._relation,
           meshedit._edge_count, meshedit.EditableMesh.collapse_many,
           meshedit.EditableMesh.collapse_edge, meshedit.EditableMesh.clone,
           features.shell_quality, features.area_normal]
    h = hashlib.sha1()
    for f in fns:
        h.update(inspect.getsource(f).encode())
    h.update(repr(regen_feats.COLLAPSE_FEATS).encode())
    h.update(repr(meshedit.REGEN_AREA_TOL).encode())
    return h.hexdigest()[:10]


def cached_collapse_cands(names_meshes: list[tuple[str, EditableMesh]], verbose: bool = True
                          ) -> dict[str, CollapseCands]:
    """{키: 1단계 후보}. 키는 메쉬 이름 — 중간 상태는 ``이름|적용한 병합`` 으로 부른다."""
    gp.ensure_dirs()
    path = os.path.join(gp.CACHE_DIR, f"regen_c1_{_feat_version()}.pkl")
    store: dict = {}
    if os.path.exists(path):
        with open(path, "rb") as fh:
            store = pickle.load(fh)
    n_new = 0
    for i, (nm, m) in enumerate(names_meshes, 1):
        if nm not in store:
            store[nm] = collapse_candidates(m)
            n_new += 1
            if verbose and n_new % 50 == 0:
                print(f"    1단계 후보 계산 {n_new}개 ...", flush=True)
    if n_new:
        # 임시 파일에 쓰고 바꿔 끼운다 — 학습 여러 개를 동시에 돌릴 때 반쯤 쓴 파일을
        # 다른 프로세스가 읽지 않도록
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(store, fh)
        os.replace(tmp, path)
    return {nm: store[nm] for nm, _ in names_meshes}

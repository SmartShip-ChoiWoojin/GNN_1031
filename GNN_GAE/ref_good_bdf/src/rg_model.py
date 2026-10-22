"""good 을 참고하는 보조 과제 — 1단계(절점 병합) GNN 채점기에 붙인다.

기존 학습은 good 에서 **"후보 중 어느 것이 사람이 한 병합인가"** 하나만 뽑아 쓴다.
메쉬 하나에 정답 후보는 1~3개뿐이라 supervision 이 sparse 하다. good 을 원본(또는 중간 상태)과 직접
비교하면 요소·절점마다 dense label 이 더 나온다:

    절점 라벨  이 절점의 **자리**가 good 에 없나     (병합으로 사라질 자리)
    요소 라벨  이 요소(꼭짓점 좌표 집합)가 good 에 없나 (지워지거나 모양이 바뀔 요소)

두 라벨을 맞히는 auxiliary head 를 인코더 위에 따로 달아 후보 손실과 함께 학습한다
(multi-task). 인코더가 "good 과 다를 곳" 을 직접 배우게 해서 후보 logit 과 STOP logit 이
그 표현을 쓰게 하려는 것이다.

**good 은 학습 때만 쓴다.** auxiliary head 는 추론에서 쓰지 않고, 후보 logit 과 STOP logit 은 기존과
똑같이 원본만 보고 낸다. 그래서 저장된 모델은 기존 파이프라인(regen_eval, 화면)에서 그대로 돈다.
``aux_w=0`` 이면 기존 GNNScorer 와 같은 학습이다 (공정한 비교 기준).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as Fnn
from torch_geometric.utils import scatter

import rg_paths  # noqa: F401
import regen_model as RM
from meshdiff import pkey, signature
from models import mlp
from regen_model import GNNScorer, TrainCfg, _collate, listwise_loss


# ──────────────────────────────────────────────────────────────────────────
# good 에서 뽑는 라벨
# ──────────────────────────────────────────────────────────────────────────
def aux_labels(m, good_sig, vertex_ids: list[int], n_shell: int,
               include_lines: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(절점 라벨, 요소 라벨). 순서는 ``regen_feats.mesh_graph`` 와 같다.

    절점: ``vertex_ids`` 순서 / 요소: 쉘(m.shells 순서) 다음 1D(m.lines 순서).
    번호가 아니라 **좌표**로 비교한다 — 사람은 남길 번호를 일정하게 고르지 않는다(meshdiff).
    """
    yv = np.array([0.0 if pkey(m.coords[v]) in good_sig.nodes else 1.0 for v in vertex_ids],
                  np.float32)
    key = lambda ns: frozenset(pkey(m.coords[n]) for n in ns)
    ye = [0.0 if key(s[2]) in good_sig.shells else 1.0 for s in m.shells]
    if include_lines:
        ye += [0.0 if key(l[2]) in good_sig.lines else 1.0 for l in m.lines]
    assert len(ye) >= n_shell
    return yv, np.array(ye, np.float32)


# ──────────────────────────────────────────────────────────────────────────
# 신경망 — 기존 CandidateNet + auxiliary head 두 개
# ──────────────────────────────────────────────────────────────────────────
class AuxCandidateNet(RM.CandidateNet):
    def __init__(self, enc, hidden: int, stage: int, d_feat: int, dropout: float = 0.3,
                 stop: bool = False):
        super().__init__(enc, hidden, stage, d_feat, dropout, stop)
        self.aux_v = mlp(2 * hidden, hidden, 1, dropout=dropout)   # 절점: [평균, 최대]
        self.aux_e = mlp(hidden, hidden, 1, dropout=dropout)       # 요소

    def forward_all(self, bt):
        """(후보 logit, STOP logit, 절점 aux logit, 요소 aux logit) — 학습 때만 쓴다."""
        H = self.enc.encode(bt.b)
        hv = H[bt.inc[1]]
        V = torch.cat([scatter(hv, bt.inc[0], dim=0, dim_size=bt.n_vertex, reduce="mean"),
                       scatter(hv, bt.inc[0], dim=0, dim_size=bt.n_vertex, reduce="max")], 1)
        z = torch.cat([V[bt.cand[:, 0]], V[bt.cand[:, 1]], bt.feats], 1)
        s = self.head(z).squeeze(-1)
        stop = None
        if self.stop_head is not None:
            B = bt.gfeat.size(0)
            pooled = torch.cat([scatter(H, bt.b.batch, dim=0, dim_size=B, reduce="mean"),
                                scatter(H, bt.b.batch, dim=0, dim_size=B, reduce="max"),
                                bt.gfeat], 1)
            stop = self.stop_head(pooled).squeeze(-1)
        return s, stop, self.aux_v(V).squeeze(-1), self.aux_e(H).squeeze(-1)


@dataclass
class AuxCfg(TrainCfg):
    #: 보조 손실 가중치 (절점·요소 각각에 곱한다). 0 이면 기존 학습과 같다.
    aux_w: float = 0.5
    #: 보조 라벨의 양성 가중치 — 바뀌는 절점·요소는 5~10% 뿐이다
    aux_pos_weight: float = 4.0


#: rg_train.py 가 명령줄 값으로 덮어쓴다 (regen_train.main 이 만드는 TrainCfg 에 끼워 넣을 길이 없어서)
DEFAULTS = {"aux_w": 0.5, "aux_pos_weight": 4.0}


class AuxGNNScorer(GNNScorer):
    kind = "gnn_aux"

    def __init__(self, stage: int, cfg: TrainCfg, device):
        if not isinstance(cfg, AuxCfg):
            cfg = AuxCfg(**cfg.__dict__, **DEFAULTS)
        super().__init__(stage, cfg, device)

    def _build(self, pretrained: bool) -> None:
        enc, _, _, hidden = RM.load_encoder(self.cfg.encoder_run, self.device, pretrained)
        self.net = AuxCandidateNet(enc, hidden, self.stage, RM.feat_dim(self.stage, self.cfg.ctx),
                                   self.cfg.dropout, stop=self.cfg.loss == "list").to(self.device)

    def fit(self, items, log=None) -> "AuxGNNScorer":
        cfg = self.cfg
        if self.stage != 1 or cfg.aux_w <= 0:
            return super().fit(items, log)
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self._build(cfg.pretrained)
        items = [it for it in items if it.n > 0]
        miss = [it.name for it in items if getattr(it, "aux_v", None) is None]
        if miss:
            raise RuntimeError(f"보조 라벨이 없는 표본 {len(miss)}개 (예: {miss[:2]}) — rg_train.py 로 학습한다")
        self.fn = RM.FeatNorm().fit(np.concatenate([it.cands.feats for it in items], 0))
        self.fng = RM.FeatNorm().fit(np.stack([it.gfeat for it in items], 0))
        enc_lr = cfg.lr * (cfg.enc_lr_scale if cfg.pretrained else 1.0)
        heads = (list(self.net.head.parameters()) + list(self.net.aux_v.parameters())
                 + list(self.net.aux_e.parameters())
                 + (list(self.net.stop_head.parameters()) if self.net.stop_head is not None else []))
        opt = torch.optim.AdamW([
            {"params": self.net.enc.parameters(), "lr": enc_lr},
            {"params": heads, "lr": cfg.lr},
        ], weight_decay=cfg.weight_decay)
        pw = torch.tensor(cfg.pos_weight, device=self.device)
        apw = torch.tensor(cfg.aux_pos_weight, device=self.device)
        idx = np.arange(len(items))
        for ep in range(cfg.epochs):
            self.net.train()
            rng.shuffle(idx)
            tot = tv = te = 0.0
            for st in range(0, len(idx), cfg.batch_size):
                chunk = [items[i] for i in idx[st: st + cfg.batch_size]]
                if len(chunk) < 2:
                    continue
                bt = _collate(chunk, self.fn, self.fng, self.device)
                if len(bt.y) < 2:
                    continue
                yv = torch.from_numpy(np.concatenate([it.aux_v for it in chunk])).to(self.device)
                ye = torch.from_numpy(np.concatenate([it.aux_e for it in chunk])).to(self.device)
                s, stop, lv, le = self.net.forward_all(bt)
                if cfg.loss == "list":
                    main = listwise_loss(s, stop, bt.y, bt.cand_graph, len(chunk))
                else:
                    main = Fnn.binary_cross_entropy_with_logits(s, bt.y, pos_weight=pw)
                av = Fnn.binary_cross_entropy_with_logits(lv, yv, pos_weight=apw)
                ae = Fnn.binary_cross_entropy_with_logits(le, ye, pos_weight=apw)
                loss = main + cfg.aux_w * (av + ae)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(main.detach()) * len(chunk)
                tv += float(av.detach()) * len(chunk)
                te += float(ae.detach()) * len(chunk)
            if log and (ep + 1) % 20 == 0:
                n = len(items)
                log(f"      [1단계+good] epoch {ep + 1}/{cfg.epochs}  후보 {tot / n:.4f}"
                    f"  절점 {tv / n:.4f}  요소 {te / n:.4f}")
        return self

    @classmethod
    def from_state(cls, s: dict, device) -> "AuxGNNScorer":
        obj = cls(s["stage"], AuxCfg(**s["cfg"]), device)
        obj._build(pretrained=True)
        obj.net.load_state_dict(s["net"])
        obj.net.eval()
        obj.fn = RM.FeatNorm.from_state(s["fn"])
        obj.fng = RM.FeatNorm.from_state(s["fng"]) if s.get("fng") else None
        return obj


# 저장된 모델을 기존 RegenModel.load / EnsembleScorer 가 읽을 수 있게 등록한다
RM.SCORERS["gnn_aux"] = AuxGNNScorer

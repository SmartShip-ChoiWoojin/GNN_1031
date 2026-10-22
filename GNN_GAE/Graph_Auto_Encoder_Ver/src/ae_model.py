"""Graph AutoEncoder — 정상 메쉬로만 배워서, 복원이 안 되는 요소를 '이상한 곳' 으로 본다.

두 가지 방식을 같은 틀에 둔다.

``mae`` (가린 피처 복원, GraphMAE 식)
    요소 일부(30%)의 피처를 학습되는 [MASK] 벡터로 바꾸고, 이웃만 보고 원래 피처를
    맞히게 한다. 채점 때는 요소를 하나씩 돌아가며 가리고 복원 오차를 잰다.
    → "주변을 보면 여기엔 이런 요소가 와야 하는데, 실제는 다르다" 가 점수가 된다.
    가린 요소에 붙은 엣지에서 **그 요소와 이웃을 비교한 칸**(면적비·최소내각 차·같은 타입)
    은 0 으로 지운다. 이웃의 값 + 차이 = 가린 요소의 값이라 답을 흘리기 때문이다.

``ae`` (고전 GAE / DOMINANT 식)
    인코더 → 좁은 잠재벡터(latent) → (1) 자기 피처 복원  (2) 잠재벡터 내적으로 인접 복원.
    채점 = 피처 복원 오차 + 인접 복원 오차.
    잠재벡터를 좁혀야 그대로 베끼지 못한다.

``layers=0`` 이면 메시지 전달 없이 요소 하나만 본다 — 그래프 문맥이 값을 하는지 보는 대조군
(``mae`` 에서는 볼 이웃이 없으니 의미가 없다. ``ae`` 에서만 쓴다).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch_geometric.nn import GINEConv
from torch_geometric.utils import batched_negative_sampling

import ae_paths  # noqa: F401
from graph_elem import EDGE_SPEC
from models import mlp

#: 가린 요소에 붙은 엣지에서 지울 칸 — 두 요소를 직접 비교한 값이라 답을 흘린다
LEAKY_EDGE_FEATS = ("area_log_ratio", "d_angle_min", "same_type")
_LEAKY_IDX = [EDGE_SPEC.index(n) for n in LEAKY_EDGE_FEATS]


class GINEStack(nn.Module):
    """잔차 연결 GINE 층 묶음 (부모 MeshGNN.encode 와 같은 모양)."""

    def __init__(self, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList(
            GINEConv(mlp(hidden, hidden, hidden), train_eps=True, edge_dim=hidden)
            for _ in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.dropout = dropout

    def forward(self, x, edge_index, e):
        for conv, norm in zip(self.convs, self.norms):
            h = Fnn.relu(norm(conv(x, edge_index, e)))
            x = x + Fnn.dropout(h, p=self.dropout, training=self.training)
        return x


class GraphAE(nn.Module):
    def __init__(self, d_node: int, d_edge: int, mode: str = "mae", hidden: int = 128,
                 layers: int = 3, latent: int = 16, dropout: float = 0.1,
                 mask_rate: float = 0.3, struct_w: float = 1.0):
        super().__init__()
        if mode not in ("mae", "ae"):
            raise ValueError(mode)
        self.mode, self.mask_rate, self.struct_w = mode, mask_rate, struct_w
        self.cfg = dict(d_node=d_node, d_edge=d_edge, mode=mode, hidden=hidden, layers=layers,
                        latent=latent, dropout=dropout, mask_rate=mask_rate, struct_w=struct_w)

        self.node_enc = nn.Sequential(nn.Linear(d_node, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
        self.edge_enc = nn.Sequential(nn.Linear(d_edge, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
        self.enc = GINEStack(hidden, layers, dropout)

        if mode == "mae":
            self.mask_token = nn.Parameter(torch.zeros(d_node))
            self.to_dec = nn.Linear(hidden, hidden)
            self.dec = GINEStack(hidden, 1, dropout)
            self.out = mlp(hidden, hidden, d_node)
        else:
            self.to_z = nn.Linear(hidden, latent)
            self.out = mlp(latent, hidden, d_node)

    # ──────────────────────────────────────────────────────────────────
    def _edges(self, data, masked: torch.Tensor | None):
        ea = data.edge_attr
        if masked is not None and ea.numel():
            hit = masked[data.edge_index[0]] | masked[data.edge_index[1]]
            ea = ea.clone()
            ea[hit.unsqueeze(1) & _leaky_cols(ea)] = 0.0
        return self.edge_enc(ea)

    def embed(self, data, masked: torch.Tensor | None = None) -> torch.Tensor:
        x = data.x
        if masked is not None:
            x = torch.where(masked.unsqueeze(1), self.mask_token.expand_as(x), x)
        e = self._edges(data, masked)
        return self.enc(self.node_enc(x), data.edge_index, e), e

    def reconstruct(self, data, masked: torch.Tensor | None = None):
        """(복원 피처, 잠재벡터 또는 None)."""
        h, e = self.embed(data, masked)
        if self.mode == "mae":
            h = self.to_dec(h)
            h = torch.where(masked.unsqueeze(1), torch.zeros_like(h), h)  # re-mask
            return self.out(self.dec(h, data.edge_index, e)), None
        z = self.to_z(h)
        return self.out(z), z

    # ──────────────────────────────────────────────────────────────────
    def random_mask(self, data, gen: torch.Generator | None = None) -> torch.Tensor:
        r = torch.rand(data.num_nodes, generator=gen, device="cpu").to(data.x.device)
        return r < self.mask_rate

    def loss(self, data, gen: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
        if self.mode == "mae":
            m = self.random_mask(data, gen)
            xh, _ = self.reconstruct(data, m)
            fl = feat_err(xh, data.x, data.x_valid)[m].mean()
            return fl, {"feat": float(fl.detach())}
        xh, z = self.reconstruct(data)
        fl = feat_err(xh, data.x, data.x_valid).mean()
        pos = data.edge_index
        neg = batched_negative_sampling(pos, data.batch)
        lp = (z[pos[0]] * z[pos[1]]).sum(-1)
        ln = (z[neg[0]] * z[neg[1]]).sum(-1)
        sl = 0.5 * (Fnn.binary_cross_entropy_with_logits(lp, torch.ones_like(lp))
                    + Fnn.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln)))
        return fl + self.struct_w * sl, {"feat": float(fl.detach()), "struct": float(sl.detach())}

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def node_scores(self, data, rounds: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
        """요소별 이상 점수. ``data`` 는 배치(Batch) 하나.

        mae: 요소를 G=round(1/mask_rate) 묶음으로 나눠 묶음마다 가리고 복원 → 모든 요소가
             라운드마다 정확히 한 번 가려진다. ``rounds`` 번 반복해 평균.
        ae : 피처 복원 오차와 인접 복원 오차(그래프 안 모든 쌍의 BCE 평균)를 따로 낸다.
        둘 다 ``resid`` (요소 × 피처 잔차, 해당 없는 칸은 0) 도 낸다 — 피처마다 '정상일 때
        얼마나 틀리나' 가 크게 달라서, 채점 쪽(ae_eval)에서 정상 잔차 분포로 표준화해 쓴다.
        """
        self.eval()
        v = data.x_valid.float()
        if self.mode == "mae":
            G = max(2, round(1 / self.mask_rate))
            gen = torch.Generator().manual_seed(seed)
            acc = torch.zeros(data.num_nodes, device=data.x.device)
            res = torch.zeros_like(data.x)
            for _ in range(rounds):
                grp = torch.randint(0, G, (data.num_nodes,), generator=gen).to(data.x.device)
                for g in range(G):
                    m = grp == g
                    xh, _ = self.reconstruct(data, m)
                    acc[m] += feat_err(xh, data.x, data.x_valid)[m]
                    res[m] += ((xh - data.x) * v)[m]
            return {"feat": acc / rounds, "resid": res / rounds}
        xh, z = self.reconstruct(data)
        fe = feat_err(xh, data.x, data.x_valid)
        se = torch.zeros_like(fe)
        for b in range(int(data.batch.max()) + 1):
            idx = (data.batch == b).nonzero().squeeze(1)
            n = idx.numel()
            off = int(idx[0])
            A = torch.zeros(n, n, device=z.device)
            ei = data.edge_index[:, data.batch[data.edge_index[0]] == b] - off
            A[ei[0], ei[1]] = 1.0
            zb = z[idx]
            logits = zb @ zb.T
            bce = Fnn.binary_cross_entropy_with_logits(logits, A, reduction="none")
            bce.fill_diagonal_(0.0)
            se[idx] = bce.sum(1) / max(1, n - 1)
        return {"feat": fe, "struct": se, "resid": (xh - data.x) * v}


def _leaky_cols(ea: torch.Tensor) -> torch.Tensor:
    c = torch.zeros(ea.size(1), dtype=torch.bool, device=ea.device)
    c[_LEAKY_IDX] = True
    return c.unsqueeze(0)


def feat_err(xh: torch.Tensor, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """요소별 평균제곱오차 — 그 요소에 해당하는 칸(valid)만."""
    v = valid.float()
    return ((xh - x) ** 2 * v).sum(1) / v.sum(1).clamp_min(1.0)

"""VGAE — Bad 메쉬의 절점 그래프를 넣고, Good 메쉬의 연결을 복원한다.

    인코더   GINE 3층 → 절점마다 μ, log σ (잠재 32차원)
             (--conv 로 GATv2 / Transformer / NNConv 로 바꿀 수 있다 — CONV_KINDS)
    잠재     z = μ + σ·ε   (학습 때만 노이즈, 추론 때는 z = μ)
    디코더   변 디코더    p(i, j 가 Good 에서 이어져 있다) = sigmoid(z_i · z_j + b)
             절점 디코더  p(i 가 Good 에 남는다)          = sigmoid(w · z_i + c)

일반 VGAE 는 입력 그래프를 그대로 복원하지만, 여기서는 **Good 메쉬의 연결**을 복원하도록
학습한다. 절점 디코더를 따로 둔 이유: 내적 디코더는 절점 집합이 고정이라 "이 절점이
없어진다" 를 직접 말할 수 없다.

``--skip`` 을 켜면 변 디코더에 Bad 연결을 더한다 (logit += α·A_bad). 메쉬의 변 대부분은
그대로 남으니, 모델이 "바뀌는 곳" 만 배우게 하려는 선택지다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch_geometric.data import Data
from torch_geometric.nn import VGAE, GATv2Conv, GINEConv, NNConv, TransformerConv

#: 인코더 메시지 전달 층. 넷 다 엣지 피처를 메시지에 쓴다.
#:   gine         h_i' = MLP((1+ε)h_i + Σ_j ReLU(h_j + W e_ij))          합 집계
#:   gatv2        이웃마다 주의 가중치 α_ij = softmax(a · LeakyReLU(W[h_i ‖ h_j ‖ e_ij]))
#:   transformer  그래프 트랜스포머(UniMP). α_ij = softmax((W_q h_i)·(W_k h_j + W_e e_ij)/√d),
#:                메시지 = W_v h_j + W_e e_ij
#:   nnconv       엣지 조건부 합성곱(ECC). 엣지 피처로 가중치 행렬 Θ(e_ij) 를 만들어
#:                h_i' = W h_i + mean_j Θ(e_ij) h_j
CONV_KINDS = ("gine", "gatv2", "transformer", "nnconv")
HEADS = 4


def make_conv(kind: str, hidden: int) -> nn.Module:
    if kind == "gine":
        return GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                        train_eps=True, edge_dim=hidden)
    if kind == "gatv2":
        return GATv2Conv(hidden, hidden // HEADS, heads=HEADS, concat=True, edge_dim=hidden)
    if kind == "transformer":
        return TransformerConv(hidden, hidden // HEADS, heads=HEADS, concat=True, edge_dim=hidden)
    if kind == "nnconv":
        # 엣지 벡터(hidden) → 작은 MLP → hidden×hidden 가중치 행렬
        return NNConv(hidden, hidden, nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                                    nn.Linear(32, hidden * hidden)), aggr="mean")
    raise ValueError(f"conv 는 {CONV_KINDS} 중 하나: {kind!r}")


class Encoder(nn.Module):
    def __init__(self, d_in: int, d_edge: int, hidden: int = 128, latent: int = 32,
                 layers: int = 3, dropout: float = 0.1, conv: str = "gine"):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.eenc = nn.Sequential(nn.Linear(d_edge, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList(make_conv(conv, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.mu = nn.Linear(hidden, latent)
        self.logstd = nn.Linear(hidden, latent)

    def forward(self, x, edge_index, edge_attr):
        h = self.inp(x)
        e = self.eenc(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(Fnn.relu(norm(conv(h, edge_index, e))))
        return self.mu(h), self.logstd(h).clamp(max=10.0)


class MeshVGAE(VGAE):
    """PyG ``VGAE`` (재매개화, KL) + 변 디코더 + 절점 디코더."""

    def __init__(self, d_in: int, d_edge: int, hidden: int = 128, latent: int = 32,
                 layers: int = 3, dropout: float = 0.1, skip: bool = False, conv: str = "gine"):
        super().__init__(Encoder(d_in, d_edge, hidden, latent, layers, dropout, conv))
        self.bias = nn.Parameter(torch.zeros(()))
        self.skip = skip
        self.alpha = nn.Parameter(torch.zeros(())) if skip else None
        self.keep_head = nn.Linear(latent, 1)

    def pair_logits(self, z, pair_index, a_bad=None):
        i, j = pair_index
        s = (z[i] * z[j]).sum(-1) + self.bias
        if self.skip:
            s = s + self.alpha * a_bad
        return s

    def keep_logits(self, z):
        return self.keep_head(z).squeeze(-1)

    def forward(self, b):
        z = self.encode(b.x, b.edge_index, b.edge_attr)
        return z, self.pair_logits(z, b.pair_index, b.a_bad), self.keep_logits(z)


# ──────────────────────────────────────────────────────────────────────────
class Norm:
    """z-score. 길이 비율처럼 꼬리가 긴 값이 있어 표준편차가 0 인 열만 1 로 둔다."""

    def __init__(self, mu=None, sd=None):
        self.mu, self.sd = mu, sd

    def fit(self, X: np.ndarray) -> "Norm":
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd).astype(np.float32)


def to_data(d: dict, nx: Norm, ne: Norm) -> Data:
    """쌍 하나 → PyG Data. 메쉬 안의 모든 절점 쌍(i<j)을 채점 대상으로 넣는다."""
    n = len(d["ids"])
    iu, ju = np.triu_indices(n, k=1)
    key = iu * n + ju
    good = np.zeros(n * n, dtype=np.float32)
    bad = np.zeros(n * n, dtype=np.float32)
    for i, j in d.get("adj_good", []):      # 테스트 메쉬는 정답이 없다 → 0 으로 둔다 (쓰지 않음)
        good[i * n + j] = 1.0
    for i, j in d["adj_bad"]:
        bad[i * n + j] = 1.0
    # 두 칸 이내(같은 요소의 대각, 이웃의 이웃)인데 이어지지 않은 쌍 — 내적 디코더가 가장
    # 잘 헷갈리는 곳이라 손실에서 따로 가중한다 (vg_train --w_near)
    A = bad.reshape(n, n)
    A = ((A + A.T) > 0).astype(np.float32)
    near = (((A @ A) > 0) & (A == 0)).astype(np.float32).reshape(-1)
    ei = d["edge_index"]
    return Data(
        x=torch.from_numpy(nx(d["x"])),
        edge_index=torch.from_numpy(np.hstack([ei, ei[::-1]]).copy()),
        edge_attr=torch.from_numpy(np.vstack([ne(d["edge_attr"])] * 2)),
        pair_index=torch.from_numpy(np.vstack([iu, ju]).astype(np.int64)),
        y_adj=torch.from_numpy(good[key]),
        a_bad=torch.from_numpy(bad[key]),
        near=torch.from_numpy(near[key]),
        y_keep=torch.from_numpy(d["keep"] if "keep" in d else np.ones(n, dtype=np.float32)),
        num_nodes=n,
    )

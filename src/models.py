"""GNN 아키텍처 — conv 만 교체하는 공용 backbone.

설계 근거
  - **Readout 에 max 를 반드시 넣는다.** 라벨은 "가장 나쁜 국소 결함"이 결정한다.
    mean 만 쓰면 큰 메쉬에서 불량 요소 하나가 희석된다.
  - **어텐션 pooling 의 게이트값이 곧 요소별 문제 점수**다. "어느 요소가 문제인지"
    산출물이 여기서 나온다 (explain.py).
  - **pooling 결과에 집계 피처를 concat** 한다. 이것으로 GNN 이 최소한 tabular
    기준선(AUC 0.961)의 표현력에서 출발하도록 보장한다.
  - 단일 logit + BCEWithLogits(pos_weight). 기존 구현은 가중치 없는 CrossEntropy 였다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch_geometric.nn import GATv2Conv, GINEConv
from torch_geometric.utils import scatter, softmax

CONV_KINDS = ("gine", "gatv2")


def mlp(*dims: int, dropout: float = 0.0) -> nn.Sequential:
    """Linear-BN-ReLU 를 쌓은 MLP. 마지막 층은 활성함수 없음."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class AttnPool(nn.Module):
    """어텐션 pooling. 게이트값(alpha)을 함께 반환해 요소 지목에 쓴다."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor, n_graph: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.gate(x).squeeze(-1)
        alpha = softmax(score, batch, num_nodes=x.size(0))
        out = scatter(alpha.unsqueeze(-1) * x, batch, dim=0, dim_size=n_graph, reduce="sum")
        return out, alpha


def _resolve_keep(
    mask_out: torch.Tensor, batch: torch.Tensor, n_graph: int
) -> torch.Tensor:
    """하드 마스킹용 — 남길 노드 불리언. 그래프가 통째로 비면 그 그래프는 마스킹을 해제한다."""
    keep = ~mask_out
    cnt = scatter(keep.float(), batch, dim=0, dim_size=n_graph, reduce="sum")
    empty = cnt < 0.5
    if bool(empty.any()):
        keep = keep | empty[batch]
    return keep


class MeshGNN(nn.Module):
    """메쉬 그래프 -> 비정상 확률(단일 logit).

    Args:
        d_node/d_edge/d_graph: 입력 피처 차원 (dataset.feature_dims).
        conv: "gine" 또는 "gatv2".
        hidden: 은닉 차원.
        layers: 메시지 전달 층 수.
        dropout: 층간 드롭아웃.
        heads: GATv2 헤드 수.
        ext_mask_index: 지정하면 그 노드 피처가 1인 노드를 readout 에서 제외한다
            (요소기반이면 ``all_ext``, 절점기반이면 ``ext``). ablation 용이며 기본은 None —
            외부연결 정보는 하드 마스킹보다 피처로 주고 학습시키는 쪽이 낫다.
            하드로 빼면 그 요소가 이웃에게 주는 문맥까지 사라진다.
        use_graph_features: 집계 피처 concat 사용 여부 (ablation 용).
        separate_encoder: 쉘 노드와 1D 노드를 각각 다른 인코더로 임베딩한다 (ablation 용).
            기본은 False — 하나의 ``Linear(d_node, hidden)`` 를 공유한다. 요소기반 그래프는
            쉘 전용 피처(s_*)와 1D 전용 피처(l_*)를 한 벡터에 담고 해당 없는 칸을 0으로
            채우는데(graph_elem.py), 공유 인코더는 이 두 가지 서로 다른 0-패턴을 한
            가중치 행렬로 처리해야 한다. 관찰된 실패 사례(예: SN2592_FWD_Clean_3127) —
            형상상 아무 이상 없는 CBEAM 이 같은 그래프의 어떤 쉘보다도 높은 어텐션을
            받고, 그 그래프는 실제로는 정상인데 모델이 비정상(p=0.985)으로 오판했다 —
            를 보면 인코더가 "형상이 나쁘다"가 아니라 "타입이 1D 다"라는 지름길을
            학습했을 가능성이 있다. ``separate_encoder=True`` 면 타입별로 가중치를
            분리해 이 지름길을 원천적으로 막는다. 그래프 구조(노드/엣지/순서)는
            전혀 바뀌지 않는다 — 노드를 처음 숫자 벡터로 바꾸는 단계만 타입별로 나뉜다.
    """

    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_graph: int,
        conv: str = "gine",
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.3,
        heads: int = 4,
        ext_mask_index: int | None = None,
        use_graph_features: bool = True,
        separate_encoder: bool = False,
    ):
        super().__init__()
        if conv not in CONV_KINDS:
            raise ValueError(f"conv 는 {CONV_KINDS} 중 하나여야 함: {conv!r}")

        self.conv_kind = conv
        self.hidden = hidden
        self.dropout = dropout
        self.ext_mask_index = ext_mask_index
        self.use_graph_features = use_graph_features
        self.separate_encoder = separate_encoder

        if separate_encoder:
            # 쉘/1D 를 각각 다른 가중치로 임베딩한다. 그래프 구조는 그대로이고,
            # "노드를 처음 벡터로 바꾸는 단계"만 노드 종류별로 나뉜다.
            self.node_enc = None
            self.node_enc_shell = nn.Sequential(
                nn.Linear(d_node, hidden), nn.BatchNorm1d(hidden), nn.ReLU()
            )
            self.node_enc_line = nn.Sequential(
                nn.Linear(d_node, hidden), nn.BatchNorm1d(hidden), nn.ReLU()
            )
        else:
            self.node_enc = nn.Sequential(
                nn.Linear(d_node, hidden), nn.BatchNorm1d(hidden), nn.ReLU()
            )
        # 엣지 피처는 한 번만 인코딩해서 모든 층이 공유한다 (전 층의 노드 차원이 hidden 이라
        # GINEConv/GATv2Conv 의 edge_dim 을 따로 맞출 필요가 없다)
        self.edge_enc = nn.Sequential(
            nn.Linear(d_edge, hidden), nn.BatchNorm1d(hidden), nn.ReLU()
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            if conv == "gine":
                self.convs.append(
                    GINEConv(mlp(hidden, hidden, hidden), train_eps=True, edge_dim=hidden)
                )
            else:
                if hidden % heads:
                    raise ValueError(f"hidden({hidden}) 이 heads({heads}) 로 나뉘지 않음")
                self.convs.append(
                    GATv2Conv(
                        hidden, hidden // heads, heads=heads, concat=True,
                        edge_dim=hidden, add_self_loops=False,
                    )
                )
            self.norms.append(nn.BatchNorm1d(hidden))

        self.attn = AttnPool(hidden)

        readout_dim = 3 * hidden  # mean + max + attention
        if use_graph_features:
            self.graph_enc = nn.Sequential(
                nn.Linear(d_graph, hidden), nn.BatchNorm1d(hidden), nn.ReLU()
            )
            readout_dim += hidden
        else:
            self.graph_enc = None

        self.head = mlp(readout_dim, hidden, hidden // 2, 1, dropout=dropout)

    # ──────────────────────────────────────────────────────────────────
    def _encode_nodes(self, data) -> torch.Tensor:
        """노드 원시 피처 -> hidden 차원 임베딩 (메시지 전달 이전 첫 단계)."""
        if not self.separate_encoder:
            return self.node_enc(data.x)

        # node_kind: graph_base.KIND_SHELL(0) / KIND_LINE(1). 노드 순서·개수는 그대로 두고,
        # 어느 통역사(인코더)를 태울지만 행 단위로 고른다 — 결과를 원래 자리에 도로 채운다.
        is_line = data.node_kind == 1
        is_shell = ~is_line
        x = torch.zeros(data.x.size(0), self.hidden, device=data.x.device, dtype=data.x.dtype)
        if bool(is_shell.any()):
            x[is_shell] = self.node_enc_shell(data.x[is_shell])
        if bool(is_line.any()):
            x[is_line] = self.node_enc_line(data.x[is_line])
        return x

    def encode(self, data) -> torch.Tensor:
        """메시지 전달 후의 노드 임베딩."""
        x = self._encode_nodes(data)
        e = (
            self.edge_enc(data.edge_attr)
            if data.edge_attr.numel()
            else torch.zeros((0, self.hidden), device=x.device, dtype=x.dtype)
        )
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, data.edge_index, e)
            h = norm(h)
            h = Fnn.relu(h)
            h = Fnn.dropout(h, p=self.dropout, training=self.training)
            x = x + h  # residual
        return x

    def forward(self, data, return_alpha: bool = False):
        """Returns:
            logit (B,) — 또는 ``return_alpha`` 면 (logit, alpha) 로 노드별 어텐션도 함께.
        """
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        n_graph = int(batch.max().item()) + 1 if batch.numel() else 1

        x = self.encode(data)

        if self.ext_mask_index is not None:
            keep = _resolve_keep(data.x[:, self.ext_mask_index] > 0.5, batch, n_graph)
            xp, bp = x[keep], batch[keep]
        else:
            keep, xp, bp = None, x, batch

        mean = scatter(xp, bp, dim=0, dim_size=n_graph, reduce="mean")
        mx = scatter(xp, bp, dim=0, dim_size=n_graph, reduce="max")
        att, alpha_kept = self.attn(xp, bp, n_graph)

        parts = [mean, mx, att]
        if self.graph_enc is not None:
            parts.append(self.graph_enc(data.graph_x))
        logit = self.head(torch.cat(parts, dim=1)).squeeze(-1)

        if not return_alpha:
            return logit

        if keep is None:
            alpha = alpha_kept
        else:
            alpha = torch.zeros(x.size(0), device=x.device, dtype=alpha_kept.dtype)
            alpha[keep] = alpha_kept
        return logit, alpha


def build_model(
    d_node: int, d_edge: int, d_graph: int, args, ext_mask_index: int | None = None
) -> MeshGNN:
    """argparse 결과에서 모델을 만든다 (train.py / explain.py 공용)."""
    return MeshGNN(
        d_node=d_node,
        d_edge=d_edge,
        d_graph=d_graph,
        conv=args.conv,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        heads=args.heads,
        ext_mask_index=ext_mask_index,
        use_graph_features=not args.no_graph_features,
        # getattr: 이 플래그 이전에 학습된 체크포인트의 args 에는 키가 없다
        separate_encoder=getattr(args, "separate_encoder", False),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

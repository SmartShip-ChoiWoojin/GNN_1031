"""그래프 빌더가 내보내는 공통 컨테이너.

빌더는 **정규화하지 않은 원시 피처**를 담은 :class:`GraphArrays` 를 만든다.
정규화는 dataset.py 가 학습 fold 통계로 수행한다 (featspec.Normalizer 참고).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraphArrays:
    """PyG Data 로 변환되기 전의 그래프 하나."""

    #: (n_node, d_node) 원시 노드 피처
    x: np.ndarray
    #: (2, n_edge) 방향 있는 엣지 인덱스 (무방향 관계는 양방향으로 두 번 들어간다)
    edge_index: np.ndarray
    #: (n_edge, d_edge) 원시 엣지 피처
    edge_attr: np.ndarray
    #: (d_graph,) 그래프 수준 집계 피처 (meshrep.GRAPH_FEATURE_NAMES 순서)
    graph_x: np.ndarray

    #: 노드 -> 원본 요소/절점 식별용 메타데이터. explain.py 가 어텐션을 되돌릴 때 쓴다.
    node_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    node_ref: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    #: (n_node, d_node) / (n_edge, d_edge) bool — 그 칸이 이 노드/엣지에 "해당"하는지.
    #: 예: 1D 요소 노드의 면적 칸은 False. 정규화 통계에서 제외되고 변환 후 0 이 된다.
    #: None 이면 전부 유효로 취급한다.
    x_valid: np.ndarray | None = None
    edge_valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.x = np.ascontiguousarray(self.x, dtype=np.float32)
        self.edge_index = np.ascontiguousarray(self.edge_index, dtype=np.int64)
        self.edge_attr = np.ascontiguousarray(self.edge_attr, dtype=np.float32)
        self.graph_x = np.ascontiguousarray(self.graph_x, dtype=np.float32)
        if self.x_valid is not None:
            self.x_valid = np.ascontiguousarray(self.x_valid, dtype=bool)
            if self.x_valid.shape != self.x.shape:
                raise ValueError(
                    f"x_valid 모양 불일치: {self.x_valid.shape} vs x {self.x.shape}"
                )
        if self.edge_valid is not None:
            self.edge_valid = np.ascontiguousarray(self.edge_valid, dtype=bool)
            if self.edge_valid.shape != self.edge_attr.shape:
                raise ValueError(
                    f"edge_valid 모양 불일치: {self.edge_valid.shape} vs "
                    f"edge_attr {self.edge_attr.shape}"
                )

        n = self.x.shape[0]
        if self.edge_index.size and (
            self.edge_index.min() < 0 or self.edge_index.max() >= n
        ):
            raise ValueError(
                f"edge_index 범위 이탈: [{self.edge_index.min()}, {self.edge_index.max()}], 노드 {n}개"
            )
        if self.edge_index.shape[1] != self.edge_attr.shape[0]:
            raise ValueError(
                f"엣지 수 불일치: index {self.edge_index.shape[1]} vs attr {self.edge_attr.shape[0]}"
            )
        if not np.isfinite(self.x).all():
            raise ValueError("노드 피처에 NaN/Inf")
        if self.edge_attr.size and not np.isfinite(self.edge_attr).all():
            raise ValueError("엣지 피처에 NaN/Inf")
        if not np.isfinite(self.graph_x).all():
            raise ValueError("그래프 피처에 NaN/Inf")

    @property
    def n_node(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_edge(self) -> int:
        return int(self.edge_index.shape[1])


def add_undirected(
    ei: list[list[int]],
    ea: list[list[float]],
    ev: list[np.ndarray],
    i: int,
    j: int,
    feat: list[float],
    valid: np.ndarray,
) -> None:
    """무방향 관계를 양방향 엣지 두 개로 추가한다 (피처는 대칭이라 동일)."""
    ei.append([i, j])
    ea.append(feat)
    ev.append(valid)
    ei.append([j, i])
    ea.append(feat)
    ev.append(valid)

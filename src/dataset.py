"""그래프 캐시 + fold 단위 전역 정규화.

핵심 규칙: 정규화 통계는 **학습 fold 에서만** 적합한다. 그래프별 정규화(기존 구현)도,
전체 데이터셋 통계로 한 번에 정규화하는 것(누출)도 하지 않는다.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time

import numpy as np
import torch
from torch_geometric.data import Data

import graphs
import paths
from bdf_io import load_raw_mesh
from data_index import Sample, list_samples
from featspec import Normalizer
from meshrep import GRAPH_FEATURE_NAMES, MeshRep

#: 그래프 수준 집계 피처의 스펙 — 개수/각도/비율이 섞여 있어 전부 z-score 대상
from featspec import FeatureSpec

GRAPH_SPEC = FeatureSpec(
    names=GRAPH_FEATURE_NAMES,
    kinds=tuple("linear" for _ in GRAPH_FEATURE_NAMES),
)


def _cache_key(kind: str, opts: dict) -> str:
    """빌더 옵션까지 반영한 캐시 파일 이름."""
    import graph_elem
    import graph_node

    mod = graph_elem if kind == "elem" else graph_node
    sig = repr(sorted(opts.items())) + repr(mod.NODE_SPEC.names) + repr(mod.EDGE_SPEC.names)
    h = hashlib.sha1(sig.encode()).hexdigest()[:10]
    return f"graphs_{kind}_{h}.pkl"


def build_cache(
    kind: str,
    samples: list[Sample] | None = None,
    opts: dict | None = None,
    force: bool = False,
    verbose: bool = True,
) -> tuple[list[str], list[int], list]:
    """전 모델을 그래프로 변환해 캐시한다.

    Returns:
        (names, labels, graph_arrays) — 세 리스트는 인덱스가 대응한다.
    """
    paths.ensure_dirs()
    opts = opts or {}
    path = os.path.join(paths.CACHE_DIR, _cache_key(kind, opts))

    if os.path.exists(path) and not force:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if verbose:
            print(f"[dataset] 캐시 사용: {os.path.basename(path)} ({len(blob['names'])}개)")
        return blob["names"], blob["labels"], blob["graphs"]

    samples = samples if samples is not None else list_samples(verbose=verbose)
    if verbose:
        print(f"[dataset] '{kind}' 그래프 변환 시작 (옵션 {opts})")

    t0 = time.time()
    names, labels, arrays = [], [], []
    for i, s in enumerate(samples, start=1):
        rep = MeshRep(load_raw_mesh(s.bdf, s.json))
        arrays.append(graphs.build(rep, kind, **opts))
        names.append(s.name)
        labels.append(s.cls)
        if verbose and i % 400 == 0:
            print(f"  ... {i}/{len(samples)}  ({time.time() - t0:.1f}s)")

    with open(path, "wb") as fh:
        pickle.dump({"names": names, "labels": labels, "graphs": arrays}, fh)
    if verbose:
        nn = np.array([g.n_node for g in arrays])
        ne = np.array([g.n_edge for g in arrays])
        print(f"[dataset] 완료 {len(arrays)}개, {time.time() - t0:.1f}s → {os.path.basename(path)}")
        print(f"  노드 평균 {nn.mean():.1f} (최소 {nn.min()}, 최대 {nn.max()})")
        print(f"  엣지 평균 {ne.mean():.1f} (최소 {ne.min()}, 최대 {ne.max()})")
        print(f"  노드 피처 {arrays[0].x.shape[1]}차원 / 엣지 피처 {arrays[0].edge_attr.shape[1]}차원"
              f" / 그래프 피처 {arrays[0].graph_x.shape[0]}차원")
    return names, labels, arrays


def _valid_of(ga, which: str) -> np.ndarray:
    """유효성 마스크를 꺼낸다. 빌더가 안 줬으면 전부 유효로 취급."""
    if which == "x":
        return ga.x_valid if ga.x_valid is not None else np.ones_like(ga.x, dtype=bool)
    return (
        ga.edge_valid
        if ga.edge_valid is not None
        else np.ones_like(ga.edge_attr, dtype=bool)
    )


class NormalizerSet:
    """노드/엣지/그래프 세 종류의 정규화기를 묶어 관리한다."""

    def __init__(self, kind: str):
        node_spec, edge_spec = graphs.specs(kind)
        self.kind = kind
        self.node = Normalizer(node_spec)
        self.edge = Normalizer(edge_spec)
        self.graph = Normalizer(GRAPH_SPEC)

    def fit(self, arrays: list, train_idx: np.ndarray) -> "NormalizerSet":
        """**학습 fold 의 그래프만** 쌓아서 통계를 적합한다."""
        xs = np.concatenate([arrays[i].x for i in train_idx], axis=0)
        xv = np.concatenate([_valid_of(arrays[i], "x") for i in train_idx], axis=0)
        self.node.fit(xs, xv)

        eas = [arrays[i].edge_attr for i in train_idx if arrays[i].n_edge > 0]
        evs = [_valid_of(arrays[i], "e") for i in train_idx if arrays[i].n_edge > 0]
        if not eas:
            raise RuntimeError("학습 fold 에 엣지가 하나도 없음")
        self.edge.fit(np.concatenate(eas, axis=0), np.concatenate(evs, axis=0))

        gs = np.stack([arrays[i].graph_x for i in train_idx], axis=0)
        self.graph.fit(gs)
        return self

    def state_dict(self) -> dict:
        return {
            "kind": self.kind,
            "node": self.node.state_dict(),
            "edge": self.edge.state_dict(),
            "graph": self.graph.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "NormalizerSet":
        obj = cls(state["kind"])
        obj.node = Normalizer.from_state_dict(state["node"])
        obj.edge = Normalizer.from_state_dict(state["edge"])
        obj.graph = Normalizer.from_state_dict(state["graph"])
        return obj


def to_pyg(ga, y: int, name: str, norm: NormalizerSet) -> Data:
    """정규화를 적용해 PyG ``Data`` 로 변환한다."""
    x = torch.from_numpy(norm.node.transform(ga.x, _valid_of(ga, "x")))
    if ga.n_edge > 0:
        ea = torch.from_numpy(norm.edge.transform(ga.edge_attr, _valid_of(ga, "e")))
    else:
        ea = torch.zeros((0, len(norm.edge.spec)), dtype=torch.float32)
    gx = torch.from_numpy(norm.graph.transform(ga.graph_x[None, :]))

    return Data(
        x=x,
        edge_index=torch.from_numpy(ga.edge_index),
        edge_attr=ea,
        graph_x=gx,                                  # (1, d_graph) — 배치에서 (B, d_graph)
        y=torch.tensor([float(y)], dtype=torch.float32),
        node_kind=torch.from_numpy(ga.node_kind),
        node_ref=torch.from_numpy(ga.node_ref),
        name=name,
    )


def make_dataset(
    arrays: list, labels: list[int], names: list[str], idx: np.ndarray, norm: NormalizerSet
) -> list[Data]:
    """인덱스 부분집합을 PyG Data 리스트로."""
    return [to_pyg(arrays[i], labels[i], names[i], norm) for i in idx]


def feature_dims(kind: str) -> tuple[int, int, int]:
    """(노드 피처 차원, 엣지 피처 차원, 그래프 피처 차원)"""
    node_spec, edge_spec = graphs.specs(kind)
    return len(node_spec), len(edge_spec), len(GRAPH_FEATURE_NAMES)


if __name__ == "__main__":
    import argparse

    # 사내 PC 는 라벨 엑셀·데이터 파일에 암호가 걸리므로 파일을 열기 전에 해제한다.
    from fasoo_drm import setting_fasoo

    setting_fasoo()

    ap = argparse.ArgumentParser(description="그래프 캐시 생성 및 점검")
    ap.add_argument("--graph", choices=graphs.GRAPH_KINDS, default="elem")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-corner-edges", action="store_true")
    ap.add_argument("--no-lines", action="store_true")
    a = ap.parse_args()

    o: dict = {}
    if a.graph == "elem":
        o = {"corner_edges": not a.no_corner_edges, "include_lines": not a.no_lines}

    nm, lb, ar = build_cache(a.graph, opts=o, force=a.force)
    print(f"\n라벨 분포: 정상 {lb.count(0)} / 비정상 {lb.count(1)}")

    # 정규화기를 전체로 적합해 값 범위를 눈으로 확인 (실제 학습에서는 fold 로만 적합)
    ns = NormalizerSet(a.graph).fit(ar, np.arange(len(ar)))
    d = to_pyg(ar[0], lb[0], nm[0], ns)
    print(f"\n예시 그래프 {nm[0]}: {d}")
    print(f"  x    범위 [{d.x.min():.2f}, {d.x.max():.2f}]")
    if d.edge_attr.numel():
        print(f"  eattr 범위 [{d.edge_attr.min():.2f}, {d.edge_attr.max():.2f}]")
    print(f"  gx   범위 [{d.graph_x.min():.2f}, {d.graph_x.max():.2f}]")

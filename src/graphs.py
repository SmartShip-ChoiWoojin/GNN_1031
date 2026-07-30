"""그래프 표현 디스패처 — ``--graph elem|node`` 한 곳에서 갈린다."""

from __future__ import annotations

from typing import Callable

import graph_elem
import graph_node
from featspec import FeatureSpec
from graph_base import GraphArrays
from meshrep import MeshRep

GRAPH_KINDS = ("elem", "node")


def builder(kind: str) -> Callable[..., GraphArrays]:
    if kind == "elem":
        return graph_elem.build
    if kind == "node":
        return graph_node.build
    raise ValueError(f"알 수 없는 그래프 종류: {kind!r} (가능: {GRAPH_KINDS})")


def specs(kind: str) -> tuple[FeatureSpec, FeatureSpec]:
    """(노드 스펙, 엣지 스펙)"""
    mod = graph_elem if kind == "elem" else graph_node if kind == "node" else None
    if mod is None:
        raise ValueError(f"알 수 없는 그래프 종류: {kind!r} (가능: {GRAPH_KINDS})")
    return mod.NODE_SPEC, mod.EDGE_SPEC


def build(rep: MeshRep, kind: str, **opts) -> GraphArrays:
    return builder(kind)(rep, **opts)

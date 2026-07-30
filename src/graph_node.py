"""절점기반(original mesh) 그래프 빌더 — 노드 = GRID, 엣지 = 쉘 요소의 변.

요소기반과 같은 학습 루프로 비교하기 위한 두 번째 표현이다.

기존 `SHI/Ver1/bdf2info.py` 와의 차이:
  1. **절대 좌표를 쓰지 않는다.** 기존 코드는 raw x,y,z (x≈243680) 를 노드 피처 앞 3칸에
     그대로 넣었다(`bdf2info.py:98,101`). 패치들이 한 선박을 타일링하므로 절대 좌표는
     "선박 어디쯤인가"를 암기시키는 누출 경로다. 여기서는 각도결손·차수·상대길이 같은
     불변량만 쓴다.
  2. **json 외부연결 플래그를 노드 피처로 넣는다.** 기존 코드는 json 을 아예 읽지 않았다.
  3. **엣지에 1D 요소 정보를 싣는다.** 이 데이터셋의 1D 는 100% 쉘 변 위에 있으므로,
     절점기반 표현에서 1D 는 "그 변에 보강재가 있는지"라는 엣지 속성이 된다.
  4. 그래프 내부 z-score 정규화를 하지 않는다 (featspec.py 참고).
"""

from __future__ import annotations

import numpy as np

import features as F
from featspec import FeatureSpec
from graph_base import GraphArrays, add_undirected
from meshrep import MeshRep, graph_feature_vector

KIND_GRID = 0

NODE_SPEC = FeatureSpec.of(
    # ── 외부연결(json) ─────────────────────────────────────────────
    ("ext", "binary"),
    ("n_ext_neighbor", "linear"),
    ("frac_ext_neighbor", "linear"),
    # ── 차수 / 인접 요소 구성 ──────────────────────────────────────
    ("deg_shell", "linear"),
    ("deg_line", "linear"),
    ("deg_edge", "linear"),
    ("n_quad4", "linear"),
    ("n_tria3", "linear"),
    ("n_bar", "linear"),
    ("n_beam", "linear"),
    ("tri_frac_local", "linear"),
    # ── 각도 (이산 곡률) ───────────────────────────────────────────
    ("angle_sum", "linear"),
    ("angle_deficit", "linear"),
    ("angle_min", "linear"),
    ("angle_max", "linear"),
    ("angle_mean", "linear"),
    # ── 인접 변 길이 ───────────────────────────────────────────────
    ("elen_min", "log"),
    ("elen_max", "log"),
    ("elen_mean", "log"),
    ("elen_std", "log"),
    ("elen_rel_min", "linear"),
    ("elen_rel_max", "linear"),
    # 비율 피처는 1 이상이고 오른쪽 꼬리가 길다 → log 후 스케일
    ("elen_ratio", "log"),
    # ── 인접 요소 품질 요약 ────────────────────────────────────────
    ("adj_area_min", "log"),
    ("adj_area_max", "log"),
    ("adj_area_ratio", "log"),
    ("adj_aspect_max", "log"),
    ("adj_skew_max", "linear"),
    ("adj_njac_min", "linear"),
    ("adj_det_ratio_min", "linear"),
    ("n_bad_adj", "linear"),
    # ── 위상 ───────────────────────────────────────────────────────
    ("n_free_edge", "linear"),
    ("n_nonman_edge", "linear"),
    ("n_edge_with_bar", "linear"),
    ("n_free_edge_cut", "linear"),
    ("is_boundary", "binary"),
    ("is_cut_boundary", "binary"),
    ("is_irregular", "binary"),
    ("max_dihedral", "linear"),
)

EDGE_SPEC = FeatureSpec.of(
    ("length", "log"),
    ("length_rel", "linear"),
    ("edge_kind", "linear"),
    ("is_free", "binary"),
    ("is_nonmanifold", "binary"),
    ("has_bar", "binary"),
    ("has_beam", "binary"),
    ("dihedral", "linear"),
    ("cos_dihedral", "linear"),
    ("area_log_ratio", "linear"),
    ("both_ends_ext", "binary"),
    ("any_end_ext", "binary"),
    ("is_cut_edge", "binary"),
)

_NI = {n: i for i, n in enumerate(NODE_SPEC.names)}
_EI = {n: i for i, n in enumerate(EDGE_SPEC.names)}

# ── 유효성 마스크 ──────────────────────────────────────────────────────
# ``adj_*`` 는 인접 쉘이 있어야 의미가 있고, 이면각·면적비는 변이 두 쉘에 공유되어야
# 의미가 있다. 해당 없는 칸을 통계에 넣으면 0 이 중앙값을 끌어당겨 나머지 값이 왜곡된다.
_ADJ_SHELL = tuple(n for n in NODE_SPEC.names if n.startswith("adj_") or n == "n_bad_adj")
_TWO_SHELL_EDGE = ("dihedral", "cos_dihedral", "area_log_ratio")


def _node_valid(has_shell: bool) -> np.ndarray:
    v = np.ones(len(NODE_SPEC), dtype=bool)
    if not has_shell:
        for n in _ADJ_SHELL:
            v[_NI[n]] = False
    return v


def _edge_valid(n_shared_shells: int) -> np.ndarray:
    v = np.ones(len(EDGE_SPEC), dtype=bool)
    if n_shared_shells < 2:
        for n in _TWO_SHELL_EDGE:
            v[_EI[n]] = False
    return v


def _node_row(rep: MeshRep, nid: int, neighbors: list[int]) -> np.ndarray:
    row = np.zeros(len(NODE_SPEC), dtype=np.float64)
    med = rep.median_edge_length

    row[_NI["ext"]] = float(rep.ext.get(nid, False))
    n_ext_nb = sum(1 for m in neighbors if rep.ext.get(m, False))
    row[_NI["n_ext_neighbor"]] = float(n_ext_nb)
    row[_NI["frac_ext_neighbor"]] = n_ext_nb / max(len(neighbors), 1)

    shells = rep.node_shells.get(nid, [])
    lines = rep.node_lines.get(nid, [])
    edges = rep.node_edges(nid)

    row[_NI["deg_shell"]] = float(len(shells))
    row[_NI["deg_line"]] = float(len(lines))
    row[_NI["deg_edge"]] = float(len(edges))
    n_quad = sum(1 for si in shells if rep.shells[si].etype == "CQUAD4")
    n_tri = len(shells) - n_quad
    row[_NI["n_quad4"]] = float(n_quad)
    row[_NI["n_tria3"]] = float(n_tri)
    row[_NI["n_bar"]] = float(sum(1 for li in lines if rep.lines[li].etype == "CBAR"))
    row[_NI["n_beam"]] = float(sum(1 for li in lines if rep.lines[li].etype == "CBEAM"))
    row[_NI["tri_frac_local"]] = n_tri / max(len(shells), 1)

    ang = rep.node_incident_angles(nid)
    total = float(ang.sum())
    row[_NI["angle_sum"]] = total
    # 360 - 내각합 = 이산 가우스 곡률. 평평한 내부 절점이면 0.
    row[_NI["angle_deficit"]] = 360.0 - total
    row[_NI["angle_min"]] = float(ang.min())
    row[_NI["angle_max"]] = float(ang.max())
    row[_NI["angle_mean"]] = float(ang.mean())

    L = np.array([rep.edge_length(e) for e in edges]) if edges else np.zeros(1)
    row[_NI["elen_min"]] = float(L.min())
    row[_NI["elen_max"]] = float(L.max())
    row[_NI["elen_mean"]] = float(L.mean())
    row[_NI["elen_std"]] = float(L.std())
    row[_NI["elen_rel_min"]] = float(L.min() / (med + F.EPS))
    row[_NI["elen_rel_max"]] = float(L.max() / (med + F.EPS))
    row[_NI["elen_ratio"]] = float(L.max() / (L.min() + F.EPS))

    if shells:
        Q = [rep.shells[si].quality for si in shells]
        areas = np.array([q.area for q in Q])
        row[_NI["adj_area_min"]] = float(areas.min())
        row[_NI["adj_area_max"]] = float(areas.max())
        row[_NI["adj_area_ratio"]] = float(areas.max() / (areas.min() + F.EPS))
        row[_NI["adj_aspect_max"]] = float(max(q.aspect for q in Q))
        row[_NI["adj_skew_max"]] = float(max(q.skew for q in Q))
        row[_NI["adj_njac_min"]] = float(min(abs(q.norm_jacobian) for q in Q))
        row[_NI["adj_det_ratio_min"]] = float(min(q.det_ratio for q in Q))
        row[_NI["n_bad_adj"]] = float(sum(1 for q in Q if F.is_bad_shell(q)))

    n_free = sum(1 for e in edges if rep.is_free_edge(e))
    n_free_cut = sum(1 for e in edges if rep.is_free_edge(e) and rep.edge_both_ext(e))
    row[_NI["n_free_edge"]] = float(n_free)
    row[_NI["n_nonman_edge"]] = float(sum(1 for e in edges if rep.is_nonmanifold_edge(e)))
    row[_NI["n_edge_with_bar"]] = float(sum(1 for e in edges if rep.edge_lines.get(e)))
    row[_NI["n_free_edge_cut"]] = float(n_free_cut)
    row[_NI["is_boundary"]] = float(n_free > 0)
    # 자유변이 전부 절단면 자유변이면, 이 절점이 경계에 있는 것은 결함이 아니라
    # 모델을 잘라낸 결과다. 사용자가 지적한 도메인 규칙을 절점 수준에서 표현한 피처.
    row[_NI["is_cut_boundary"]] = float(n_free > 0 and n_free == n_free_cut)
    row[_NI["is_irregular"]] = float(
        n_free == 0 and not rep.ext.get(nid, False) and len(shells) != 4
    )
    row[_NI["max_dihedral"]] = rep.node_max_dihedral(nid)
    return row


def _edge_row(rep: MeshRep, e: frozenset) -> list[float]:
    row = np.zeros(len(EDGE_SPEC), dtype=np.float64)
    med = rep.median_edge_length

    L = rep.edge_length(e)
    row[_EI["length"]] = L
    row[_EI["length_rel"]] = L / (med + F.EPS)

    shells = rep.edge_shells.get(e, [])
    kind = len(shells)
    row[_EI["edge_kind"]] = float(kind)
    row[_EI["is_free"]] = float(kind == 1)
    row[_EI["is_nonmanifold"]] = float(kind > 2)

    lines = rep.edge_lines.get(e, [])
    row[_EI["has_bar"]] = float(any(rep.lines[li].etype == "CBAR" for li in lines))
    row[_EI["has_beam"]] = float(any(rep.lines[li].etype == "CBEAM" for li in lines))

    if kind >= 2:
        q1 = rep.shells[shells[0]].quality
        q2 = rep.shells[shells[1]].quality
        d = F.dihedral_deg(q1.normal, q2.normal)
        row[_EI["dihedral"]] = d
        row[_EI["cos_dihedral"]] = float(np.cos(np.radians(d)))
        row[_EI["area_log_ratio"]] = float(
            abs(np.log((q1.area + F.EPS) / (q2.area + F.EPS)))
        )

    both = rep.edge_both_ext(e)
    row[_EI["both_ends_ext"]] = float(both)
    row[_EI["any_end_ext"]] = float(any(rep.ext.get(n, False) for n in e))
    # 자유변이면서 두 끝점이 모두 외부연결 = 구멍이 아니라 모델 절단면
    row[_EI["is_cut_edge"]] = float(kind == 1 and both)
    return row.tolist()


def build(rep: MeshRep, **_ignored) -> GraphArrays:
    """MeshRep -> 절점기반 :class:`GraphArrays`.

    노드 순서는 ``rep.node_ids`` (절점 id 오름차순).
    """
    idx = rep.node_index

    neighbors: dict[int, list[int]] = {n: [] for n in rep.node_ids}
    for e in rep.edge_shells:
        a, b = tuple(e)
        neighbors[a].append(b)
        neighbors[b].append(a)

    x = np.stack([_node_row(rep, n, neighbors[n]) for n in rep.node_ids])
    x_valid = np.stack(
        [_node_valid(bool(rep.node_shells.get(n))) for n in rep.node_ids]
    )

    ei: list[list[int]] = []
    ea: list[list[float]] = []
    ev: list[np.ndarray] = []
    for e in rep.edge_shells:
        a, b = tuple(e)
        add_undirected(
            ei, ea, ev, idx[a], idx[b],
            _edge_row(rep, e), _edge_valid(rep.edge_kind(e)),
        )

    edge_index = np.array(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.array(ea, dtype=np.float64) if ea else np.zeros((0, len(EDGE_SPEC)))
    edge_valid = np.stack(ev) if ev else np.zeros((0, len(EDGE_SPEC)), dtype=bool)

    return GraphArrays(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        graph_x=graph_feature_vector(rep),
        node_kind=np.full(len(rep.node_ids), KIND_GRID, dtype=np.int64),
        node_ref=np.array(rep.node_ids, dtype=np.int64),
        x_valid=x_valid,
        edge_valid=edge_valid,
    )

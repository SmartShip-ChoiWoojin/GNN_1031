"""요소기반(dual) 그래프 빌더 — 노드 = FE 요소.

메인 표현으로 삼는 이유: 라벨을 지배하는 신호(삼각형 비율, skew, 면적 편차, 불량 내각)가
전부 **요소 속성**이고, "어느 요소가 문제인가" 산출물이 노드에서 바로 나온다.

기존 `SHI/Ver1/bdf2info_v2.py` 와의 차이:
  1. **CBAR/CBEAM 을 노드로 포함한다.** 기존 파서(`SHI/Ver1/bdf.py:126`)는 1D 를 인식조차
     하지 않았다 — "fe model 1D" 데이터셋인데 1D 가 그래프에 없었다.
  2. **꼭짓점 인접을 버리지 않는다.** 기존 코드는 `if cnt < 2: continue` 로 절점 1개만
     공유하는 쌍을 전부 버렸고(`bdf2info_v2.py:203`), 그 결과 1D 연결도 함께 사라졌다.
     여기서는 관계 4종을 서로 다른 엣지 타입으로 구분해 모두 넣는다.
  3. 정규화를 그래프 내부에서 하지 않는다 (featspec.py 참고).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

import features as F
from featspec import FeatureSpec
from graph_base import GraphArrays, add_undirected
from meshrep import MeshRep, graph_feature_vector

# 노드 종류 코드 — GraphArrays.node_kind 에 저장, explain.py 가 해석
KIND_SHELL = 0
KIND_LINE = 1

# 엣지 관계 종류
REL_SHELL_EDGE = 0    # 쉘-쉘, 변(절점 2개) 공유
REL_SHELL_CORNER = 1  # 쉘-쉘, 절점 1개만 공유
REL_LINE_SHELL = 2    # 1D 가 그 쉘의 변 위에 놓임
REL_LINE_LINE = 3     # 1D-1D, 절점 공유
N_REL = 4


NODE_SPEC = FeatureSpec.of(
    # ── 요소 타입 one-hot (bdf_io.TYPE_INDEX 순서) ──────────────────
    ("is_quad4", "binary"),
    ("is_tria3", "binary"),
    ("is_bar", "binary"),
    ("is_beam", "binary"),
    # ── 외부연결(json) — 모든 요소에 유효 ───────────────────────────
    ("ext_frac", "linear"),
    ("all_ext", "binary"),
    ("any_ext", "binary"),
    # ── 쉘 기하 (1D 는 0) ──────────────────────────────────────────
    ("s_area", "log"),
    ("s_area_rel", "linear"),
    # 형상비는 1 이상이고 오른쪽 꼬리가 매우 길다(최대 수십) → log 후 스케일
    ("s_aspect", "log"),
    ("s_diag_ratio", "log"),
    ("s_angle_min", "linear"),
    ("s_angle_max", "linear"),
    ("s_skew", "linear"),
    ("s_warp", "linear"),
    ("s_taper", "linear"),
    ("s_njac_abs", "linear"),
    ("s_det_ratio", "linear"),
    ("s_len_min", "log"),
    ("s_len_max", "log"),
    ("s_len_mean", "log"),
    ("s_len_std", "log"),
    ("s_perimeter", "log"),
    ("s_len_rel_min", "linear"),
    ("s_len_rel_max", "linear"),
    ("s_n_corner", "linear"),
    ("s_nx", "linear"),
    ("s_ny", "linear"),
    ("s_nz", "linear"),
    ("s_bad_angle", "binary"),
    # ── 쉘 위상 ───────────────────────────────────────────────────
    ("s_free_edges", "linear"),
    ("s_nonman_edges", "linear"),
    ("s_edges_with_bar", "linear"),
    ("s_free_edges_cut", "linear"),
    ("s_corner_deg_min", "linear"),
    ("s_corner_deg_max", "linear"),
    ("s_corner_deg_mean", "linear"),
    # ── 1D 기하 (쉘은 0) ──────────────────────────────────────────
    ("l_len", "log"),
    ("l_len_rel", "linear"),
    ("l_dx", "linear"),
    ("l_dy", "linear"),
    ("l_dz", "linear"),
    # ── 1D 위상 ───────────────────────────────────────────────────
    ("l_edge_kind", "linear"),
    ("l_on_free_edge", "binary"),
    ("l_end_deg_min", "linear"),
    ("l_end_deg_max", "linear"),
    ("l_dangling_internal", "binary"),
)

EDGE_SPEC = FeatureSpec.of(
    ("rel_shell_edge", "binary"),
    ("rel_shell_corner", "binary"),
    ("rel_line_shell", "binary"),
    ("rel_line_line", "binary"),
    ("shared_len", "log"),
    ("shared_len_rel", "linear"),
    ("dihedral", "linear"),
    ("cos_dihedral", "linear"),
    ("area_log_ratio", "linear"),
    ("d_angle_min", "linear"),
    ("same_type", "binary"),
    ("shared_edge_kind", "linear"),
    ("shared_has_bar", "binary"),
    ("shared_both_ext", "binary"),
    ("shared_any_ext", "binary"),
)

_NI = {n: i for i, n in enumerate(NODE_SPEC.names)}
_EI = {n: i for i, n in enumerate(EDGE_SPEC.names)}

#: 관계 코드 -> EDGE_SPEC 의 one-hot 피처 이름
_REL_NAME = {
    REL_SHELL_EDGE: "rel_shell_edge",
    REL_SHELL_CORNER: "rel_shell_corner",
    REL_LINE_SHELL: "rel_line_shell",
    REL_LINE_LINE: "rel_line_line",
}

# ── 유효성 마스크 템플릿 ───────────────────────────────────────────────
# 피처 이름 접두사가 곧 적용 대상이다: ``s_`` = 쉘 전용, ``l_`` = 1D 전용, 나머지는 공용.
# 마스킹된 칸을 통계에서 빼지 않으면(예: 1D 노드의 면적 0), 그 0 이 중앙값을 끌어당겨
# 반대 타입 노드가 전부 클립 경계값(-10)을 받게 된다.
_SHELL_ONLY = np.array([n.startswith("s_") for n in NODE_SPEC.names])
_LINE_ONLY = np.array([n.startswith("l_") for n in NODE_SPEC.names])
_SHARED = ~(_SHELL_ONLY | _LINE_ONLY)

VALID_SHELL_NODE = _SHARED | _SHELL_ONLY
VALID_LINE_NODE = _SHARED | _LINE_ONLY

# 엣지 피처의 유효성은 관계 종류에 따라 갈린다.
#   공유 변 기하 (shared_len / shared_len_rel / shared_edge_kind / shared_has_bar)
#     → 절점 2개를 공유하는 관계에서만 의미가 있다.
#   쉘 쌍 기하 (dihedral / cos_dihedral / area_log_ratio / d_angle_min)
#     → 양쪽이 모두 쉘일 때만 의미가 있다.
_SHARED_EDGE_GEOM = ("shared_len", "shared_len_rel", "shared_edge_kind", "shared_has_bar")
_SHELL_PAIR_GEOM = ("dihedral", "cos_dihedral", "area_log_ratio", "d_angle_min")


def _edge_valid(has_shared_edge: bool, both_shells: bool) -> np.ndarray:
    v = np.ones(len(EDGE_SPEC), dtype=bool)
    if not has_shared_edge:
        for n in _SHARED_EDGE_GEOM:
            v[_EI[n]] = False
    if not both_shells:
        for n in _SHELL_PAIR_GEOM:
            v[_EI[n]] = False
    return v


#: 관계 코드 -> (절점 2개 공유 여부, 양쪽 모두 쉘 여부)
_REL_GEOM = {
    REL_SHELL_EDGE: (True, True),
    REL_SHELL_CORNER: (False, True),
    REL_LINE_SHELL: (True, False),
    REL_LINE_LINE: (False, False),
}


def _shell_row(rep: MeshRep, si: int) -> np.ndarray:
    """쉘 요소 하나의 원시 노드 피처."""
    row = np.zeros(len(NODE_SPEC), dtype=np.float64)
    sh = rep.shells[si]
    q = sh.quality
    med = rep.median_edge_length

    row[_NI["is_quad4" if q.is_quad else "is_tria3"]] = 1.0

    ext_flags = [rep.ext.get(n, False) for n in sh.nodes]
    row[_NI["ext_frac"]] = sum(ext_flags) / len(ext_flags)
    row[_NI["all_ext"]] = float(all(ext_flags))
    row[_NI["any_ext"]] = float(any(ext_flags))

    row[_NI["s_area"]] = q.area
    row[_NI["s_area_rel"]] = q.area / (med * med + F.EPS)
    row[_NI["s_aspect"]] = q.aspect
    row[_NI["s_diag_ratio"]] = q.diag_ratio
    row[_NI["s_angle_min"]] = q.angle_min
    row[_NI["s_angle_max"]] = q.angle_max
    row[_NI["s_skew"]] = q.skew
    row[_NI["s_warp"]] = q.warp
    row[_NI["s_taper"]] = q.taper
    # 감김 방향에 따라 부호가 뒤집히므로 크기만 쓴다
    row[_NI["s_njac_abs"]] = abs(q.norm_jacobian)
    # det_ratio 는 부호를 살린다 — 음수면 요소가 접혀 있다는 뜻
    row[_NI["s_det_ratio"]] = q.det_ratio
    row[_NI["s_len_min"]] = q.len_min
    row[_NI["s_len_max"]] = q.len_max
    row[_NI["s_len_mean"]] = q.len_mean
    row[_NI["s_len_std"]] = q.len_std
    row[_NI["s_perimeter"]] = q.perimeter
    row[_NI["s_len_rel_min"]] = q.len_min / (med + F.EPS)
    row[_NI["s_len_rel_max"]] = q.len_max / (med + F.EPS)
    row[_NI["s_n_corner"]] = float(len(sh.nodes))
    nrm = F.unsigned_normal(q.normal)
    row[_NI["s_nx"]], row[_NI["s_ny"]], row[_NI["s_nz"]] = nrm
    row[_NI["s_bad_angle"]] = float(F.is_bad_shell(q))

    free, nonman, withbar = rep.shell_edge_stats(si)
    row[_NI["s_free_edges"]] = float(free)
    row[_NI["s_nonman_edges"]] = float(nonman)
    row[_NI["s_edges_with_bar"]] = float(withbar)
    # 자유변이면서 두 끝점이 모두 외부연결 = 모델 절단면. 구멍이 아니므로 감점 대상이 아니다.
    row[_NI["s_free_edges_cut"]] = float(
        sum(1 for e in sh.edges() if rep.is_free_edge(e) and rep.edge_both_ext(e))
    )

    deg = [len(rep.node_shells.get(n, ())) for n in sh.nodes]
    row[_NI["s_corner_deg_min"]] = float(min(deg))
    row[_NI["s_corner_deg_max"]] = float(max(deg))
    row[_NI["s_corner_deg_mean"]] = float(np.mean(deg))
    return row


def _line_row(rep: MeshRep, li: int) -> np.ndarray:
    """1D 요소 하나의 원시 노드 피처."""
    row = np.zeros(len(NODE_SPEC), dtype=np.float64)
    ln = rep.lines[li]
    med = rep.median_edge_length

    row[_NI["is_bar" if ln.etype == "CBAR" else "is_beam"]] = 1.0

    ext_flags = [rep.ext.get(n, False) for n in ln.nodes]
    row[_NI["ext_frac"]] = sum(ext_flags) / 2.0
    row[_NI["all_ext"]] = float(all(ext_flags))
    row[_NI["any_ext"]] = float(any(ext_flags))

    row[_NI["l_len"]] = ln.length
    row[_NI["l_len_rel"]] = ln.length / (med + F.EPS)
    row[_NI["l_dx"]], row[_NI["l_dy"]], row[_NI["l_dz"]] = ln.direction

    e = ln.edge
    row[_NI["l_edge_kind"]] = float(rep.edge_kind(e))
    row[_NI["l_on_free_edge"]] = float(rep.is_free_edge(e))
    dmin, dmax = rep.line_degree(li)
    row[_NI["l_end_deg_min"]] = float(dmin)
    row[_NI["l_end_deg_max"]] = float(dmax)
    # 보강재 체인이 끊긴 끝점인데 그 절점이 외부연결이 아닌 경우 = 진짜 끊김.
    # ext 를 무시한 지표(0.132)보다 라벨과의 상관이 훨씬 높다(0.395).
    row[_NI["l_dangling_internal"]] = float(
        any(
            len(rep.node_lines.get(n, ())) == 1 and not rep.ext.get(n, False)
            for n in ln.nodes
        )
    )
    return row


def _edge_row(
    rep: MeshRep,
    rel: int,
    shared_nodes: tuple[int, ...],
    a_quality: F.ShellQuality | None,
    b_quality: F.ShellQuality | None,
    same_type: bool,
) -> tuple[list[float], np.ndarray]:
    """엣지 하나의 (원시 피처, 유효성 마스크)."""
    row = np.zeros(len(EDGE_SPEC), dtype=np.float64)
    row[_EI[_REL_NAME[rel]]] = 1.0
    med = rep.median_edge_length

    if len(shared_nodes) == 2:
        e = frozenset(shared_nodes)
        L = rep.edge_length(e)
        row[_EI["shared_len"]] = L
        row[_EI["shared_len_rel"]] = L / (med + F.EPS)
        row[_EI["shared_edge_kind"]] = float(rep.edge_kind(e))
        row[_EI["shared_has_bar"]] = float(bool(rep.edge_lines.get(e)))
        row[_EI["shared_both_ext"]] = float(rep.edge_both_ext(e))
        row[_EI["shared_any_ext"]] = float(
            any(rep.ext.get(n, False) for n in shared_nodes)
        )
    else:
        n0 = shared_nodes[0]
        row[_EI["shared_edge_kind"]] = 0.0
        row[_EI["shared_both_ext"]] = float(rep.ext.get(n0, False))
        row[_EI["shared_any_ext"]] = float(rep.ext.get(n0, False))

    if a_quality is not None and b_quality is not None:
        d = F.dihedral_deg(a_quality.normal, b_quality.normal)
        row[_EI["dihedral"]] = d
        row[_EI["cos_dihedral"]] = np.cos(np.radians(d))
        row[_EI["area_log_ratio"]] = abs(
            np.log((a_quality.area + F.EPS) / (b_quality.area + F.EPS))
        )
        row[_EI["d_angle_min"]] = abs(a_quality.angle_min - b_quality.angle_min)

    row[_EI["same_type"]] = float(same_type)
    return row.tolist(), _edge_valid(*_REL_GEOM[rel])


def build(rep: MeshRep, corner_edges: bool = True, include_lines: bool = True) -> GraphArrays:
    """MeshRep -> 요소기반 :class:`GraphArrays`.

    Args:
        corner_edges: 절점 1개만 공유하는 쉘 쌍도 엣지로 넣을지 (기본 True).
        include_lines: CBAR/CBEAM 을 노드로 넣을지. ablation 용으로 False 가능.
    """
    n_shell = len(rep.shells)
    n_line = len(rep.lines) if include_lines else 0

    rows = [_shell_row(rep, si) for si in range(n_shell)]
    valids = [VALID_SHELL_NODE] * n_shell
    if include_lines:
        rows += [_line_row(rep, li) for li in range(n_line)]
        valids += [VALID_LINE_NODE] * n_line
    x = np.stack(rows) if rows else np.zeros((0, len(NODE_SPEC)))
    x_valid = np.stack(valids) if valids else np.zeros((0, len(NODE_SPEC)), dtype=bool)

    node_kind = np.array([KIND_SHELL] * n_shell + [KIND_LINE] * n_line, dtype=np.int64)
    node_ref = np.array(
        [s.eid for s in rep.shells] + [l.eid for l in rep.lines[:n_line]], dtype=np.int64
    )

    def line_node(li: int) -> int:
        return n_shell + li

    ei: list[list[int]] = []
    ea: list[list[float]] = []
    ev: list[np.ndarray] = []

    # ── 관계 0: 쉘-쉘, 변 공유 ───────────────────────────────────────
    edge_adjacent: set[tuple[int, int]] = set()
    for e, shells in rep.edge_shells.items():
        if len(shells) < 2:
            continue
        # 비매니폴드 변(패널 교차)에서는 그 변을 공유하는 모든 쌍을 연결한다
        for i, j in combinations(sorted(shells), 2):
            edge_adjacent.add((i, j))
            feat, val = _edge_row(
                rep, REL_SHELL_EDGE, tuple(e),
                rep.shells[i].quality, rep.shells[j].quality,
                rep.shells[i].etype == rep.shells[j].etype,
            )
            add_undirected(ei, ea, ev, i, j, feat, val)

    # ── 관계 1: 쉘-쉘, 절점 1개만 공유 ──────────────────────────────
    if corner_edges:
        shared_node: dict[tuple[int, int], list[int]] = defaultdict(list)
        for nid, shells in rep.node_shells.items():
            for i, j in combinations(sorted(shells), 2):
                shared_node[(i, j)].append(nid)
        for (i, j), nodes in shared_node.items():
            if (i, j) in edge_adjacent:
                continue
            feat, val = _edge_row(
                rep, REL_SHELL_CORNER, (nodes[0],),
                rep.shells[i].quality, rep.shells[j].quality,
                rep.shells[i].etype == rep.shells[j].etype,
            )
            add_undirected(ei, ea, ev, i, j, feat, val)

    if include_lines:
        # ── 관계 2: 1D 가 얹힌 쉘 ───────────────────────────────────
        for li, ln in enumerate(rep.lines):
            for si in rep.edge_shells.get(ln.edge, ()):
                feat, val = _edge_row(rep, REL_LINE_SHELL, ln.nodes, None, None, False)
                add_undirected(ei, ea, ev, line_node(li), si, feat, val)

        # ── 관계 3: 1D-1D, 절점 공유 ────────────────────────────────
        seen_ll: set[tuple[int, int]] = set()
        for nid, lines in rep.node_lines.items():
            for li, lj in combinations(sorted(lines), 2):
                if (li, lj) in seen_ll:
                    continue
                seen_ll.add((li, lj))
                feat, val = _edge_row(
                    rep, REL_LINE_LINE, (nid,), None, None,
                    rep.lines[li].etype == rep.lines[lj].etype,
                )
                add_undirected(ei, ea, ev, line_node(li), line_node(lj), feat, val)

    edge_index = (
        np.array(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
    )
    edge_attr = (
        np.array(ea, dtype=np.float64) if ea else np.zeros((0, len(EDGE_SPEC)))
    )
    edge_valid = (
        np.stack(ev) if ev else np.zeros((0, len(EDGE_SPEC)), dtype=bool)
    )

    return GraphArrays(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        graph_x=graph_feature_vector(rep),
        node_kind=node_kind,
        node_ref=node_ref,
        x_valid=x_valid,
        edge_valid=edge_valid,
    )

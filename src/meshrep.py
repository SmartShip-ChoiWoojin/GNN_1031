"""MeshRep — bdf/json 을 한 번 파싱해 만든 정규 중간 표현.

절점기반 그래프 빌더, 요소기반 그래프 빌더, 집계 피처 기준선이 모두 이 하나를 소비한다.
위상(엣지 incidence, free/manifold/nonmanifold 분류, 1D↔엣지 매핑)과 요소별 기하 품질을
여기서 한 번만 계산해 캐싱한다.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

import features as F
from bdf_io import LINE_TYPES, SHELL_TYPES, RawMesh, load_raw_mesh

Edge = frozenset  # frozenset[int] — 무방향 절점쌍


@dataclass
class ShellEntry:
    """쉘 요소 하나 + 품질 캐시."""

    eid: int
    etype: str  # "CQUAD4" | "CTRIA3"
    nodes: list[int]
    quality: F.ShellQuality
    centroid: np.ndarray = field(repr=False)

    @property
    def n_corner(self) -> int:
        return len(self.nodes)

    def edges(self) -> list[Edge]:
        """요소를 이루는 변들(무방향)."""
        k = len(self.nodes)
        return [Edge((self.nodes[i], self.nodes[(i + 1) % k])) for i in range(k)]


@dataclass
class LineEntry:
    """1D 요소 하나 + 기하 캐시."""

    eid: int
    etype: str  # "CBAR" | "CBEAM"
    nodes: tuple[int, int]
    length: float
    direction: np.ndarray = field(repr=False)  # 부호없는 방향코사인
    centroid: np.ndarray = field(repr=False)

    @property
    def edge(self) -> Edge:
        return Edge(self.nodes)


class MeshRep:
    """메쉬 패치 하나의 기하 + 위상 + 외부연결 정보."""

    def __init__(self, raw: RawMesh):
        self.raw = raw
        self.name = raw.name

        self.node_ids: list[int] = raw.node_ids
        self.node_index: dict[int, int] = {n: i for i, n in enumerate(self.node_ids)}
        self.points: np.ndarray = raw.points
        self.ext: dict[int, bool] = raw.ext

        self.shells: list[ShellEntry] = []
        for eid, etype, nodes in raw.shells:
            P = np.array([raw.coords[n] for n in nodes], dtype=float)
            self.shells.append(
                ShellEntry(
                    eid=eid,
                    etype=etype,
                    nodes=list(nodes),
                    quality=F.shell_quality(P),
                    centroid=P.mean(axis=0),
                )
            )

        self.lines: list[LineEntry] = []
        for eid, etype, (a, b) in raw.lines:
            p1 = np.array(raw.coords[a], dtype=float)
            p2 = np.array(raw.coords[b], dtype=float)
            ln, d = F.line_length_dir(p1, p2)
            self.lines.append(
                LineEntry(
                    eid=eid,
                    etype=etype,
                    nodes=(a, b),
                    length=ln,
                    direction=d,
                    centroid=0.5 * (p1 + p2),
                )
            )

        # ── 위상 ──────────────────────────────────────────────────────
        #: 변 -> 그 변을 공유하는 쉘 인덱스 목록
        self.edge_shells: dict[Edge, list[int]] = defaultdict(list)
        for si, sh in enumerate(self.shells):
            for e in sh.edges():
                self.edge_shells[e].append(si)

        #: 변 -> 그 변 위에 놓인 1D 요소 인덱스 목록
        self.edge_lines: dict[Edge, list[int]] = defaultdict(list)
        for li, ln in enumerate(self.lines):
            self.edge_lines[ln.edge].append(li)

        #: 절점 -> 인접 쉘 / 인접 1D
        self.node_shells: dict[int, list[int]] = defaultdict(list)
        for si, sh in enumerate(self.shells):
            for n in sh.nodes:
                self.node_shells[n].append(si)
        self.node_lines: dict[int, list[int]] = defaultdict(list)
        for li, ln in enumerate(self.lines):
            self.node_lines[ln.nodes[0]].append(li)
            self.node_lines[ln.nodes[1]].append(li)

    # ── 스케일 기준 ───────────────────────────────────────────────────
    @cached_property
    def median_edge_length(self) -> float:
        """모델 내 쉘 변 길이의 중앙값. 길이 피처를 상대화할 때의 기준."""
        L = [self.edge_length(e) for e in self.edge_shells]
        return float(np.median(L)) if L else 1.0

    def edge_length(self, e: Edge) -> float:
        a, b = tuple(e)
        return float(np.linalg.norm(np.array(self.raw.coords[a]) - np.array(self.raw.coords[b])))

    # ── 변 분류 ───────────────────────────────────────────────────────
    def edge_kind(self, e: Edge) -> int:
        """변을 공유하는 쉘 개수. 1=자유변, 2=정상, 3+=비매니폴드(패널 교차)."""
        return len(self.edge_shells.get(e, ()))

    def is_free_edge(self, e: Edge) -> bool:
        return self.edge_kind(e) == 1

    def is_nonmanifold_edge(self, e: Edge) -> bool:
        return self.edge_kind(e) > 2

    def edge_both_ext(self, e: Edge) -> bool:
        """변의 두 끝점이 모두 모델 외부와 연결되어 있는지.

        자유변인데 이 값이 True 면 "구멍"이 아니라 모델을 잘라낸 절단면이다.
        즉 결함으로 감점해서는 안 되는 경계 — 사용자가 지적한 도메인 규칙의 핵심.
        """
        return all(self.ext.get(n, False) for n in e)

    # ── 절점 파생량 ───────────────────────────────────────────────────
    def node_angle_sum(self, nid: int) -> float:
        """절점 주위 인접 쉘 내각의 합(도).

        평평한 내부 절점이면 360°. 이보다 작으면 경계/원뿔, 크면 안장/교차부.
        360 - 이 값이 이산 가우스 곡률이며 평행이동·회전 불변량이다.
        """
        total = 0.0
        for si in self.node_shells.get(nid, ()):
            sh = self.shells[si]
            P = self.points[[self.node_index[n] for n in sh.nodes]]
            ang = F.interior_angles(P)
            total += float(ang[sh.nodes.index(nid)])
        return total

    def node_incident_angles(self, nid: int) -> np.ndarray:
        """절점에 모인 인접 쉘 내각들(도)."""
        out = []
        for si in self.node_shells.get(nid, ()):
            sh = self.shells[si]
            P = self.points[[self.node_index[n] for n in sh.nodes]]
            out.append(float(F.interior_angles(P)[sh.nodes.index(nid)]))
        return np.array(out) if out else np.zeros(1)

    def node_edges(self, nid: int) -> list[Edge]:
        """절점에 붙은 쉘 변들."""
        return [e for e in self.edge_shells if nid in e]

    def node_max_dihedral(self, nid: int) -> float:
        """절점에 모인 쉘들 사이의 최대 법선 사잇각(도).

        0°에 가까우면 한 평면 위, 90°에 가까우면 패널이 교차하는 지점.
        법선 부호에 무관하도록 |cos| 을 쓴다(features.dihedral_deg).
        """
        idx = self.node_shells.get(nid, ())
        if len(idx) < 2:
            return 0.0
        best = 0.0
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                d = F.dihedral_deg(
                    self.shells[idx[i]].quality.normal, self.shells[idx[j]].quality.normal
                )
                best = max(best, d)
        return best

    # ── 요소 파생량 ───────────────────────────────────────────────────
    def shell_edge_stats(self, si: int) -> tuple[int, int, int]:
        """쉘 하나의 (자유변 수, 비매니폴드변 수, 1D가 얹힌 변 수)."""
        free = nonman = withbar = 0
        for e in self.shells[si].edges():
            k = self.edge_kind(e)
            if k == 1:
                free += 1
            elif k > 2:
                nonman += 1
            if self.edge_lines.get(e):
                withbar += 1
        return free, nonman, withbar

    def shell_ext_frac(self, si: int) -> float:
        nodes = self.shells[si].nodes
        return sum(1 for n in nodes if self.ext.get(n, False)) / len(nodes)

    def line_ext_frac(self, li: int) -> float:
        nodes = self.lines[li].nodes
        return sum(1 for n in nodes if self.ext.get(n, False)) / 2.0

    def line_degree(self, li: int) -> tuple[int, int]:
        """1D 요소 양 끝점의 1D 차수 (min, max).

        차수 1 인 끝점은 보강재 체인이 끊긴 곳이다. 그 절점이 ext 면 모델 밖으로
        이어지는 정상 상황이고, ext 가 아니면 진짜 끊김이다.
        """
        d = [len(self.node_lines.get(n, ())) for n in self.lines[li].nodes]
        return min(d), max(d)

    def dangling_line_ends(self, ignore_ext: bool = True) -> int:
        """1D 체인이 끊긴 끝점 수. ``ignore_ext`` 면 외부 연결 절점은 세지 않는다.

        측정 결과 ext 를 제외한 쪽이 라벨과 훨씬 강하게 상관한다 (0.395 vs 0.132) —
        사용자가 말한 도메인 규칙의 정량적 근거.
        """
        cnt = 0
        for nid, lst in self.node_lines.items():
            if len(lst) != 1:
                continue
            if ignore_ext and self.ext.get(nid, False):
                continue
            cnt += 1
        return cnt

    # ── 그래프 수준 집계 피처 ─────────────────────────────────────────
    def graph_features(self) -> dict[str, float]:
        """그래프 하나를 요약하는 집계 피처.

        두 곳에서 쓰인다:
          1. ``baseline.py`` 의 tabular 기준선 입력 (랜덤 5-fold AUC ≈ 0.961)
          2. GNN readout 뒤에 concat 되는 전역 피처 — GNN 이 최소한 기준선의
             표현력에서 출발하도록 보장한다.
        """
        Q = [s.quality for s in self.shells]
        n_shell = len(Q)
        n_tri = sum(1 for q in Q if not q.is_quad)

        areas = np.array([q.area for q in Q])
        aspect = np.array([q.aspect for q in Q])
        skew = np.array([q.skew for q in Q])
        warp = np.array([q.warp for q in Q])
        taper = np.array([q.taper for q in Q])
        amin = np.array([q.angle_min for q in Q])
        amax = np.array([q.angle_max for q in Q])
        njac = np.array([abs(q.norm_jacobian) for q in Q])
        dratio = np.array([q.det_ratio for q in Q])

        # 내부 변의 이면각 분포
        dih = [
            F.dihedral_deg(self.shells[v[0]].quality.normal, self.shells[v[1]].quality.normal)
            for v in self.edge_shells.values()
            if len(v) == 2
        ]
        dih_arr = np.array(dih) if dih else np.zeros(1)

        # 서로 다른 패널 방향 개수 (부호 정규화한 법선을 0.1 단위로 뭉갠 뒤 집합 크기)
        orients = {
            tuple(np.round(F.unsigned_normal(q.normal), 1)) for q in Q
        }

        n_free = sum(1 for v in self.edge_shells.values() if len(v) == 1)
        n_nonman = sum(1 for v in self.edge_shells.values() if len(v) > 2)

        # 불량 요소 — 전체 / 내부절점 위 / ext 절점 위
        bad_int = bad_ext = 0
        for si, q in enumerate(Q):
            if not F.is_bad_shell(q):
                continue
            if all(self.ext.get(n, False) for n in self.shells[si].nodes):
                bad_ext += 1
            else:
                bad_int += 1

        deg_shell = [len(v) for v in self.node_shells.values()] or [0]
        deg_line = [len(v) for v in self.node_lines.values()] or [0]

        # 규칙적이지 않은 내부 절점: 모든 인접 변이 공유되고 ext 가 아닌데 쉘차수 != 4
        irregular = 0
        for nid in self.node_ids:
            if self.ext.get(nid, False):
                continue
            if any(self.edge_kind(e) < 2 for e in self.node_edges(nid)):
                continue
            if len(self.node_shells.get(nid, ())) != 4:
                irregular += 1

        return {
            # 규모
            "n_node": float(len(self.node_ids)),
            "n_shell": float(n_shell),
            "n_quad": float(n_shell - n_tri),
            "n_tri": float(n_tri),
            "tri_frac": n_tri / n_shell,
            "n_bar": float(sum(1 for l in self.lines if l.etype == "CBAR")),
            "n_beam": float(sum(1 for l in self.lines if l.etype == "CBEAM")),
            # 쉘 품질
            "aspect_max": float(aspect.max()),
            "aspect_mean": float(aspect.mean()),
            "aspect_p90": float(np.percentile(aspect, 90)),
            "skew_max": float(skew.max()),
            "skew_mean": float(skew.mean()),
            "warp_max": float(warp.max()),
            "warp_mean": float(warp.mean()),
            "taper_min": float(taper.min()),
            "angle_min": float(amin.min()),
            "angle_max": float(amax.max()),
            "n_bad_angle": float(bad_int + bad_ext),
            "njac_min": float(njac.min()),
            "njac_mean": float(njac.mean()),
            "det_ratio_min": float(dratio.min()),
            "area_cv": float(areas.std() / (areas.mean() + F.EPS)),
            "area_ratio": float(areas.max() / (areas.min() + F.EPS)),
            # 위상
            "n_orient": float(len(orients)),
            "dih_max": float(dih_arr.max()),
            "dih_mean": float(dih_arr.mean()),
            "n_free_edge": float(n_free),
            "n_nonmanifold_edge": float(n_nonman),
            "deg_shell_max": float(max(deg_shell)),
            "deg_line_max": float(max(deg_line)),
            "n_irregular_node": float(irregular),
            # 외부연결(json) 을 반영한 지표
            "ext_frac": sum(1 for v in self.ext.values() if v) / max(len(self.ext), 1),
            "n_bad_internal": float(bad_int),
            "n_bad_ext_only": float(bad_ext),
            "n_dangling_internal": float(self.dangling_line_ends(ignore_ext=True)),
            "n_dangling_all": float(self.dangling_line_ends(ignore_ext=False)),
            "n_free_edge_cut": float(
                sum(
                    1
                    for e, v in self.edge_shells.items()
                    if len(v) == 1 and self.edge_both_ext(e)
                )
            ),
        }


#: ``graph_features`` 가 반환하는 키 순서 — 벡터화할 때의 기준.
GRAPH_FEATURE_NAMES: tuple[str, ...] = (
    "n_node", "n_shell", "n_quad", "n_tri", "tri_frac", "n_bar", "n_beam",
    "aspect_max", "aspect_mean", "aspect_p90", "skew_max", "skew_mean",
    "warp_max", "warp_mean", "taper_min", "angle_min", "angle_max",
    "n_bad_angle", "njac_min", "njac_mean", "det_ratio_min", "area_cv", "area_ratio",
    "n_orient", "dih_max", "dih_mean", "n_free_edge", "n_nonmanifold_edge",
    "deg_shell_max", "deg_line_max", "n_irregular_node",
    "ext_frac", "n_bad_internal", "n_bad_ext_only",
    "n_dangling_internal", "n_dangling_all", "n_free_edge_cut",
)


def graph_feature_vector(rep: MeshRep) -> np.ndarray:
    """``graph_features()`` 를 ``GRAPH_FEATURE_NAMES`` 순서의 float32 벡터로."""
    d = rep.graph_features()
    return np.array([d[k] for k in GRAPH_FEATURE_NAMES], dtype=np.float32)


def load_meshrep(bdf_path: str, json_path: str | None = None) -> MeshRep:
    """편의 함수 — 파일 경로에서 바로 MeshRep."""
    return MeshRep(load_raw_mesh(bdf_path, json_path))

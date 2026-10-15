"""수정 전/후 메쉬 쌍 — 로딩과 대응 찾기.

`data_folder/2026_10_08_n509` 는 비정상으로 판정된 메쉬를 **사람이 직접 고친** 결과다
(508쌍). 이름이 `<원본이름>_good` 이라 쌍은 이름으로 바로 맺어진다.

사람이 고치면서 요소·절점이 지워지거나 새로 생겼고 번호도 일부 바뀌었다. 그래서 전/후를
비교하려면 **절점 대응**을 먼저 찾아야 한다. 다행히 실측으로 원본 절점의 98.3%가 같은
번호로 남아 있고 그중 99.6%는 좌표도 그대로다 — 번호로 맞추고, 번호가 안 맞는 나머지만
좌표로 맞추면 거의 전부 대응된다.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

import gae_paths as gp
from bdf_fixed import load_fixed_mesh
from bdf_io import RawMesh, discover_pairs, load_raw_mesh

#: 사람이 고친 메쉬가 들어 있는 폴더
GOOD_DIR = os.path.join(gp.ROOT, "data_folder", "2026_10_08_n509")

#: 좌표가 같다고 볼 허용오차(mm). 선체 좌표가 24만 mm 단위라 1e-3 은 충분히 엄격하다.
XYZ_TOL = 1e-3


def _key(p) -> tuple:
    return tuple(np.round(np.asarray(p, dtype=float) / XYZ_TOL).astype(np.int64))


def discover_good_pairs(folder: str = GOOD_DIR) -> dict[str, tuple[str, str]]:
    """``{원본이름: (good bdf, good json)}``."""
    out: dict[str, tuple[str, str]] = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".bdf"):
            continue
        stem = fn[:-4]
        js = os.path.join(folder, stem + ".json")
        if not os.path.exists(js):
            continue
        out[re.sub(r"_good$", "", stem)] = (os.path.join(folder, fn), js)
    return out


@dataclass
class NodeMatch:
    """원본 ↔ 수정본 절점 대응."""

    #: 원본 절점 id -> 수정본 절점 id
    fwd: dict[int, int] = field(default_factory=dict)
    #: 번호가 그대로이고 좌표도 그대로인 절점
    kept: set[int] = field(default_factory=set)
    #: 번호는 같고 좌표만 움직인 절점 (병합으로 설명되는 것은 여기서 뺀다)
    moved: set[int] = field(default_factory=set)
    #: 좌표로만 맞춘 절점 (번호가 바뀐 것)
    renumbered: dict[int, int] = field(default_factory=dict)
    #: 수정본에서 사라진 원본 절점 (병합으로 흡수된 것은 여기서 뺀다)
    deleted: set[int] = field(default_factory=set)
    #: 수정본에만 있는 절점 (새로 생성)
    added: set[int] = field(default_factory=set)
    #: **절점 병합** — {흡수된 절점: 살아남은 절점}.
    #: 판정은 `nodemap.build_node_map` 이 한다 (연결관계 기준). 흡수된 쪽을 키로 두는 건
    #: 한 절점이 둘 이상을 흡수하는 경우가 있기 때문이다 (실측 23건, 최대 3개).
    absorbed: dict[int, int] = field(default_factory=dict)
    #: 사라진 절점을 **새로 만든 절점** 이 대신한 경우 (병합이 아니라 교체)
    replaced_by_new: dict[int, int] = field(default_factory=dict)
    #: 병합으로 살아남은 절점이 제자리를 벗어난 경우 -> 어디로 갔는지
    survivor_landed: dict[int, str] = field(default_factory=dict)
    #: 판정 근거 {흡수된 절점: "쉘교체"|"좌표+인접"|"퇴화추론"|"쉘교체(비인접)"|...}
    basis: dict[int, str] = field(default_factory=dict)
    #: 검산 — 병합된 절점을 쓰던 쉘이 대응과 맞아떨어지는 비율. 1.0 이어야 정상이지만,
    #: 병합과 동시에 주변을 다시 깔면 1.0 이 안 나온다 (실측 508쌍 중 98쌍).
    explained: float = 1.0
    #: 다른 절점과 좌표가 겹쳐 버린 절점 — 둘 다 수정본에 남아 있다.
    #: 병합이 아니라 작업 실수로 보인다 (실측 5건).
    coincident: dict[int, int] = field(default_factory=dict)

    @property
    def survivors(self) -> dict[int, list[int]]:
        """살아남은 절점 -> 그가 흡수한 절점들 (1:N)."""
        out: dict[int, list[int]] = {}
        for b, a in self.absorbed.items():
            out.setdefault(a, []).append(b)
        return {k: sorted(v) for k, v in out.items()}

    def summary(self) -> str:
        return (f"유지 {len(self.kept)} / 이동 {len(self.moved)} / 병합 {len(self.absorbed)} / "
                f"번호변경 {len(self.renumbered)} / 삭제 {len(self.deleted)} / "
                f"신규 {len(self.added)} / 신규교체 {len(self.replaced_by_new)} / "
                f"좌표겹침 {len(self.coincident)}")


def match_nodes(before: RawMesh, after: RawMesh) -> NodeMatch:
    """두 메쉬의 절점을 맞춘다. 번호 우선, 남은 것은 좌표로."""
    m = NodeMatch()
    bkeys = {n: _key(p) for n, p in before.coords.items()}
    akeys = {n: _key(p) for n, p in after.coords.items()}

    for n in before.coords:
        if n in after.coords:
            m.fwd[n] = n
            (m.kept if bkeys[n] == akeys[n] else m.moved).add(n)

    # 번호로 못 맞춘 것끼리 좌표로 맞춘다
    left_b = [n for n in before.coords if n not in m.fwd]
    used_a = set(m.fwd.values())
    by_xyz: dict[tuple, list[int]] = {}
    for n, k in akeys.items():
        if n not in used_a:
            by_xyz.setdefault(k, []).append(n)
    for n in left_b:
        cand = by_xyz.get(bkeys[n])
        if cand:
            a = cand.pop()
            m.fwd[n] = a
            m.renumbered[n] = a
        else:
            m.deleted.add(n)

    m.added = set(after.coords) - set(m.fwd.values())

    # ── 절점 병합 판정 ────────────────────────────────────────────────
    # 좌표로 판정하면 안 된다. 사람은 두 절점을 합친 뒤 자리를 다시 잡기도 하고
    # (`_1081`), 셋을 한 점으로 모으기도 한다 (`_0021`). 좌표만 보면 450건 중
    # 99건밖에 못 잡았다. 연결관계로 푸는 일은 nodemap 에 맡긴다.
    from nodemap import build_node_map, where_survivor_landed

    nmap = build_node_map(before, after)
    m.absorbed = dict(nmap.absorbed)
    m.replaced_by_new = dict(nmap.replaced_by_new)
    m.basis = dict(nmap.basis)
    m.explained = nmap.explained
    m.deleted = set(nmap.dropped)  # 아무도 이어받지 않은 절점만 '삭제' 로 남는다

    # 병합으로 살아남은 절점이 제자리를 벗어났다면 그건 '좌표 이동' 이 아니라
    # 병합의 일부다. 따로 적어 두고 moved 에서는 뺀다.
    m.survivor_landed = {s: w for s, w in where_survivor_landed(before, after, nmap).items()
                         if w != "제자리"}
    m.moved -= set(m.survivor_landed)

    # 좌표가 겹쳐 버린 절점 — 병합이 아니라 작업 실수로 보인다
    seen: dict[tuple, int] = {}
    for n in sorted(after.coords):
        k = akeys[n]
        if k in seen:
            m.coincident[n] = seen[k]
        else:
            seen[k] = n
    return m


@dataclass
class Pair:
    """원본과 사람이 고친 결과, 그리고 절점 대응."""

    name: str
    before: RawMesh
    after: RawMesh
    nodes: NodeMatch

    @staticmethod
    def _shell_keys(mesh: RawMesh, remap: dict[int, int] | None = None) -> set:
        """쉘을 '절점 집합' 으로 식별한다 — 요소 번호는 수정 과정에서 바뀐다."""
        out = set()
        for _, _, nodes in mesh.shells:
            ns = [remap.get(n, -n) if remap else n for n in nodes]
            out.add(frozenset(ns))
        return out

    def shell_diff(self) -> tuple[set, set, set]:
        """(그대로 남은 쉘, 사라진 쉘, 새로 생긴 쉘) — 원본 절점 번호 기준."""
        b = self._shell_keys(self.before)
        a = self._shell_keys(self.after)
        # after 의 절점을 before 번호로 되돌려 비교한다
        rev = {v: k for k, v in self.nodes.fwd.items()}
        a_in_b = {frozenset(rev.get(n, -n) for n in k) for k in a}
        return b & a_in_b, b - a_in_b, a_in_b - b

    def summary(self) -> dict:
        kept, removed, created = self.shell_diff()
        tri = lambda m: sum(1 for _, t, _ in m.shells if t == "CTRIA3")
        return dict(
            name=self.name,
            n_node_before=len(self.before.coords), n_node_after=len(self.after.coords),
            n_shell_before=len(self.before.shells), n_shell_after=len(self.after.shells),
            n_tri_before=tri(self.before), n_tri_after=tri(self.after),
            n_line_before=len(self.before.lines), n_line_after=len(self.after.lines),
            node_kept=len(self.nodes.kept), node_moved=len(self.nodes.moved),
            node_renumbered=len(self.nodes.renumbered),
            node_deleted=len(self.nodes.deleted), node_added=len(self.nodes.added),
            node_absorbed=len(self.nodes.absorbed),
            node_replaced_by_new=len(self.nodes.replaced_by_new),
            node_coincident=len(self.nodes.coincident),
            explained=self.nodes.explained,
            shell_kept=len(kept), shell_removed=len(removed), shell_created=len(created),
        )


def load_pair(name: str, good: dict[str, tuple[str, str]] | None = None,
              orig: dict[str, tuple[str, str]] | None = None) -> Pair:
    good = good if good is not None else discover_good_pairs()
    if orig is None:
        orig, _, _ = discover_pairs(gp.FE_DIR)
    before = load_raw_mesh(*orig[name])
    after = load_fixed_mesh(*good[name])
    return Pair(name=name, before=before, after=after, nodes=match_nodes(before, after))


def all_pairs() -> list[str]:
    """원본이 존재하는 수정 쌍의 이름 목록."""
    good = discover_good_pairs()
    orig, _, _ = discover_pairs(gp.FE_DIR)
    return sorted(n for n in good if n in orig)

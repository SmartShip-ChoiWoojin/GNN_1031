"""수정 전/후 메쉬 쌍 — 로딩과 절점 대응 찾기.

`data_folder/2026_10_08_n509` 는 비정상으로 판정된 메쉬를 **사람이 직접 고친** 결과다
(508쌍). 파일 이름이 `<원본이름>_good` 이라 쌍은 이름으로 바로 맺어진다.

사람이 고치면서 요소·절점이 지워지거나 새로 생겼다. 그래서 전/후를 비교하려면
**절점 대응**을 먼저 찾아야 한다. 실측으로 원본 절점의 98.3%가 같은 번호로 남아 있고
그중 99.6%는 좌표도 그대로다 — 번호로 맞추고, 번호가 안 맞는 나머지만 좌표로 맞추면
거의 전부 대응된다.
"""

from __future__ import annotations

import os
import re

import numpy as np

import gae_paths
from bdf_fixed import load_fixed_mesh
from bdf_io import discover_pairs, load_raw_mesh
from nodemap import build_node_map, where_survivor_landed

#: 사람이 고친 메쉬가 들어 있는 폴더
GOOD_DIR = gae_paths.GOOD_DIR

#: 좌표가 같다고 볼 허용오차(mm). 선체 좌표가 24만 mm 단위라 1e-3 은 충분히 엄격하다.
XYZ_TOL = 1e-3


def coordinate_key(point) -> tuple:
    """좌표를 허용오차 단위로 끊어 사전 키로 쓴다."""
    return tuple(np.round(np.asarray(point, dtype=float) / XYZ_TOL).astype(np.int64))


def discover_good_pairs(folder: str = GOOD_DIR) -> dict:
    """{원본이름: (수정본 bdf 경로, 수정본 json 경로)}."""
    result = {}
    for file_name in sorted(os.listdir(folder)):
        if not file_name.endswith(".bdf"):
            continue
        stem = file_name[:-4]
        json_path = os.path.join(folder, stem + ".json")
        if not os.path.exists(json_path):
            continue
        original_name = re.sub(r"_good$", "", stem)
        result[original_name] = (os.path.join(folder, file_name), json_path)
    return result


class NodeMatch:
    """원본 ↔ 수정본 절점 대응."""

    def __init__(self):
        #: {원본 절점: 수정본 절점}
        self.fwd = {}
        #: 번호가 그대로이고 좌표도 그대로인 절점
        self.kept = set()
        #: 번호는 같고 좌표만 움직인 절점 (병합으로 설명되는 것은 여기서 뺀다)
        self.moved = set()
        #: {원본 절점: 수정본 절점} 좌표로만 맞춘 절점 (번호가 바뀐 것)
        self.renumbered = {}
        #: 수정본에서 사라진 원본 절점 (병합으로 흡수된 것은 여기서 뺀다)
        self.deleted = set()
        #: 수정본에만 있는 절점 (새로 생성)
        self.added = set()
        #: **절점 병합** — {흡수된 절점: 살아남은 절점}. 판정은 nodemap.build_node_map 이 한다.
        #: 흡수된 쪽을 키로 두는 건 한 절점이 둘 이상을 흡수하는 경우가 있기 때문이다.
        self.absorbed = {}
        #: {사라진 절점: 새로 만든 절점} — 병합이 아니라 교체
        self.replaced_by_new = {}
        #: {살아남은 절점: 어디로 갔는지} — 병합으로 살아남은 절점이 제자리를 벗어난 경우
        self.survivor_landed = {}
        #: {흡수된 절점: 판정 근거}
        self.basis = {}
        #: 검산 — 병합된 절점을 쓰던 쉘이 대응과 맞아떨어지는 비율
        self.explained = 1.0
        #: {절점: 좌표가 겹친 다른 절점} — 작업 실수로 보인다 (실측 5건)
        self.coincident = {}

    @property
    def survivors(self) -> dict:
        """{살아남은 절점: [그가 흡수한 절점들]}."""
        result = {}
        for absorbed_node, survivor_node in self.absorbed.items():
            if survivor_node not in result:
                result[survivor_node] = []
            result[survivor_node].append(absorbed_node)

        sorted_result = {}
        for survivor_node, absorbed_list in result.items():
            sorted_result[survivor_node] = sorted(absorbed_list)
        return sorted_result

    def summary(self) -> str:
        return (f"유지 {len(self.kept)} / 이동 {len(self.moved)} / 병합 {len(self.absorbed)} / "
                f"번호변경 {len(self.renumbered)} / 삭제 {len(self.deleted)} / "
                f"신규 {len(self.added)} / 신규교체 {len(self.replaced_by_new)} / "
                f"좌표겹침 {len(self.coincident)}")


def match_nodes(before, after) -> NodeMatch:
    """두 메쉬의 절점을 맞춘다. 번호 우선, 남은 것은 좌표로."""
    match = NodeMatch()

    before_keys = {}
    for node_id, point in before.coords.items():
        before_keys[node_id] = coordinate_key(point)
    after_keys = {}
    for node_id, point in after.coords.items():
        after_keys[node_id] = coordinate_key(point)

    # 1) 같은 번호끼리
    for node_id in before.coords:
        if node_id in after.coords:
            match.fwd[node_id] = node_id
            if before_keys[node_id] == after_keys[node_id]:
                match.kept.add(node_id)
            else:
                match.moved.add(node_id)

    # 2) 번호로 못 맞춘 것끼리 좌표로
    left_before = []
    for node_id in before.coords:
        if node_id not in match.fwd:
            left_before.append(node_id)
    used_after = set(match.fwd.values())

    after_nodes_by_position = {}
    for node_id, key in after_keys.items():
        if node_id in used_after:
            continue
        if key not in after_nodes_by_position:
            after_nodes_by_position[key] = []
        after_nodes_by_position[key].append(node_id)

    for node_id in left_before:
        candidates = after_nodes_by_position.get(before_keys[node_id])
        if candidates:
            after_node = candidates.pop()
            match.fwd[node_id] = after_node
            match.renumbered[node_id] = after_node
        else:
            match.deleted.add(node_id)

    match.added = set(after.coords) - set(match.fwd.values())

    # 3) 절점 병합 판정 — 좌표가 아니라 연결관계로 (nodemap)
    # 사람은 두 절점을 합친 뒤 자리를 다시 잡기도 하고(`_1081`), 셋을 한 점으로
    # 모으기도 한다(`_0021`). 좌표만 보면 450건 중 99건밖에 못 잡았다.
    node_map = build_node_map(before, after)
    match.absorbed = dict(node_map.absorbed)
    match.replaced_by_new = dict(node_map.replaced_by_new)
    match.basis = dict(node_map.basis)
    match.explained = node_map.explained
    # 아무도 이어받지 않은 절점만 '삭제' 로 남는다
    match.deleted = set(node_map.dropped)

    # 병합으로 살아남은 절점이 제자리를 벗어났다면 그건 '좌표 이동' 이 아니라
    # 병합의 일부다. 따로 적어 두고 moved 에서는 뺀다.
    landed = where_survivor_landed(before, after, node_map)
    for survivor_node, where in landed.items():
        if where != "제자리":
            match.survivor_landed[survivor_node] = where
    match.moved = match.moved - set(match.survivor_landed)

    # 4) 좌표가 겹쳐 버린 절점 — 병합이 아니라 작업 실수로 보인다
    first_node_at_position = {}
    for node_id in sorted(after.coords):
        key = after_keys[node_id]
        if key in first_node_at_position:
            match.coincident[node_id] = first_node_at_position[key]
        else:
            first_node_at_position[key] = node_id
    return match


class Pair:
    """원본과 사람이 고친 결과, 그리고 절점 대응."""

    def __init__(self, name: str, before, after, nodes: NodeMatch):
        self.name = name
        #: 원본 (RawMesh)
        self.before = before
        #: 사람이 고친 결과 (RawMesh)
        self.after = after
        #: 절점 대응 (NodeMatch)
        self.nodes = nodes

    @staticmethod
    def shell_keys(mesh) -> set:
        """쉘을 '절점 집합' 으로 식별한다 — 요소 번호는 수정 과정에서 바뀐다."""
        result = set()
        for element_id, element_type, node_ids in mesh.shells:
            result.add(frozenset(node_ids))
        return result

    def shell_diff(self):
        """(그대로 남은 쉘, 사라진 쉘, 새로 생긴 쉘) — 원본 절점 번호 기준."""
        before_keys = self.shell_keys(self.before)
        after_keys = self.shell_keys(self.after)

        # after 의 절점을 before 번호로 되돌려 비교한다. 대응이 없는 절점은 음수로 둔다.
        after_to_before = {}
        for before_node, after_node in self.nodes.fwd.items():
            after_to_before[after_node] = before_node

        after_in_before_numbers = set()
        for shell_key in after_keys:
            renamed = []
            for node_id in shell_key:
                if node_id in after_to_before:
                    renamed.append(after_to_before[node_id])
                else:
                    renamed.append(-node_id)
            after_in_before_numbers.add(frozenset(renamed))

        kept = before_keys & after_in_before_numbers
        removed = before_keys - after_in_before_numbers
        created = after_in_before_numbers - before_keys
        return kept, removed, created

    def summary(self) -> dict:
        kept, removed, created = self.shell_diff()
        return dict(
            name=self.name,
            n_node_before=len(self.before.coords), n_node_after=len(self.after.coords),
            n_shell_before=len(self.before.shells), n_shell_after=len(self.after.shells),
            n_tri_before=count_triangles(self.before), n_tri_after=count_triangles(self.after),
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


def count_triangles(mesh) -> int:
    """CTRIA3 요소 수."""
    triangle_count = 0
    for element_id, element_type, node_ids in mesh.shells:
        if element_type == "CTRIA3":
            triangle_count = triangle_count + 1
    return triangle_count


def load_pair(name: str, good_paths: dict | None = None, original_paths: dict | None = None) -> Pair:
    """이름 하나로 원본·수정본을 읽고 절점 대응까지 구한다.

    good_paths     : discover_good_pairs() 결과 (없으면 새로 찾는다)
    original_paths : bdf_io.discover_pairs(FE_DIR)[0] 결과 (없으면 새로 찾는다)
    """
    if good_paths is None:
        good_paths = discover_good_pairs()
    if original_paths is None:
        original_paths, _, _ = discover_pairs(gae_paths.FE_DIR)

    original_bdf, original_json = original_paths[name]
    good_bdf, good_json = good_paths[name]
    before = load_raw_mesh(original_bdf, original_json)
    after = load_fixed_mesh(good_bdf, good_json)
    return Pair(name=name, before=before, after=after, nodes=match_nodes(before, after))


def all_pairs() -> list:
    """원본이 존재하는 수정 쌍의 이름 목록 (정렬)."""
    good_paths = discover_good_pairs()
    original_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    names = []
    for name in good_paths:
        if name in original_paths:
            names.append(name)
    return sorted(names)

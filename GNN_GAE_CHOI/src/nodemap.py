"""사라진 절점이 어느 절점에 흡수됐는지 찾는다 (절점 병합 판정).

좌표로 판정하면 안 된다. 사람은 두 절점을 합친 뒤 그 자리를 다시 잡기도 하고, 셋을 한
점으로 모으기도 한다. 실측 실패 사례:

  - `_1081`: 22196 이 22197 을 흡수했는데 최종 위치는 22197 자리에서 45mm 떨어짐
  - `_0021`: 15278 + 15279 + 23439 셋이 하나로. 변으로 이어진 건 15279 인데
    최종 위치는 23439 자리

그래서 **연결관계**로 푼다. "사라진 쉘이 새 쉘이 될 때 그 자리를 누가 이어받았나" 를
반복적으로 좁히고, 마지막에 **검산**한다 — 찾은 대응을 원본 쉘에 적용했을 때 실제로
수정본 쉘이 나오는지.

용어:
    before   원본 메쉬 (RawMesh)
    after    사람이 고친 메쉬 (RawMesh)
    흡수     사라진 절점의 자리를 다른 절점이 이어받은 것 (절점 병합)
"""

from __future__ import annotations

import numpy as np

import gae_paths  # noqa: F401  (부모 src 를 sys.path 에 넣는다)

#: 좌표가 같다고 볼 허용오차(mm)
XYZ_TOL = 1e-3

#: 대응을 좁히는 반복 횟수 상한. 보통 2~3 회에 끝난다.
MAX_ROUNDS = 8


class NodeMap:
    """원본 절점 → 수정본 절점 대응."""

    def __init__(self):
        #: {흡수된 절점: 그 자리를 이어받은 절점}
        self.absorbed = {}
        #: 아무도 이어받지 않고 그냥 없어진 절점
        self.dropped = set()
        #: {흡수된 절점: 새로 만든 절점} — 이어받은 쪽이 원본에 없던 절점이면 병합이 아니라 교체
        self.replaced_by_new = {}
        #: {흡수된 절점: 판정 근거}
        #:   "좌표+인접"      사라진 절점 자리에 이웃 절점이 정확히 들어앉음
        #:   "쉘교체"         사라진 쉘이 새 쉘로 바뀌며 한 자리만 달라 직접 읽음 (강한 근거)
        #:   "쉘교체(비인접)"  위와 같지만 원래 이웃이 아니었음
        #:   "퇴화추론"       쉘이 찌그러져 없어져 같은 쉘의 다른 절점 중에서 골라냄 (약한 근거)
        #:   "퇴화추론+좌표"  퇴화추론이 동점이라 좌표로 가름
        self.basis = {}
        #: 검산 — 병합된 절점을 쓰던 쉘 중 대응과 맞아떨어지는 비율. 1.0 이어야 정상.
        self.explained = 0.0
        #: 원본에만 있는 쉘 수
        self.n_removed_shell = 0

    @property
    def survivors(self) -> dict:
        """{살아남은 절점: [그가 흡수한 절점들]} (한 절점이 여럿을 흡수할 수 있다)."""
        result = {}
        for absorbed_node, survivor_node in self.absorbed.items():
            if survivor_node not in result:
                result[survivor_node] = []
            result[survivor_node].append(absorbed_node)

        sorted_result = {}
        for survivor_node, absorbed_list in result.items():
            sorted_result[survivor_node] = sorted(absorbed_list)
        return sorted_result


def coordinate_key(point) -> tuple:
    """좌표를 허용오차 단위로 끊어 사전 키로 쓴다 (같은 자리인지 비교용)."""
    return tuple(np.round(np.asarray(point, dtype=float) / XYZ_TOL).astype(np.int64))


def shell_node_sets(mesh) -> list:
    """메쉬의 쉘마다 절점 집합(frozenset)."""
    result = []
    for element_id, element_type, node_ids in mesh.shells:
        result.append(frozenset(node_ids))
    return result


def build_edge_neighbors(mesh) -> dict:
    """{절점: 쉘 변으로 이어진 이웃 절점 집합}."""
    neighbors = {}
    for element_id, element_type, node_ids in mesh.shells:
        corner_count = len(node_ids)
        for corner_index in range(corner_count):
            node_a = node_ids[corner_index]
            node_b = node_ids[(corner_index + 1) % corner_count]
            if node_a not in neighbors:
                neighbors[node_a] = set()
            if node_b not in neighbors:
                neighbors[node_b] = set()
            neighbors[node_a].add(node_b)
            neighbors[node_b].add(node_a)
    return neighbors


def is_neighbor(neighbors: dict, node_a: int, node_b: int) -> bool:
    """두 절점이 원본에서 쉘 변으로 이어져 있었나."""
    if node_a not in neighbors:
        return False
    return node_b in neighbors[node_a]


def add_vote(votes: dict, node: int, candidate: int, weight: int) -> None:
    """``votes[node][candidate]`` 에 ``weight`` 표를 더한다 (없으면 만든다)."""
    if node not in votes:
        votes[node] = {}
    if candidate not in votes[node]:
        votes[node][candidate] = 0
    votes[node][candidate] = votes[node][candidate] + weight


def most_voted(candidate_votes: dict) -> int:
    """표가 가장 많은 후보. 동점이면 먼저 표를 받은 후보 (collections.Counter.most_common(1) 과 같은 규칙)."""
    best_candidate = None
    best_count = None
    for candidate, count in candidate_votes.items():
        if best_count is None or count > best_count:
            best_candidate = candidate
            best_count = count
    return best_candidate


def build_node_map(before, after) -> NodeMap:
    """연결관계로 절점 대응을 푼다.

    번호가 바뀐 절점은 이 데이터셋에 없다(실측 0건)는 사실을 쓴다 — 양쪽에 같은 번호로
    있으면 같은 절점이다. 그래서 풀어야 할 것은 **사라진 절점의 행방**뿐이다.
    """
    node_map = NodeMap()
    gone_nodes = set(before.coords) - set(after.coords)
    new_nodes = set(after.coords) - set(before.coords)

    before_shell_sets = set(shell_node_sets(before))
    after_shell_sets = set(shell_node_sets(after))

    # 원본에만 있는 쉘(사라진 쉘)과 수정본에만 있는 쉘(새 쉘)
    removed_shells = []
    for element_id, element_type, node_ids in before.shells:
        if frozenset(node_ids) not in after_shell_sets:
            removed_shells.append(list(node_ids))
    created_shells = []
    for element_id, element_type, node_ids in after.shells:
        if frozenset(node_ids) not in before_shell_sets:
            created_shells.append(set(node_ids))
    node_map.n_removed_shell = len(removed_shells)

    if len(gone_nodes) == 0:
        if len(removed_shells) == 0:
            node_map.explained = 1.0
        else:
            node_map.explained = compute_explained(removed_shells, created_shells, {})
        return node_map

    # 원래 변으로 이어져 있던 이웃. 병합은 거의 전부 이웃끼리 일어나므로(실측 92%)
    # 후보를 고를 때 가장 강한 제약이 된다.
    neighbors = build_edge_neighbors(before)

    #: {흡수된 절점: 이어받은 절점} — 반복하며 채워 나간다
    mapping = {}

    # ── 0순위: 좌표 증거 ─────────────────────────────────────────────
    # 사라진 절점 B 의 자리에 **정확히** 들어앉은 이웃 절점이 있으면 그가 흡수자다.
    # 이건 쉘 모양이 어떻게 바뀌든 흔들리지 않는다. 실측 `_0032` 에서 쉘 투표가
    # 엉뚱한 절점(15339)을 가리켰는데, 실제 흡수자(26865)는 26866 자리로 정확히
    # 옮겨와 있었다 — 그 경우를 이 규칙이 잡는다.
    node_at_position = {}
    for node_id, point in after.coords.items():
        key = coordinate_key(point)
        if key not in node_at_position:
            node_at_position[key] = node_id
    for gone_node in sorted(gone_nodes):
        key = coordinate_key(before.coords[gone_node])
        if key not in node_at_position:
            continue
        candidate = node_at_position[key]
        if candidate == gone_node:
            continue
        if not is_neighbor(neighbors, gone_node, candidate):
            continue
        mapping[gone_node] = candidate
        node_map.basis[gone_node] = "좌표+인접"

    # ── 1·2순위: 쉘 투표를 반복 ──────────────────────────────────────
    for round_number in range(MAX_ROUNDS):
        # 1순위(strong): 사라진 쉘이 **새 쉘로 바뀐** 경우. 한 자리만 다르면 그 자리가 곧 대응이다.
        strong_votes = {}
        # 2순위(weak): 사라진 쉘이 **퇴화해 없어진** 경우. 두 절점이 하나로 합쳐지면 삼각형이
        # 찌그러져 사라지므로 바뀐 쉘이 안 생긴다. 이때는 "같은 쉘 안의 다른 절점 중
        # 하나로 합쳐졌다" 까지만 알 수 있어서, 여러 쉘의 표를 모아 좁힌다.
        # (실측 `_0021`: 15278+15279+23439 가 한 점으로 모이며 삼각형 2개가 퇴화)
        weak_votes = {}

        for shell_nodes in removed_shells:
            # 이 쉘에서 아직 행방을 모르는 절점
            unknown_nodes = []
            for node_id in shell_nodes:
                if node_id in gone_nodes and node_id not in mapping:
                    unknown_nodes.append(node_id)
            if len(unknown_nodes) != 1:
                # 모르는 자리가 없거나 둘 이상이면 이번 회차에는 판단을 보류한다
                continue
            unknown_node = unknown_nodes[0]

            # 나머지 절점은 지금까지 알아낸 대응으로 바꿔 둔다
            known_nodes = set()
            for node_id in shell_nodes:
                if node_id == unknown_node:
                    continue
                if node_id in mapping:
                    known_nodes.add(mapping[node_id])
                else:
                    known_nodes.add(node_id)

            found_in_new_shell = False
            for new_shell in created_shells:
                if len(new_shell) != len(shell_nodes):
                    continue
                if not known_nodes.issubset(new_shell):
                    continue
                remaining_nodes = new_shell - known_nodes
                if len(remaining_nodes) != 1:
                    continue
                candidate = list(remaining_nodes)[0]
                # 이웃이 아닌 후보는 표를 반만 준다. 쉘이 통째로 다시 깔리면
                # 사라진 쉘과 새 쉘이 **우연히** 한 자리만 다를 수 있는데,
                # 그때 엉뚱한 절점이 흡수자로 뽑힌다 (실측 `_0032`).
                if is_neighbor(neighbors, unknown_node, candidate):
                    add_vote(strong_votes, unknown_node, candidate, 2)
                else:
                    add_vote(strong_votes, unknown_node, candidate, 1)
                found_in_new_shell = True

            if not found_in_new_shell:
                for known_node in known_nodes:
                    if is_neighbor(neighbors, unknown_node, known_node):
                        add_vote(weak_votes, unknown_node, known_node, 2)
                    else:
                        add_vote(weak_votes, unknown_node, known_node, 1)

        picked_this_round = {}
        basis_this_round = {}

        for gone_node, candidate_votes in strong_votes.items():
            best_candidate = most_voted(candidate_votes)
            picked_this_round[gone_node] = best_candidate
            if is_neighbor(neighbors, gone_node, best_candidate):
                basis_this_round[gone_node] = "쉘교체"
            else:
                basis_this_round[gone_node] = "쉘교체(비인접)"

        for gone_node, candidate_votes in weak_votes.items():
            if gone_node in picked_this_round:
                continue
            best_count = max(candidate_votes.values())
            tied_candidates = []
            for candidate, count in candidate_votes.items():
                if count == best_count:
                    tied_candidates.append(candidate)
            tied_candidates = sorted(tied_candidates)

            if len(tied_candidates) == 1:
                picked_this_round[gone_node] = tied_candidates[0]
                basis_this_round[gone_node] = "퇴화추론"
            else:
                # 동점이면 좌표로만 가른다 — 흡수된 절점 자리에 최종적으로 놓인 쪽
                gone_point = np.asarray(before.coords[gone_node], dtype=float)
                exactly_there = []
                for candidate in tied_candidates:
                    if candidate not in after.coords:
                        continue
                    candidate_point = np.asarray(after.coords[candidate], dtype=float)
                    distance = float(np.linalg.norm(candidate_point - gone_point))
                    if distance < XYZ_TOL:
                        exactly_there.append(candidate)
                if len(exactly_there) == 1:
                    picked_this_round[gone_node] = exactly_there[0]
                    basis_this_round[gone_node] = "퇴화추론+좌표"

        if len(picked_this_round) == 0:
            break
        for gone_node, candidate in picked_this_round.items():
            mapping[gone_node] = candidate
        for gone_node, basis_text in basis_this_round.items():
            node_map.basis[gone_node] = basis_text

    # ── 결과 정리 ────────────────────────────────────────────────────
    for gone_node, candidate in mapping.items():
        if candidate in new_nodes:
            node_map.replaced_by_new[gone_node] = candidate
        else:
            node_map.absorbed[gone_node] = candidate
    node_map.dropped = gone_nodes - set(mapping)
    node_map.explained = compute_explained(removed_shells, created_shells, mapping)
    return node_map


def compute_explained(removed_shells: list, created_shells: list, mapping: dict) -> float:
    """검산 — **병합된 절점을 쓰던** 쉘이 대응과 모순되지 않는가.

    대응이 맞다면 그런 쉘은 매핑 후 (a) 새 쉘이 되거나 (b) 절점이 겹쳐 퇴화해 없어진다.
    둘 다 아니면 대응이 틀린 것이다.

    병합된 절점을 **안 쓰던** 쉘은 세지 않는다 — 그건 사람이 그냥 다시 깐 것이라
    절점 대응으로 설명될 이유가 없다.
    """
    created_set = set()
    for new_shell in created_shells:
        created_set.add(frozenset(new_shell))

    related_shells = []
    for shell_nodes in removed_shells:
        uses_mapped_node = False
        for node_id in shell_nodes:
            if node_id in mapping:
                uses_mapped_node = True
        if uses_mapped_node:
            related_shells.append(shell_nodes)

    if len(related_shells) == 0:
        return 1.0

    consistent_count = 0
    for shell_nodes in related_shells:
        mapped_nodes = []
        for node_id in shell_nodes:
            if node_id in mapping:
                mapped_nodes.append(mapping[node_id])
            else:
                mapped_nodes.append(node_id)
        collapsed = len(set(mapped_nodes)) < len(mapped_nodes)
        became_new_shell = frozenset(mapped_nodes) in created_set
        if collapsed or became_new_shell:
            consistent_count = consistent_count + 1
    return consistent_count / len(related_shells)


# ──────────────────────────────────────────────────────────────────────────
def where_survivor_landed(before, after, node_map: NodeMap) -> dict:
    """살아남은 절점이 최종적으로 어디에 자리잡았나. {살아남은 절점: 설명}.

    흡수한 절점 **전부** 와 자기 원래 자리를 모두 비교한다. 하나만 보고 판정하면
    `_0021`(셋이 하나로 합쳐진 경우)처럼 오판한다.
    """
    result = {}
    for survivor_node, absorbed_list in node_map.survivors.items():
        if survivor_node not in after.coords:
            continue
        if survivor_node not in before.coords:
            continue

        final_point = np.asarray(after.coords[survivor_node], dtype=float)
        original_point = np.asarray(before.coords[survivor_node], dtype=float)
        if float(np.linalg.norm(final_point - original_point)) < XYZ_TOL:
            result[survivor_node] = "제자리"
            continue

        landed_on_absorbed = False
        for absorbed_node in absorbed_list:
            absorbed_point = np.asarray(before.coords[absorbed_node], dtype=float)
            if float(np.linalg.norm(final_point - absorbed_point)) < XYZ_TOL:
                landed_on_absorbed = True
                break
        if landed_on_absorbed:
            result[survivor_node] = "흡수한 절점 자리"
            continue

        related_points = [before.coords[survivor_node]]
        for absorbed_node in absorbed_list:
            related_points.append(before.coords[absorbed_node])
        middle_point = np.mean(np.asarray(related_points, dtype=float), axis=0)
        if float(np.linalg.norm(final_point - middle_point)) < XYZ_TOL:
            result[survivor_node] = "관련 절점들의 중점"
        else:
            result[survivor_node] = "그 밖의 위치"
    return result

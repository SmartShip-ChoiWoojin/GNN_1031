"""두 메쉬의 차이 — 채점과 화면 강조가 **같은 정의**를 쓰도록 한 곳에 모은다.

비교는 절점 **번호가 아니라 좌표**로 한다. 사람은 두 절점을 합칠 때 남길 번호를
일정하게 고르지 않는다 — 흡수한 쪽 번호를 흡수된 자리로 옮기기도 하고(99건), 제 번호를
제자리에 두기도 한다(343건). 결과 메쉬는 기하적으로 똑같은데 번호로 비교하면 "다르다" 가
나온다. 그래서

  - 요소/1D 는 꼭짓점 **좌표 집합**으로 식별한다 (번호 무관)
  - 절점은 **좌표**로 식별한다 — "어느 자리가 사라졌나 / 새로 생겼나"
  - 다만 화면의 "이동" 표시는 번호로 판정한다. 같은 번호가 다른 자리로 갔다는 사실이
    사람의 작업을 읽는 데는 더 직관적이라서다 (사람 결과에만 나온다 — 재생성은 좌표를
    하나도 안 옮긴다)

    signature(a) == signature(b)   ⇔ 두 메쉬가 기하·연결·1D 까지 같다
    diff(before, after)            → 지운 요소 / 만든 요소 / 사라진 자리 / 새 자리 / 옮긴 절점
"""

from __future__ import annotations

import numpy as np

#: 같은 자리로 볼 허용오차(mm). pairs.XYZ_TOL 과 같다 — 좌표가 24만 mm 단위라 충분히 엄격.
XYZ_TOL = 1e-3


def pkey(point) -> tuple:
    """좌표를 허용오차 단위로 끊은 사전 키 ("position key")."""
    return tuple(np.round(np.asarray(point, dtype=float) / XYZ_TOL).astype(np.int64))


def element_position_key(mesh, node_ids) -> frozenset:
    """요소(쉘 또는 1D)를 꼭짓점 좌표 키의 집합으로 식별한다 (번호 무관)."""
    position_keys = []
    for node_id in node_ids:
        position_keys.append(pkey(mesh.coords[node_id]))
    return frozenset(position_keys)


class MeshSignature:
    """번호와 무관한 메쉬 서명. 두 서명이 같으면 같은 메쉬다.

    nodes   절점 좌표 키의 집합
    shells  쉘마다 '꼭짓점 좌표 키 집합' 의 집합
    lines   1D 마다 '양 끝 좌표 키 집합' 의 집합
    """

    def __init__(self, nodes: frozenset, shells: frozenset, lines: frozenset):
        self.nodes = nodes
        self.shells = shells
        self.lines = lines

    def __eq__(self, other) -> bool:
        if not isinstance(other, MeshSignature):
            return False
        if self.nodes != other.nodes:
            return False
        if self.shells != other.shells:
            return False
        if self.lines != other.lines:
            return False
        return True

    def __hash__(self) -> int:
        return hash((self.nodes, self.shells, self.lines))

    def __len__(self) -> int:
        return len(self.shells)


def signature(mesh) -> MeshSignature:
    """``RawMesh`` / ``EditableMesh`` 어느 쪽이든 받는다 (coords/shells/lines 만 쓴다)."""
    node_keys = []
    for point in mesh.coords.values():
        node_keys.append(pkey(point))

    shell_keys = []
    for shell in mesh.shells:
        shell_keys.append(element_position_key(mesh, shell[2]))

    line_keys = []
    for line in mesh.lines:
        line_keys.append(element_position_key(mesh, line[2]))

    return MeshSignature(frozenset(node_keys), frozenset(shell_keys), frozenset(line_keys))


# ──────────────────────────────────────────────────────────────────────────
class MeshDiff:
    """before → after 에서 바뀐 것. 인덱스는 각 메쉬의 ``shells`` / ``lines`` 목록 위치."""

    def __init__(self):
        #: before 기준 쉘 인덱스 — after 에 같은 좌표로 없는 쉘
        self.removed_shells = []
        #: after 기준 쉘 인덱스 — before 에 같은 좌표로 없던 쉘
        self.created_shells = []
        #: before 기준 1D 인덱스
        self.removed_lines = []
        #: after 기준 1D 인덱스
        self.created_lines = []
        #: {before 절점 번호: 설명} — 그 **자리**가 after 에 없는 절점
        self.deleted_nodes = {}
        #: {after 절점 번호: 설명} — before 에 없던 **자리**
        self.added_nodes = {}
        #: {절점 번호: (출발 좌표, 도착 좌표, 설명)} — 같은 번호가 자리를 옮긴 절점
        self.moved_nodes = {}

    @property
    def n_changes(self) -> int:
        return (len(self.removed_shells) + len(self.created_shells)
                + len(self.removed_lines) + len(self.created_lines)
                + len(self.deleted_nodes) + len(self.added_nodes) + len(self.moved_nodes))


def diff(before, after, absorbed: dict | None = None) -> MeshDiff:
    """before → after 의 차이.

    absorbed: {사라진 절점: 이어받은 절점} — 설명 문구에만 쓴다. 재생성 결과면 연산
        기록에서, 사람 결과면 ``pairs.NodeMatch.absorbed`` 에서 온다.
    """
    if absorbed is None:
        absorbed = {}
    result = MeshDiff()

    # ── 쉘 ──
    before_shell_keys = []
    for shell in before.shells:
        before_shell_keys.append(element_position_key(before, shell[2]))
    after_shell_keys = []
    for shell in after.shells:
        after_shell_keys.append(element_position_key(after, shell[2]))
    before_shell_set = set(before_shell_keys)
    after_shell_set = set(after_shell_keys)

    for shell_index in range(len(before_shell_keys)):
        if before_shell_keys[shell_index] not in after_shell_set:
            result.removed_shells.append(shell_index)
    for shell_index in range(len(after_shell_keys)):
        if after_shell_keys[shell_index] not in before_shell_set:
            result.created_shells.append(shell_index)

    # ── 1D ──
    before_line_keys = []
    for line in before.lines:
        before_line_keys.append(element_position_key(before, line[2]))
    after_line_keys = []
    for line in after.lines:
        after_line_keys.append(element_position_key(after, line[2]))
    before_line_set = set(before_line_keys)
    after_line_set = set(after_line_keys)

    for line_index in range(len(before_line_keys)):
        if before_line_keys[line_index] not in after_line_set:
            result.removed_lines.append(line_index)
    for line_index in range(len(after_line_keys)):
        if after_line_keys[line_index] not in before_line_set:
            result.created_lines.append(line_index)

    # ── 절점 ──
    before_node_at_position = {}
    for node_id, point in before.coords.items():
        before_node_at_position[pkey(point)] = node_id
    after_node_at_position = {}
    for node_id, point in after.coords.items():
        after_node_at_position[pkey(point)] = node_id

    for node_id, point in before.coords.items():
        if pkey(point) in after_node_at_position:
            continue
        if node_id in after.coords:
            # 번호가 남아 다른 자리로 갔다 — 아래 '이동' 에서 다룬다
            continue
        if node_id in absorbed:
            result.deleted_nodes[node_id] = f"절점 {node_id} — {absorbed[node_id]} 에 합쳐져 사라짐"
        else:
            result.deleted_nodes[node_id] = f"절점 {node_id} — 삭제됨 (이어받은 절점 없음)"

    for node_id, point in after.coords.items():
        if node_id in before.coords:
            start_point = np.asarray(before.coords[node_id], float)
            end_point = np.asarray(point, float)
            move_distance = float(np.linalg.norm(end_point - start_point))
            if move_distance <= XYZ_TOL:
                continue
            node_that_was_there = before_node_at_position.get(pkey(point))
            if node_that_was_there is not None and node_that_was_there != node_id:
                explanation = f"절점 {node_id} — {node_that_was_there} 자리로 옮겨가며 {node_that_was_there} 를 흡수"
            else:
                explanation = f"절점 {node_id} — 이동 ({move_distance:.1f} mm)"
            result.moved_nodes[node_id] = (start_point, end_point, explanation)
        elif pkey(point) not in before_node_at_position:
            result.added_nodes[node_id] = f"절점 {node_id} — 새로 생김"
    return result


# ──────────────────────────────────────────────────────────────────────────
# 채점 — 재생성 결과를 사람 결과와 비교
# ──────────────────────────────────────────────────────────────────────────
def precision_recall(predicted: set, true: set):
    """(정밀도, 재현율). 둘 다 비면 1.0 — '아무것도 안 바꿔야 하는데 안 바꿨다' 는 맞힌 것."""
    if len(predicted) == 0 and len(true) == 0:
        return 1.0, 1.0
    true_positive = len(predicted & true)
    if len(predicted) > 0:
        precision = true_positive / len(predicted)
    else:
        precision = 0.0
    if len(true) > 0:
        recall = true_positive / len(true)
    else:
        recall = 0.0
    return precision, recall


def compare(bad, predicted, good) -> dict:
    """재생성 결과(predicted)를 사람 결과(good)에 대어 본다. 셋 다 같은 원본(bad)에서 출발.

    정확 일치(exact)가 주 지표다. 나머지는 "어디까지 맞았나" 를 보는 부분 점수:
      - 지운 요소(rm): bad 의 요소 중 결과에서 사라진 것 (좌표 집합 기준)
      - 만든 요소(cr): 결과에 새로 생긴 요소
      - 사라진 자리(del): bad 의 절점 자리 중 결과에 없는 것
    """
    bad_signature = signature(bad)
    predicted_signature = signature(predicted)
    good_signature = signature(good)

    removed_by_model = bad_signature.shells - predicted_signature.shells
    removed_by_human = bad_signature.shells - good_signature.shells
    created_by_model = predicted_signature.shells - bad_signature.shells
    created_by_human = good_signature.shells - bad_signature.shells
    deleted_by_model = bad_signature.nodes - predicted_signature.nodes
    deleted_by_human = bad_signature.nodes - good_signature.nodes

    result = {
        "exact": predicted_signature == good_signature,
        "exact_shells": predicted_signature.shells == good_signature.shells,
        "n_rm_pred": len(removed_by_model), "n_rm_true": len(removed_by_human),
        "n_cr_pred": len(created_by_model), "n_cr_true": len(created_by_human),
        "n_del_pred": len(deleted_by_model), "n_del_true": len(deleted_by_human),
    }

    precision, recall = precision_recall(removed_by_model, removed_by_human)
    result["rm_prec"] = precision
    result["rm_rec"] = recall
    precision, recall = precision_recall(created_by_model, created_by_human)
    result["cr_prec"] = precision
    result["cr_rec"] = recall
    precision, recall = precision_recall(deleted_by_model, deleted_by_human)
    result["del_prec"] = precision
    result["del_rec"] = recall

    # 바뀐 요소 전체(지운 것 + 만든 것)의 IoU — 한 숫자로 "얼마나 겹치나"
    changed_by_model = removed_by_model | created_by_model
    changed_by_human = removed_by_human | created_by_human
    union = changed_by_model | changed_by_human
    if len(union) > 0:
        result["change_iou"] = len(changed_by_model & changed_by_human) / len(union)
    else:
        result["change_iou"] = 1.0

    result["did_nothing"] = predicted_signature == bad_signature
    return result

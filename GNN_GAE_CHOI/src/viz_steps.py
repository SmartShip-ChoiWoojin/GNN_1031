"""편집 과정 단계별 보기 — 재생성 연산이 메쉬를 한 단계씩 어떻게 바꾸는지, 실행되는 코드와 함께.

    원본 → 1단계 후보 생성 → [edge collapse 고르기 → 적용] × n → 1단계 종료
         → 2단계 후보 생성 → [triangle-pair merge 고르기 → 적용] × m → 최종 결과

단계마다 세 가지를 보여준다.
  - 그 단계가 끝난 메쉬 (바뀐 element 를 칠하고, 없어진 것은 이전 위치에 점선으로)
  - 그 단계에서 실행되는 **실제 소스 코드** — GNN_GAE_CHOI/src 의 파일을 ast 로 읽어 함수 범위를
    찾고, 해당 줄을 강조한다 (코드를 옮겨 적지 않으므로 소스가 바뀌어도 어긋나지 않는다. 다만
    강조할 줄은 '그 줄에 들어 있는 글자' 로 찾으므로, 그 줄을 고치면 아래 marks 도 같이 고친다)
  - 그 메쉬에서 실제로 일어난 일 — element 마다 node 목록이 어떻게 바뀌어 어떤 결과가 났는지,
    검사 수치(이면각, 면적, 비매니폴드 변 수)

모델은 돌리지 않는다. 채점 때 기록한 연산(collapses / merges 열)과 margin 을 쓴다.
앱(app_regen.py)의 '편집 과정' 탭이 이 모듈을 쓰고, 발표용 단독 HTML 도 만든다:

    python src/viz_steps.py --run runs/regen_final --name SN2592_FWD_Clean_0199
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

import gae_paths
import features as F
import meshedit
from meshdiff import compare, pkey
from meshedit import (EditableMesh, EditError, apply_merges, collapse_pairs, compact_nodes,
                      count_nonmanifold_edges, merge_candidates, shared_edge_of)
from meshrep import Edge
from regen_oracle import count_in_order, parse_collapses, parse_edges
from viz_compare import NEUTRAL_LINE, NEUTRAL_SHELL, SHELL_EDGE, add_segment

FOCUS = "#fbbf24"        # 이번 단계에서 영향을 받는 element
NEW = "#22c55e"          # 이번 단계에서 새로 생기거나 모양이 바뀐 element
CAND = "#bae6fd"         # 후보 element (2단계)
CAND_EDGE = "#0d9488"    # 후보 node 쌍 (1단계)
GONE = "#ef4444"         # 없어진 것 (범례 색)
GONE_LINE = "#b91c1c"    # 없어진 것 (점선 — 초록 위에서도 보이도록 진하게)
REMOVE_NODE = "#dc2626"
KEEP_NODE = "#2563eb"
BAD_EDGE = "#111827"


# ──────────────────────────────────────────────────────────────────────────
# 단계 하나
# ──────────────────────────────────────────────────────────────────────────
class Step:
    """화면의 단계 하나."""

    def __init__(self, title: str, mesh, fill=None, ghost_shells=None, ghost_segments=None, thick_segments=None,
                 candidate_segments=None, line_color=None, labels=None, show_bad: bool = False, focus=None,
                 code=None, notes=None, legend=None):
        # 인자를 안 주면 빈 목록/사전으로 둔다
        if fill is None:
            fill = {}
        if ghost_shells is None:
            ghost_shells = []
        if ghost_segments is None:
            ghost_segments = []
        if thick_segments is None:
            thick_segments = []
        if candidate_segments is None:
            candidate_segments = []
        if line_color is None:
            line_color = {}
        if labels is None:
            labels = []
        if code is None:
            code = []
        if notes is None:
            notes = []
        if legend is None:
            legend = []

        self.title = title
        self.mesh = mesh
        #: {element 인덱스: 색} (칠할 것만)
        self.fill = fill
        #: 없어진 shell 의 꼭짓점 좌표 목록들 (점선으로 그린다)
        self.ghost_shells = ghost_shells
        #: 없어진 선(1D, 공유 변)의 양 끝 좌표 [(p0, p1)]
        self.ghost_segments = ghost_segments
        #: 굵게 그릴 선 [(p0, p1, 색)]
        self.thick_segments = thick_segments
        #: 얇게 그릴 후보 선 [(p0, p1)]
        self.candidate_segments = candidate_segments
        #: {1D 인덱스: 색} (강조할 1D)
        self.line_color = line_color
        #: 글자 표시 [(좌표, 글자, 색)]
        self.labels = labels
        #: 불량각 element 테두리를 그릴지
        self.show_bad = show_bad
        #: '편집 위치만 확대' 를 켰을 때 쓸 영역 (중심 xyz, 반폭). None 이면 확대 없음.
        self.focus = focus
        #: [("파일.py", "함수 또는 Class.method", [강조할 줄에 들어 있는 문자열들])]
        self.code = code
        self.notes = notes
        #: [(색, 설명)]
        self.legend = legend


def node_points(mesh, node_ids) -> list:
    point_list = []
    for node_id in node_ids:
        point_list.append(mesh.coords[node_id])
    return point_list


def total_area_of_shells(mesh) -> float:
    areas = []
    for shell in mesh.shells:
        areas.append(F.area_normal(np.array(node_points(mesh, shell[2]), float))[0])
    return float(sum(areas))


def angle_text(mesh, node_ids) -> str:
    quality = F.shell_quality(np.array(node_points(mesh, node_ids), float))
    return f"내각 {quality.angle_min:.1f}°~{quality.angle_max:.1f}°"


def shell_index_by_id(mesh, element_id: int):
    for shell_index in range(len(mesh.shells)):
        if mesh.shells[shell_index][0] == element_id:
            return shell_index
    return None


def focus_on_edge(mesh, remove_node: int, keep_node: int) -> tuple:
    """edge collapse 단계의 확대 영역 — remove·keep 의 중점을 중심으로.

    변 주변만 보이도록 중앙 변 길이와 remove–keep 거리로 크기를 정한다.
    """
    remove_point = np.asarray(mesh.coords[remove_node], float)
    keep_point = np.asarray(mesh.coords[keep_node], float)
    half_width = max(1.0 * mesh.rep.median_edge_length, 2.5 * float(np.linalg.norm(remove_point - keep_point)))
    center = []
    for value in (remove_point + keep_point) / 2:
        center.append(float(value))
    return (tuple(center), half_width)


def focus_on_nodes(mesh, node_ids) -> tuple:
    """triangle-pair merge 단계의 확대 영역 — 두 삼각형(= 새 사각형)을 담는 정육면체."""
    points = np.array(node_points(mesh, node_ids), float)
    lowest = points.min(0)
    highest = points.max(0)
    half_width = 0.5 * float((highest - lowest).max()) * 1.3 + 0.3 * mesh.rep.median_edge_length
    center = []
    for value in (lowest + highest) / 2:
        center.append(float(value))
    return (tuple(center), half_width)


def created_compared_to(base, mesh) -> set:
    """mesh 의 element 중 base 에 같은 좌표로 없는 것의 인덱스 (좌표 기준)."""
    base_keys = set()
    for shell in base.shells:
        position_keys = []
        for node_id in shell[2]:
            position_keys.append(pkey(base.coords[node_id]))
        base_keys.add(frozenset(position_keys))
    created = set()
    for shell_index in range(len(mesh.shells)):
        position_keys = []
        for node_id in mesh.shells[shell_index][2]:
            position_keys.append(pkey(mesh.coords[node_id]))
        if frozenset(position_keys) not in base_keys:
            created.add(shell_index)
    return created


def collapse_rejection_reason(message: str) -> str:
    """collapse 거부 메시지 → 화면에 보일 짧은 사유."""
    if "접힌다" in message:
        return "접힘(법선 반전)"
    if "오목" in message:
        return "오목한 사각형"
    if "면적이 0" in message:
        return "면적 0"
    if "비매니폴드" in message:
        return "비매니폴드 변 증가"
    if "1D" in message:
        return "1D 가 shell 변을 벗어남"
    if "총 면적" in message:
        return "총 면적 5% 초과"
    return "기타"


def merge_rejection_reason(message: str) -> str:
    """merge 후보가 막힌 메시지 → 화면에 보일 짧은 사유."""
    if "다른 패널" in message:
        return "이면각 > 20°"
    if "오목" in message:
        return "오목한 사각형"
    if "면적" in message:
        return "면적 변화 > 1%"
    return "기타"


def reasons_text(reasons: list) -> str:
    """[사유, ...] → "사유 n개, 사유 n개" (많은 순). 없으면 "없음"."""
    parts = []
    for reason, count in count_in_order(reasons):
        parts.append(f"{reason} {count}개")
    if len(parts) == 0:
        return "없음"
    return ", ".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# 단계 만들기
# ──────────────────────────────────────────────────────────────────────────
def build_steps(bad: EditableMesh, collapses: list, merges: list, margins: list | None,
                stage1_threshold: float, stage2_threshold: float, human=None) -> list:
    """기록된 연산을 원본에 하나씩 다시 적용하며 단계를 만든다."""
    steps = []
    mesh = bad.clone()
    mesh.log.clear()
    triangle_count = 0
    bad_count = 0
    for shell in mesh.shells:
        quality = F.shell_quality(np.array(node_points(mesh, shell[2]), float))
        if not quality.is_quad:
            triangle_count = triangle_count + 1
        if F.is_bad_shell(quality):
            bad_count = bad_count + 1

    # ── 0. 원본 ────────────────────────────────────────────────────────
    steps.append(Step(
        "0. 원본 (비정상 메쉬)", mesh.clone(), show_bad=True,
        code=[("regen_model.py", "RegenModel.regenerate",
               ["def regenerate", "sequence = decode_collapses", "mesh_after_stage1 = mesh.clone()",
                "merged_edges, _ = apply_merges"])],
        notes=[f"node {len(mesh.coords)}개, shell element {len(mesh.shells)}개 (삼각형 {triangle_count}개), "
               f"1D element {len(mesh.lines)}개",
               f"불량각 element (최소 내각 30° 미만 또는 최대 내각 150° 초과): {bad_count}개 — 검은 굵은 테두리",
               f"재생성은 1단계 edge collapse {len(collapses)}회 → 2단계 triangle-pair merge "
               f"{len(merges)}회로 진행됩니다 (기록된 연산)."],
        legend=[(BAD_EDGE, "불량각 element")]))

    # ── 1. 1단계 후보 생성 ──────────────────────────────────────────────
    all_pairs = collapse_pairs(mesh)
    applicable_pairs = []
    rejection_reasons = []
    for remove_node, keep_node, relation in all_pairs:
        try:
            mesh.clone().collapse_edge(remove_node, keep_node)
        except EditError as error:
            rejection_reasons.append(collapse_rejection_reason(str(error)))
            continue
        applicable_pairs.append((remove_node, keep_node))
    candidate_edges = set()
    for remove_node, keep_node in applicable_pairs:
        candidate_edges.add(Edge((remove_node, keep_node)))
    candidate_segments = []
    for edge in candidate_edges:
        candidate_segments.append(tuple(node_points(mesh, tuple(edge))))
    diagonal_count = 0
    for remove_node, keep_node, relation in all_pairs:
        if relation == "diag":
            diagonal_count = diagonal_count + 1
    diagonal_count = diagonal_count // 2

    steps.append(Step(
        "1. 1단계 후보 생성", mesh.clone(), candidate_segments=candidate_segments,
        code=[("regen_feats.py", "collapse_candidates",
               ["for remove_node, keep_node, relation in collapse_pairs(mesh):", "preview = mesh.clone()",
                "operation = preview.collapse_edge(remove_node, keep_node)", "except EditError:", "continue"])],
        notes=[f"node 쌍: 변으로 이어진 쌍 {len(all_pairs) // 2 - diagonal_count}개 + 사각형 대각 쌍 {diagonal_count}개 "
               f"→ 방향(remove/keep)을 나누면 후보 {len(all_pairs)}개 (collapse_pairs)",
               f"각 후보를 사본에 실제로 적용해 보고 거부되면 뺍니다 → **적용 가능한 후보 {len(applicable_pairs)}개**",
               "거부 사유: " + reasons_text(rejection_reasons),
               "청록 선 = 적용 가능한 후보가 있는 node 쌍. 모델(GNN)이 이 후보들과 STOP 에 점수를 매깁니다."],
        legend=[(CAND_EDGE, "적용 가능한 후보 node 쌍")]))

    # ── 2. edge collapse 하나씩 ───────────────────────────────────────
    for collapse_number in range(1, len(collapses) + 1):
        remove_node, keep_node = collapses[collapse_number - 1]
        relation = mesh.collapse_relation(remove_node, keep_node)
        touched_indices = []
        for shell_index in range(len(mesh.shells)):
            if remove_node in mesh.shells[shell_index][2]:
                touched_indices.append(shell_index)
        margin = None
        if margins and collapse_number - 1 < len(margins):
            margin = margins[collapse_number - 1]

        if relation == "edge":
            relation_text = "변으로 이어진 쌍"
        else:
            relation_text = "사각형 대각 쌍"
        if margin is not None:
            margin_note = (f"margin = 후보 logit − STOP logit = **{margin:.2f}** ≥ thr1 = {stage1_threshold:.2f}"
                           f" → 적용")
        else:
            margin_note = f"기록된 {collapse_number}번째 edge collapse (이 결과 파일에는 margin 이 기록돼 있지 않습니다)"
        pick_notes = [f"모델 점수 1위 후보: remove node {remove_node} → keep node {keep_node} ({relation_text})",
                      margin_note,
                      f"node {remove_node} 를 쓰는 shell element {len(touched_indices)}개(노랑)가 영향을 받습니다."]
        if mesh.ext.get(remove_node):
            pick_notes.append(f"node {remove_node} 는 ext node 입니다 — 합치면 keep node {keep_node} 가 "
                              f"ext 플래그를 이어받습니다.")
        touched_fill = {}
        for shell_index in touched_indices:
            touched_fill[shell_index] = FOCUS

        steps.append(Step(
            f"{len(steps)}. edge collapse {collapse_number} — 고르기 ({remove_node} → {keep_node})", mesh.clone(),
            fill=touched_fill,
            thick_segments=[(mesh.coords[remove_node], mesh.coords[keep_node], REMOVE_NODE)],
            labels=[(mesh.coords[remove_node], f"remove {remove_node}", REMOVE_NODE),
                    (mesh.coords[keep_node], f"keep {keep_node}", KEEP_NODE)],
            focus=focus_on_edge(mesh, remove_node, keep_node),
            code=[("regen_model.py", "decode_collapses",
                   ["candidate_scores, stop_score = stage1_scorer.scores",
                    "for candidate_index in np.argsort(-candidate_scores",
                    "margin = float(candidate_scores[candidate_index] - stop_score)", "if margin < min_margin",
                    "picked = ((remove_node, keep_node)", "sequence.append((chosen_pair, margin))"])],
            notes=pick_notes,
            legend=[(FOCUS, f"node {remove_node} 를 쓰는 element"), (REMOVE_NODE, "합칠 변 (remove → keep)")]))

        # 적용 — element 마다 무슨 일이 일어나는지 먼저 계산하고 실제로 적용한다
        shell_fates = []
        for shell_index in touched_indices:
            element_id, element_type, node_ids = mesh.shells[shell_index]
            replaced_nodes = []
            for node_id in node_ids:
                if node_id == remove_node:
                    replaced_nodes.append(keep_node)
                else:
                    replaced_nodes.append(node_id)
            compacted = compact_nodes(replaced_nodes)
            shell_fates.append((element_id, element_type, list(node_ids), replaced_nodes, compacted))
        line_fates = []
        for element_id, element_type, (node_a, node_b) in mesh.lines:
            if remove_node == node_a or remove_node == node_b:
                new_a = node_a
                new_b = node_b
                if node_a == remove_node:
                    new_a = keep_node
                if node_b == remove_node:
                    new_b = keep_node
                line_fates.append((element_id, element_type, (node_a, node_b), (new_a, new_b)))

        nonmanifold_before = count_nonmanifold_edges(mesh.shells)
        area_before = total_area_of_shells(mesh)
        after_mesh = mesh.clone()
        operation = after_mesh.collapse_edge(remove_node, keep_node)
        removed_element_ids = set()
        for removed_shell in operation.info["removed_shells"]:
            removed_element_ids.add(removed_shell[0])
        dropped_line_ids = set()
        for dropped_line in operation.info["dropped_lines"]:
            dropped_line_ids.add(dropped_line[0])
        nonmanifold_after = count_nonmanifold_edges(after_mesh.shells)
        area_after = total_area_of_shells(after_mesh)

        notes = [f"node {remove_node} 를 쓰던 element 의 node 목록에서 {remove_node} 를 {keep_node} 로 바꾸고 "
                 f"`compact_nodes` 로 정리:"]
        new_fill = {}
        ghost_shells = []
        for element_id, element_type, node_ids, replaced_nodes, compacted in shell_fates:
            head = f"{element_type} {element_id}: {node_ids} → {replaced_nodes}"
            if element_id in removed_element_ids:
                if compacted is None:
                    if len(node_ids) == 3:
                        tail = "꼭짓점이 2개만 남아 **삭제**"
                    else:
                        tail = "마주 보는 꼭짓점이 겹쳐 면적 0 → **삭제**"
                else:
                    tail = "모양이 바뀐 뒤 다른 element 와 node 가 같아져 **삭제** (중복 element 방지)"
                ghost_shells.append(node_points(mesh, node_ids))
            elif compacted is not None and len(compacted) < len(node_ids):
                tail = f"붙은 두 꼭짓점이 겹쳐 {compacted} → **사각형이 삼각형으로** (CQUAD4 → CTRIA3)"
            else:
                tail = f"node 만 바뀌어 **모양 변경** ({angle_text(after_mesh, compacted)})"
            index_after = shell_index_by_id(after_mesh, element_id)
            if index_after is not None:
                new_fill[index_after] = NEW
            notes.append(f"· {head} → {tail}")

        highlighted_lines = {}
        for element_id, element_type, old_ends, new_ends in line_fates:
            if element_id in dropped_line_ids:
                notes.append(f"· {element_type} {element_id}: {old_ends} → {new_ends} → 길이 0 또는 중복이라 **삭제**")
            else:
                notes.append(f"· {element_type} {element_id}: 끝점 {old_ends} → {new_ends}")
                for line_index in range(len(after_mesh.lines)):
                    if after_mesh.lines[line_index][0] == element_id:
                        highlighted_lines[line_index] = NEW
                        break

        area_change_percent = 100 * (area_after - area_before) / area_before
        notes.append(f"검사 통과 — 남는 element 의 법선 반전·면적 0·오목 없음, 비매니폴드 변 "
                     f"{nonmanifold_before} → {nonmanifold_after}개, 총 면적 변화 {area_change_percent:+.3f}% "
                     f"(한도 ±{100 * meshedit.REGEN_AREA_TOL:.0f}%)")
        notes.append(f"node {remove_node} 삭제 (node {len(mesh.coords)} → {len(after_mesh.coords)}개, "
                     f"shell element {len(mesh.shells)} → {len(after_mesh.shells)}개). 좌표는 하나도 바뀌지 않습니다.")

        ghost_segments = []
        for element_id, element_type, (node_a, node_b), new_ends in line_fates:
            if element_id in dropped_line_ids:
                ghost_segments.append((mesh.coords[node_a], mesh.coords[node_b]))

        steps.append(Step(
            f"{len(steps)}. edge collapse {collapse_number} — 적용", after_mesh.clone(),
            fill=new_fill, ghost_shells=ghost_shells, ghost_segments=ghost_segments, line_color=highlighted_lines,
            labels=[(mesh.coords[remove_node], f"삭제된 node {remove_node}", REMOVE_NODE),
                    (mesh.coords[keep_node], f"keep {keep_node}", KEEP_NODE)],
            focus=focus_on_edge(mesh, remove_node, keep_node),
            code=[("meshedit.py", "EditableMesh.collapse_many",
                   ["relation = node_relation(work_shells, remove_node, keep_node)",
                    "for work_shell in work_shells:",
                    "work_shell[2] = compact_nodes(replaced_nodes)",
                    "if current_area <= 1e-9", "if float(np.dot(original_normal, current_normal)) <= 0.0",
                    "is_convex_quad(self.points_of(current_nodes))",
                    "if nonmanifold_after > nonmanifold_before", "self.shells = new_shells",
                    "del self.coords[remove_node]"]),
                  ("meshedit.py", "compact_nodes",
                   ["elif result[-1] != node_id", "result.pop()", "return None", "return result"])],
            notes=notes,
            legend=[(NEW, "모양이 바뀌거나 삼각형이 된 element"), (GONE, "삭제된 element (이전 위치, 점선)")]))
        mesh = after_mesh

    # ── 3. 1단계 종료 ─────────────────────────────────────────────────
    changed_so_far = created_compared_to(bad, mesh)
    changed_fill = {}
    for shell_index in changed_so_far:
        changed_fill[shell_index] = NEW
    steps.append(Step(
        f"{len(steps)}. 1단계 종료", mesh.clone(), fill=changed_fill,
        code=[("regen_model.py", "decode_collapses",
               ["if item.candidate_count == 0", "if margin < min_margin", "break", "if picked is None"])],
        notes=[f"edge collapse {len(collapses)}회 적용 후 종료 — 다음 step 에서 최고 후보의 margin 이 "
               f"thr1 = {stage1_threshold:.2f} 보다 작거나 적용할 후보가 없으면 멈춥니다 (stopping criterion).",
               f"원본 대비 바뀐 element {len(changed_so_far)}개 (초록).",
               f"node {len(bad.coords)} → {len(mesh.coords)}개, shell element {len(bad.shells)} → {len(mesh.shells)}개"],
        legend=[(NEW, "원본 대비 바뀐 element")]))

    # ── 4. 2단계 후보 ─────────────────────────────────────────────────
    all_candidates = merge_candidates(mesh, drop_lines=True)
    usable_candidates = []
    blocked_reasons = []
    for candidate in all_candidates:
        if candidate.ok:
            usable_candidates.append(candidate)
        else:
            blocked_reasons.append(merge_rejection_reason(candidate.blocked))
    candidate_fill = {}
    for candidate in usable_candidates:
        candidate_fill[candidate.si] = CAND
        candidate_fill[candidate.sj] = CAND
    steps.append(Step(
        f"{len(steps)}. 2단계 후보 생성", mesh.clone(), fill=candidate_fill,
        code=[("regen_feats.py", "merge_cands",
               ["for candidate in merge_candidates(mesh, drop_lines=True):", "if candidate.ok:"]),
              ("meshedit.py", "merge_candidates",
               ["for edge, shell_indices in rep.edge_shells.items():", "if len(shell_indices) != 2",
                "merged = mesh.merged_quality(shell_index_a, shell_index_b, drop_lines=drop_lines)"])],
        notes=[f"변을 공유하는 삼각형 쌍 {len(all_candidates)}개 중 적용 가능한 후보 **{len(usable_candidates)}개** (하늘색)",
               "거부 사유: " + reasons_text(blocked_reasons),
               "2단계 채점기(gradient boosting)가 후보마다 확률을 냅니다."],
        legend=[(CAND, "적용 가능한 후보 삼각형")]))

    # ── 5. triangle-pair merge 하나씩 ─────────────────────────────────
    for merge_number in range(1, len(merges) + 1):
        edge = merges[merge_number - 1]
        found = None
        for candidate in merge_candidates(mesh, drop_lines=True):
            if shared_edge_of(mesh, candidate.si, candidate.sj) == edge:
                found = candidate
                break
        if found is None or not found.ok:
            steps.append(Step(f"{len(steps)}. triangle-pair merge {merge_number} — 적용 불가", mesh.clone(),
                              notes=[f"공유 변 {sorted(edge)}: 후보에 없음 (기록과 다름)"]))
            continue

        node_a, node_b = sorted(edge)
        both_triangle_nodes = set(mesh.shells[found.si][2]) | set(mesh.shells[found.sj][2])
        steps.append(Step(
            f"{len(steps)}. triangle-pair merge {merge_number} — 고르기 (공유 변 {node_a}-{node_b})", mesh.clone(),
            fill={found.si: FOCUS, found.sj: FOCUS},
            thick_segments=[(mesh.coords[node_a], mesh.coords[node_b], REMOVE_NODE)],
            focus=focus_on_nodes(mesh, both_triangle_nodes),
            code=[("regen_model.py", "pick_merges",
                   ["for candidate_index in np.argsort(-probabilities",
                    "if probabilities[candidate_index] < threshold",
                    "if shell_index_a in used_shells or shell_index_b in used_shells",
                    "picked_edges.append(candidates.edges[candidate_index])"])],
            notes=[f"선택: {mesh.shells[found.si][1]} {mesh.shells[found.si][0]} + "
                   f"{mesh.shells[found.sj][1]} {mesh.shells[found.sj][0]} (공유 변 {node_a}-{node_b})",
                   f"확률 ≥ thr2 = {stage2_threshold:.2f} 인 후보를 확률 순으로 고르고, "
                   f"이미 고른 삼각형을 쓰는 후보는 건너뜁니다."],
            legend=[(FOCUS, "합칠 삼각형 2개"), (REMOVE_NODE, "없어질 공유 변")]))

        quality_a = mesh.quality(found.si)
        quality_b = mesh.quality(found.sj)
        quad_nodes = mesh._merged_quad_nodes(found.si, found.sj, drop_lines=True)
        quad_points = np.array(node_points(mesh, quad_nodes), float)
        fan_dot = float(np.dot(np.cross(quad_points[1] - quad_points[0], quad_points[2] - quad_points[0]),
                               np.cross(quad_points[2] - quad_points[0], quad_points[3] - quad_points[0])))
        triangles_area = quality_a.area + quality_b.area
        quad_area = F.area_normal(quad_points)[0]
        lines_on_edge = []
        for line in mesh.lines:
            if Edge(tuple(line[2])) == edge:
                lines_on_edge.append((line[0], line[1], tuple(line[2])))

        after_mesh = mesh.clone()
        apply_merges(after_mesh, [edge])
        new_quad_index = None
        for shell_index in range(len(after_mesh.shells)):
            if set(after_mesh.shells[shell_index][2]) == set(quad_nodes):
                new_quad_index = shell_index
                break

        if len(lines_on_edge) > 0:
            line_note = f"공유 변 위 1D element {len(lines_on_edge)}개 함께 삭제"
        else:
            line_note = "공유 변 위 1D element 없음"
        if new_quad_index is not None:
            new_quality = after_mesh.quality(new_quad_index)
            result_note = (f"결과: CQUAD4 {after_mesh.shells[new_quad_index][0]} "
                           f"({angle_text(after_mesh, after_mesh.shells[new_quad_index][2])}, "
                           f"skew {new_quality.skew:.1f}°) — 원래 두 삼각형 중 최소 내각 "
                           f"{min(quality_a.angle_min, quality_b.angle_min):.1f}°")
            new_fill = {new_quad_index: NEW}
        else:
            result_note = "결과 사각형을 찾지 못함"
            new_fill = {}

        ghost_segments = [(mesh.coords[node_a], mesh.coords[node_b])]
        for element_id, element_type, (line_a, line_b) in lines_on_edge:
            ghost_segments.append((mesh.coords[line_a], mesh.coords[line_b]))

        area_change_percent = 100 * (quad_area - triangles_area) / triangles_area
        steps.append(Step(
            f"{len(steps)}. triangle-pair merge {merge_number} — 적용", after_mesh.clone(),
            fill=new_fill, focus=focus_on_nodes(mesh, set(quad_nodes)), ghost_segments=ghost_segments,
            code=[("meshedit.py", "EditableMesh._merged_quad_nodes",
                   ["dihedral = F.dihedral_deg", "if dihedral > max_dihedral",
                    "quad_nodes = [apex_a, shared_first, apex_b, shared_second]",
                    "if float(np.dot(fan_normal_1, fan_normal_2)) <= 0",
                    "quad_area = F.area_normal(quad_points)[0]", "if relative_change > MERGE_AREA_TOL"]),
                  ("meshedit.py", "EditableMesh.merge_tris",
                   ["dropped_lines = self._drop_lines_on(shared_edge)",
                    'self.shells[keep_index] = [self.shells[keep_index][0], "CQUAD4", quad_nodes]',
                    "del self.shells[delete_index]"])],
            notes=[f"이면각 {found.dihedral:.2f}° ≤ {meshedit.MAX_DIHEDRAL_DEG:.0f}° → 같은 면",
                   f"사각형 node 순서 {quad_nodes}, 볼록 검사 n1·n2 = {fan_dot:.3g} > 0 → 오목 아님",
                   f"면적 {quality_a.area:.1f} + {quality_b.area:.1f} = {triangles_area:.1f} → 사각형 {quad_area:.1f} "
                   f"(변화 {area_change_percent:+.3f}%, 한도 ±{100 * meshedit.MERGE_AREA_TOL:.0f}%)",
                   line_note, result_note],
            legend=[(NEW, "새 사각형 (CQUAD4)"), (GONE, "없어진 공유 변 (점선)")]))
        mesh = after_mesh

    # ── 6. 최종 ───────────────────────────────────────────────────────
    changed_total = created_compared_to(bad, mesh)
    final_triangle_count = 0
    for shell in mesh.shells:
        if len(shell[2]) == 3:
            final_triangle_count = final_triangle_count + 1
    notes = [f"원본 대비 바뀐 element {len(changed_total)}개 (초록)",
             f"node {len(bad.coords)} → {len(mesh.coords)}, shell element {len(bad.shells)} → {len(mesh.shells)}, "
             f"삼각형 {triangle_count} → {final_triangle_count}"]
    if human is not None:
        result = compare(bad, mesh, human)
        if result["exact"]:
            notes.append("사람 수정 결과와 **좌표 기준 정확히 일치** (shell·1D·node 좌표 모두 같음)")
        else:
            notes.append(f"사람 수정 결과와 다름 — 바뀐 요소 IoU {result['change_iou']:.2f}")
    final_fill = {}
    for shell_index in changed_total:
        final_fill[shell_index] = NEW
    steps.append(Step(
        f"{len(steps)}. 최종 결과", mesh.clone(), fill=final_fill,
        code=[("regen_model.py", "RegenModel.regenerate",
               ["mesh_after_stage1 = mesh.clone()", "stage2_candidates, stage2_probabilities = self.stage2",
                "merged_edges, _ = apply_merges", "return RegenResult"])],
        notes=notes, legend=[(NEW, "원본 대비 바뀐 element")]))
    return steps


# ──────────────────────────────────────────────────────────────────────────
# 그리기
# ──────────────────────────────────────────────────────────────────────────
def segments_trace(segments: list, color: str, width: float, dash=None):
    x_values, y_values, z_values = [], [], []
    for point_a, point_b in segments:
        add_segment(x_values, y_values, z_values, point_a, point_b)
    line_style = dict(color=color, width=width)
    if dash:
        line_style["dash"] = dash
    return go.Scatter3d(x=x_values, y=y_values, z=z_values, mode="lines", line=line_style, hoverinfo="skip", name="")


def closed_loop_segments(points: list) -> list:
    """꼭짓점 목록을 닫힌 테두리 선분 목록으로."""
    segments = []
    for corner_index in range(len(points)):
        segments.append((points[corner_index], points[(corner_index + 1) % len(points)]))
    return segments


def step_traces(step: Step) -> list:
    mesh = step.mesh
    x_values, y_values, z_values = [], [], []
    triangle_i, triangle_j, triangle_k = [], [], []
    face_colors = []
    hover_texts = []
    vertex_count = 0
    for shell_index in range(len(mesh.shells)):
        element_id, element_type, node_ids = mesh.shells[shell_index]
        corner_points = node_points(mesh, node_ids)
        quality = F.shell_quality(np.array(corner_points, float))
        node_text_list = []
        for node_id in node_ids:
            node_text_list.append(str(node_id))
        hover = (f"<b>{element_type} {element_id}</b><br>node {', '.join(node_text_list)}<br>"
                 f"내각 {quality.angle_min:.1f}°~{quality.angle_max:.1f}°  skew {quality.skew:.1f}°")
        for point in corner_points:
            x_values.append(point[0])
            y_values.append(point[1])
            z_values.append(point[2])
            hover_texts.append(hover)
        color = step.fill.get(shell_index, NEUTRAL_SHELL)
        triangle_i.append(vertex_count)
        triangle_j.append(vertex_count + 1)
        triangle_k.append(vertex_count + 2)
        face_colors.append(color)
        if len(corner_points) == 4:
            triangle_i.append(vertex_count)
            triangle_j.append(vertex_count + 2)
            triangle_k.append(vertex_count + 3)
            face_colors.append(color)
        vertex_count = vertex_count + len(corner_points)
    traces = [go.Mesh3d(x=x_values, y=y_values, z=z_values, i=triangle_i, j=triangle_j, k=triangle_k,
                        facecolor=face_colors, flatshading=True, text=hover_texts, hoverinfo="text")]

    shell_edges = set()
    for element_id, element_type, node_ids in mesh.shells:
        for corner_index in range(len(node_ids)):
            shell_edges.add(Edge((node_ids[corner_index], node_ids[(corner_index + 1) % len(node_ids)])))
    edge_segments = []
    for edge in shell_edges:
        edge_segments.append(tuple(node_points(mesh, tuple(edge))))
    traces.append(segments_trace(edge_segments, SHELL_EDGE, 2.5))

    if step.show_bad:
        bad_segments = []
        for element_id, element_type, node_ids in mesh.shells:
            corner_points = node_points(mesh, node_ids)
            if F.is_bad_shell(F.shell_quality(np.array(corner_points, float))):
                bad_segments.extend(closed_loop_segments(corner_points))
        traces.append(segments_trace(bad_segments, BAD_EDGE, 7))

    plain_line_segments = []
    for line_index in range(len(mesh.lines)):
        if line_index in step.line_color:
            continue
        node_a, node_b = mesh.lines[line_index][2]
        plain_line_segments.append((mesh.coords[node_a], mesh.coords[node_b]))
    traces.append(segments_trace(plain_line_segments, NEUTRAL_LINE, 4))
    if len(step.line_color) > 0:
        highlighted_segments = []
        for line_index in step.line_color:
            node_a, node_b = mesh.lines[line_index][2]
            highlighted_segments.append((mesh.coords[node_a], mesh.coords[node_b]))
        traces.append(segments_trace(highlighted_segments, NEW, 10))
    if len(step.candidate_segments) > 0:
        traces.append(segments_trace(step.candidate_segments, CAND_EDGE, 5))
    for ghost_points in step.ghost_shells:
        traces.append(segments_trace(closed_loop_segments(ghost_points), GONE_LINE, 7, dash="dash"))
    if len(step.ghost_segments) > 0:
        traces.append(segments_trace(step.ghost_segments, GONE_LINE, 9, dash="dash"))
    for point_a, point_b, color in step.thick_segments:
        traces.append(segments_trace([(point_a, point_b)], color, 11))

    # remove / keep 이 붙어 있는 짧은 변에서 글자가 겹치지 않도록 첫째는 왼쪽 위, 둘째는 오른쪽 아래
    for label_number in range(len(step.labels)):
        point, text, color = step.labels[label_number]
        if label_number % 2 == 0:
            text_position = "top left"
        else:
            text_position = "bottom right"
        traces.append(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode="markers+text", text=[text],
                                   textposition=text_position, textfont=dict(size=13, color=color),
                                   marker=dict(size=6, color=color), hoverinfo="text"))
    return traces


def read_source(file_name: str, qualified_name: str):
    """GNN_GAE_CHOI/src/<file_name> 에서 함수(또는 Class.method)의 소스 줄 목록과 시작 줄 번호.

    import 하지 않고 글자로 읽어 ast 로 함수 범위만 찾는다.
    """
    path = os.path.join(gae_paths.SRC_DIR, file_name)
    with open(path, encoding="utf-8") as file_handle:
        source_text = file_handle.read()
    source_lines = source_text.splitlines()
    body = ast.parse(source_text).body
    found_node = None
    for name_part in qualified_name.split("."):
        found_node = None
        for node in body:
            is_definition = isinstance(node, (ast.FunctionDef, ast.ClassDef))
            if is_definition and node.name == name_part:
                found_node = node
                break
        if found_node is None:
            raise KeyError(f"{file_name} 에 {qualified_name} 이 없다")
        body = found_node.body
    return source_lines[found_node.lineno - 1: found_node.end_lineno], found_node.lineno


def code_html(file_name: str, qualified_name: str, marks: list) -> str:
    """함수 소스를 파일 줄 번호와 함께, ``marks`` 중 하나가 들어 있는 줄을 강조해서."""
    source_lines, start_line = read_source(file_name, qualified_name)
    rows = []
    for line_offset in range(len(source_lines)):
        line = source_lines[line_offset]
        is_highlighted = False
        for mark in marks:
            if mark in line:
                is_highlighted = True
        if is_highlighted:
            css_class = "ln hl"
        else:
            css_class = "ln"
        rows.append(f'<div class="{css_class}"><span class="no">{start_line + line_offset}</span>'
                    f'{html.escape(line.rstrip())}</div>')
    return (f'<div class="codehead">GNN_GAE_CHOI/src/{html.escape(file_name)} · {html.escape(qualified_name)}</div>'
            f'<pre class="code">{"".join(rows)}</pre>')


def bold_markdown_to_html(text: str) -> str:
    """**굵게** 만 HTML 로."""
    escaped = html.escape(text)
    parts = escaped.split("**")
    result = ""
    for part_index in range(len(parts)):
        if part_index % 2 == 1:
            result = result + f"<b>{parts[part_index]}</b>"
        else:
            result = result + parts[part_index]
    return result


# 아래 문자열은 웹 페이지(HTML/CSS/JavaScript) 틀이다. {이름} 자리에 값을 넣는다.
PAGE_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<script>{plotlyjs}</script>
<style>
  :root {{ color-scheme: light; }}
  html, body {{ height:100%; margin:0; }}
  body {{ font-family:"Malgun Gothic","Segoe UI",system-ui,sans-serif; background:#f8fafc;
         color:#0f172a; font-size:13px; display:flex; flex-direction:column; }}
  .nav {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; padding:8px 12px;
          background:#fff; border-bottom:1px solid #e2e8f0; }}
  .nav button {{ font:inherit; padding:3px 9px; border:1px solid #cbd5e1; background:#fff;
                 border-radius:5px; cursor:pointer; }}
  .nav button.on {{ background:#0f172a; color:#fff; border-color:#0f172a; }}
  .nav .sep {{ width:1px; height:20px; background:#e2e8f0; margin:0 4px; }}
  .main {{ flex:1; min-height:0; display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,1fr); }}
  #plot {{ min-height:0; background:#fff; }}
  .side {{ min-height:0; overflow:auto; padding:12px 16px; border-left:1px solid #e2e8f0; }}
  h2 {{ font-size:16px; margin:0 0 8px; }}
  ul {{ margin:6px 0 10px; padding-left:18px; }}
  li {{ margin:3px 0; line-height:1.45; }}
  .legend span {{ display:inline-flex; align-items:center; gap:5px; margin-right:12px; }}
  .key {{ width:12px; height:12px; border-radius:2px; display:inline-block; }}
  .codehead {{ margin-top:12px; font-weight:600; color:#334155; }}
  pre.code {{ margin:4px 0 0; background:#0f172a; color:#e2e8f0; border-radius:6px; padding:8px 0;
              font:12px/1.5 Consolas,"D2Coding",monospace; overflow:auto; max-height:46vh; }}
  .ln {{ white-space:pre; padding:0 10px 0 0; }}
  .ln.hl {{ background:#854d0e; color:#fff; }}
  .no {{ display:inline-block; width:42px; text-align:right; padding-right:10px; color:#64748b; }}
  .ln.hl .no {{ color:#fde68a; }}
</style>
<div class="nav">
  <button id="prev">◀ 이전</button><button id="next">다음 ▶</button>
  <label title="축 범위를 편집 위치 주변으로 좁힙니다. 범위 밖 부분은 잘려 보입니다.">
    <input type="checkbox" id="zoom"> 편집 위치만 확대 (범위 밖은 잘림)</label><span class="sep"></span>
  {buttons}
</div>
<div class="main">
  <div id="plot"></div>
  <div class="side" id="side"></div>
</div>
<script>
const STEPS = {steps};
const RANGES = {ranges};
const layout = {{margin:{{l:0,r:0,t:0,b:0}}, showlegend:false, hoverlabel:{{align:'left'}},
  scene:{{aspectmode:'data', xaxis:{{range:RANGES[0], showticklabels:false, title:''}},
          yaxis:{{range:RANGES[1], showticklabels:false, title:''}},
          zaxis:{{range:RANGES[2], showticklabels:false, title:''}}}}}};
let cur = 0;
function show(i) {{
  cur = Math.max(0, Math.min(STEPS.length - 1, i));
  const s = STEPS[cur];
  // Plotly.react 는 3D 장면에서 면 색·새 트레이스를 다시 그리지 못하는 경우가 있어
  // 단계마다 newPlot 으로 새로 그린다. 보던 각도는 카메라를 저장해 두었다가 그대로 넣어 유지한다.
  const gd = document.getElementById('plot');
  const cam = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.camera;
  const lay = JSON.parse(JSON.stringify(layout));
  lay.scene.camera = cam ? JSON.parse(JSON.stringify(cam))
                         : {{eye:{{x:-1.3, y:-1.5, z:0.6}}, center:{{x:0, y:0, z:0}}, up:{{x:0, y:0, z:1}}}};
  if (s.focus && document.getElementById('zoom').checked) {{
    const [c, h] = s.focus;
    lay.scene.xaxis.range = [c[0] - h, c[0] + h];
    lay.scene.yaxis.range = [c[1] - h, c[1] + h];
    lay.scene.zaxis.range = [c[2] - h, c[2] + h];
  }}
  Plotly.newPlot(gd, s.data, lay, {{responsive:true, displaylogo:false}});
  document.getElementById('side').innerHTML = s.side;
  document.querySelectorAll('.nav button[data-i]').forEach(b => b.classList.toggle('on', +b.dataset.i === cur));
  document.querySelectorAll('pre.code').forEach(pre => {{
    const hl = pre.querySelector('.ln.hl');
    if (hl) pre.scrollTop = Math.max(0, hl.offsetTop - pre.clientHeight / 3);
  }});
}}
document.querySelectorAll('.nav button[data-i]').forEach(b => b.onclick = () => show(+b.dataset.i));
document.getElementById('prev').onclick = () => show(cur - 1);
document.getElementById('next').onclick = () => show(cur + 1);
document.getElementById('zoom').onchange = () => show(cur);
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') show(cur + 1);
  if (e.key === 'ArrowLeft') show(cur - 1);
}});
show(0);
window.__steps = {{ready:true, n:STEPS.length, show, cur:() => cur}};
</script>
"""


def render_steps_html(steps: list, name: str) -> str:
    """단계 목록 → 단독 HTML."""
    all_points = []
    for step in steps:
        for point in step.mesh.coords.values():
            all_points.append(point)
    points = np.array(all_points, float)
    lowest = points.min(0)
    highest = points.max(0)
    padding = 0.05 * np.maximum(highest - lowest, 1.0)
    ranges = []
    for axis_index in range(3):
        ranges.append([float(lowest[axis_index] - padding[axis_index]),
                       float(highest[axis_index] + padding[axis_index])])

    step_data = []
    for step in steps:
        figure = go.Figure(data=step_traces(step))
        legend_html = ""
        for color, text in step.legend:
            legend_html = legend_html + (f'<span><span class="key" style="background:{color}"></span>'
                                         f'{html.escape(text)}</span>')
        notes_html = ""
        for note in step.notes:
            notes_html = notes_html + f"<li>{bold_markdown_to_html(note)}</li>"
        code_part = ""
        for file_name, qualified_name, marks in step.code:
            code_part = code_part + code_html(file_name, qualified_name, marks)
        side_html = (f"<h2>{html.escape(step.title)}</h2>"
                     f'<div class="legend">{legend_html}</div>'
                     f"<ul>{notes_html}</ul>" + code_part)
        if step.focus:
            focus_value = [list(step.focus[0]), step.focus[1]]
        else:
            focus_value = None
        step_data.append({"data": figure.to_plotly_json()["data"], "side": side_html, "focus": focus_value})

    buttons_html = ""
    for step_index in range(len(steps)):
        buttons_html = buttons_html + (f'<button data-i="{step_index}" title="{html.escape(steps[step_index].title)}">'
                                       f'{step_index}</button>')
    return PAGE_TEMPLATE.format(plotlyjs=get_plotlyjs(), steps=json.dumps(step_data, cls=PlotlyJSONEncoder),
                                ranges=json.dumps(ranges), buttons=buttons_html)


# ──────────────────────────────────────────────────────────────────────────
def parse_record(row):
    """결과 표의 한 줄 → (collapses, merges, margins). margins 가 없으면 None."""
    collapses = parse_collapses(row["collapses"])
    merges = parse_edges(row["merges"])
    margins = None
    if "margins" in row and str(row["margins"]).strip():
        margins = []
        for margin_text in str(row["margins"]).split(";"):
            if margin_text:
                margins.append(float(margin_text))
    return collapses, merges, margins


def main() -> None:
    import pandas as pd

    from bdf_io import discover_pairs
    from pairs import discover_good_pairs, load_pair

    parser = argparse.ArgumentParser(description="편집 과정 단계별 보기 (단독 HTML)")
    parser.add_argument("--run", default="runs/regen_final")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if os.path.isabs(args.run):
        run_dir = args.run
    else:
        run_dir = os.path.join(gae_paths.ROOT, args.run)
    with open(os.path.join(run_dir, "model.json"), encoding="utf-8") as file_handle:
        meta = json.load(file_handle)

    row = None
    for file_name in ("eval_test.csv", "eval_cv.csv", "cv_pred.csv"):
        path = os.path.join(run_dir, file_name)
        if not os.path.exists(path):
            continue
        table = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        if (table.name == args.name).any():
            row = table[table.name == args.name].iloc[0]
            break
    if row is None:
        raise SystemExit(f"{args.name} 의 기록이 {run_dir} 에 없습니다.")

    collapses, merges, margins = parse_record(row)
    pair = load_pair(args.name, discover_good_pairs(), discover_pairs(gae_paths.FE_DIR)[0])
    steps = build_steps(EditableMesh(pair.before), collapses, merges, margins, meta["thr1"], meta["thr2"],
                        EditableMesh(pair.after))
    gae_paths.ensure_dirs()
    out_path = os.path.join(gae_paths.OUT_DIR, f"edit_steps_{args.name}.html")
    with open(out_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(render_steps_html(steps, args.name))
    print(f"단계 {len(steps)}개 → {out_path}")


if __name__ == "__main__":
    main()

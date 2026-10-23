"""재생성 결과 뷰어 — 왼쪽 원본, 오른쪽 위 모델 재생성 결과 / 아래 사람 수정 결과. 세 화면 시점 연동.

    ┌──────────────────────┬────────────────────────┐
    │                      │ 모델 재생성 결과         │
    │  원본 (비정상 메쉬)    │ (지우기 + 만들기 반영)   │
    │                      ├────────────────────────┤
    │  지워진 element 표시   │ 사람 수정 결과           │
    │                      │ (지우기 + 만들기 반영)   │
    └──────────────────────┴────────────────────────┘

오른쪽 두 화면은 **최종 메쉬**다 — 지운 element 는 이미 없고, 새로 만든 element 만 초록으로
칠한다. 지워진 것은 결과에 없으므로 왼쪽 원본 위에 표시한다.

색 규칙:
    빨강   모델과 사람이 둘 다 지운 element       (왼쪽, '비교' 보기)
    주황   **모델만** 삭제·변형한 element (왼쪽)
    보라   **사람만** 삭제·변형한 element (왼쪽)
    초록   새로 만든 element (오른쪽)

왼쪽 원본은 "원본에 표시할 삭제" 로 보기를 바꾼다:
    비교   누가 지웠는지 색으로 (빨강/주황/보라)
    모델   모델이 지운 것만 빨강
    사람   사람이 지운 것만 빨강

강조를 끄면 그 element 는 회색으로 돌아간다. 같은 자리에 면을 두 장 겹치면 깜빡이므로
(z-fighting), 강조할 element 는 기본 면에서 빼고 "색칠한 것 / 회색 사본" 을 따로 두어 하나만
보이게 한다. 트레이스마다 meta 에 묶음 이름을 적어 두고, 화면의 스크립트가 그 이름으로
보이기/숨기기를 정한다.
"""

from __future__ import annotations

import html
import json

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

import features as F
from meshdiff import diff, pkey
from viz_compare import BAD_OUTLINE, EXT_MARKER, NEUTRAL_LINE, NEUTRAL_SHELL, SHELL_EDGE, add_segment, stats

CREATED = "#22c55e"          # 새로 만든 element
CREATED_EDGE = "#15803d"
FOCUS = "#2563eb"            # '변경 위치로' 표시

#: 누가 바꿨나 — 면 색 / 선 색 / 글
WHO_FILL = {"both": "#ef4444", "model": "#fb923c", "human": "#a78bfa"}
WHO_LINE = {"both": "#b91c1c", "model": "#ea580c", "human": "#7c3aed"}
WHO_TEXT = {"both": "모델·사람 둘 다", "model": "모델만", "human": "사람만"}
#: '모델' / '사람' 보기에서 그쪽이 지운 element 의 색
SIDE_FILL = "#ef4444"
SIDE_LINE = "#b91c1c"

#: 원본에 표시할 삭제 보기 (값, 화면 글)
MODES = [("cmp", "비교"), ("model", "모델"), ("human", "사람")]

#: 도구막대 체크박스 (묶음 이름, 표시, 색, 처음 상태)
TOGGLES = [
    ("fill_rm", "지운 element", WHO_FILL["both"], True),
    ("fill_cr", "만든 element", CREATED, True),
    ("line1d", "1D 변경", WHO_LINE["both"], True),
    ("edges", "쉘 경계", SHELL_EDGE, True),
    ("bad", "불량각 element", BAD_OUTLINE, False),
    ("lines", "1D element", NEUTRAL_LINE, True),
    ("ext", "ext node", EXT_MARKER, False),
]


# ──────────────────────────────────────────────────────────────────────────
# 화면 명세
# ──────────────────────────────────────────────────────────────────────────
class OrigSpec:
    """왼쪽 원본 화면. 지워진 element / 1D 를 '누가 지웠나' 와 함께 갖는다."""

    def __init__(self, mesh, title: str = "원본 (비정상 메쉬)"):
        self.mesh = mesh
        #: {쉘 인덱스: "both" | "model" | "human"}
        self.removed_shell_who = {}
        #: {1D 인덱스: 누가}
        self.removed_line_who = {}
        self.title = title


class ResultSpec:
    """오른쪽 결과 화면 (모델 또는 사람). 지우기·만들기가 반영된 최종 메쉬."""

    def __init__(self, mesh, who: str, title: str, created=None, created_lines=None):
        self.mesh = mesh
        #: "model" | "human"
        self.who = who
        self.title = title
        #: {새로 만든 쉘 인덱스: 설명}
        if created is None:
            created = {}
        self.created = created
        #: {새로 생긴 1D 인덱스: 설명}
        if created_lines is None:
            created_lines = {}
        self.created_lines = created_lines


# ──────────────────────────────────────────────────────────────────────────
# 트레이스 부품
# ──────────────────────────────────────────────────────────────────────────
def shell_hover(mesh, qualities: list, shell_index: int, extra: str = "") -> str:
    element_id, element_type, node_ids = mesh.shells[shell_index]
    quality = qualities[shell_index]
    bad_text = ""
    if F.is_bad_shell(quality):
        bad_text = " <b>불량</b>"
    node_text_list = []
    for node_id in node_ids:
        node_text_list.append(str(node_id))
    text = (f"<b>{element_type} {element_id}</b><br>내각 {quality.angle_min:.1f}°~{quality.angle_max:.1f}°"
            f"{bad_text}<br>"
            f"aspect {quality.aspect:.2f}  skew {quality.skew:.1f}°<br>"
            f"node {', '.join(node_text_list)}")
    if extra:
        text = text + f"<br><b>{extra}</b>"
    return text


def faces_trace(mesh, shell_indices: list, colors, hover: dict, meta: str, visible=True):
    """쉘 묶음 하나를 Mesh3d 로. ``colors`` 는 {쉘 인덱스: 색} 또는 색 문자열 하나."""
    x_values, y_values, z_values = [], [], []
    triangle_i, triangle_j, triangle_k = [], [], []
    face_colors = []
    hover_texts = []
    vertex_count = 0
    for shell_index in shell_indices:
        node_ids = mesh.shells[shell_index][2]
        if isinstance(colors, str):
            color = colors
        else:
            color = colors[shell_index]
        for node_id in node_ids:
            point = mesh.coords[node_id]
            x_values.append(point[0])
            y_values.append(point[1])
            z_values.append(point[2])
            hover_texts.append(hover[shell_index])
        triangle_i.append(vertex_count)
        triangle_j.append(vertex_count + 1)
        triangle_k.append(vertex_count + 2)
        face_colors.append(color)
        if len(node_ids) == 4:
            triangle_i.append(vertex_count)
            triangle_j.append(vertex_count + 2)
            triangle_k.append(vertex_count + 3)
            face_colors.append(color)
        vertex_count = vertex_count + len(node_ids)
    return go.Mesh3d(x=x_values, y=y_values, z=z_values, i=triangle_i, j=triangle_j, k=triangle_k,
                     facecolor=face_colors, flatshading=True, opacity=1.0, text=hover_texts, hoverinfo="text",
                     meta=meta, visible=visible)


def outline_trace(mesh, shell_indices: list, color: str, width: float, meta: str, visible=True):
    """쉘 테두리."""
    x_values, y_values, z_values = [], [], []
    for shell_index in shell_indices:
        corner_points = []
        for node_id in mesh.shells[shell_index][2]:
            corner_points.append(mesh.coords[node_id])
        for point in corner_points + [corner_points[0]]:
            x_values.append(point[0])
            y_values.append(point[1])
            z_values.append(point[2])
        x_values.append(None)
        y_values.append(None)
        z_values.append(None)
    return go.Scatter3d(x=x_values, y=y_values, z=z_values, mode="lines", line=dict(color=color, width=width),
                        hoverinfo="skip", meta=meta, visible=visible)


def lines_trace(mesh, items: list, color: str, width: float, meta: str, visible=True):
    """1D 요소 선. ``items`` = [(1D 인덱스, 설명)]."""
    x_values, y_values, z_values, texts = [], [], [], []
    for line_index, explanation in items:
        element_id, element_type, (node_a, node_b) = mesh.lines[line_index]
        add_segment(x_values, y_values, z_values, mesh.coords[node_a], mesh.coords[node_b])
        text = f"<b>{element_type} {element_id}</b><br>node {node_a}-{node_b}"
        if explanation:
            text = text + f"<br><b>{explanation}</b>"
        texts.extend([text, text, ""])
    return go.Scatter3d(x=x_values, y=y_values, z=z_values, mode="lines", line=dict(color=color, width=width),
                        text=texts, hoverinfo="text", meta=meta, visible=visible)


def points_trace(point_list: list, texts, marker: dict, meta: str, visible=True):
    points = np.asarray(point_list, dtype=float).reshape(-1, 3)
    if texts:
        hover_info = "text"
    else:
        hover_info = "skip"
    return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", marker=marker,
                        text=texts, hoverinfo=hover_info, meta=meta, visible=visible)


def common_traces(mesh, qualities: list) -> list:
    """세 화면 공통 — 쉘 경계, 불량각 테두리, 1D element, ext node, '변경 위치로' 표시."""
    rep = mesh.rep
    edge_x, edge_y, edge_z = [], [], []
    for edge in rep.edge_shells:
        node_a, node_b = tuple(edge)
        add_segment(edge_x, edge_y, edge_z, mesh.coords[node_a], mesh.coords[node_b])

    bad_indices = []
    for shell_index in range(len(qualities)):
        if F.is_bad_shell(qualities[shell_index]):
            bad_indices.append(shell_index)
    all_line_items = []
    for line_index in range(len(mesh.lines)):
        all_line_items.append((line_index, ""))
    ext_points = []
    for node_id in sorted(mesh.coords):
        if mesh.ext.get(node_id, False):
            ext_points.append(mesh.coords[node_id])

    return [
        go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines", line=dict(color=SHELL_EDGE, width=2.5),
                     hoverinfo="skip", meta="edges"),
        outline_trace(mesh, bad_indices, BAD_OUTLINE, 5, "bad", visible=False),
        lines_trace(mesh, all_line_items, NEUTRAL_LINE, 4, "lines"),
        points_trace(ext_points, None, dict(size=4, color=EXT_MARKER, symbol="circle-open", line=dict(width=2)),
                     "ext", visible=False),
        go.Scatter3d(x=[], y=[], z=[], mode="markers", marker=dict(size=34, color=FOCUS, opacity=0.25, symbol="circle"),
                     hoverinfo="skip", meta="focus"),
    ]


def mesh_qualities(mesh) -> list:
    qualities = []
    for shell_index in range(len(mesh.shells)):
        qualities.append(F.shell_quality(mesh.corners(shell_index)))
    return qualities


def removed_by(who: str, side: str) -> bool:
    """'모델'/'사람' 보기에서 이 element 를 그쪽이 지웠나 (둘 다 지운 것도 포함)."""
    return who == "both" or who == side


# ──────────────────────────────────────────────────────────────────────────
# 화면별 트레이스
# ──────────────────────────────────────────────────────────────────────────
def orig_traces(spec: OrigSpec) -> list:
    """왼쪽 원본. 지워진 element 는 보기(비교/모델/사람)마다 따로 칠한 트레이스를 둔다."""
    mesh = spec.mesh
    qualities = mesh_qualities(mesh)
    removed_indices = sorted(spec.removed_shell_who)

    hover = {}
    for shell_index in range(len(mesh.shells)):
        extra = ""
        if shell_index in spec.removed_shell_who:
            extra = f"{WHO_TEXT[spec.removed_shell_who[shell_index]]} 지움"
        hover[shell_index] = shell_hover(mesh, qualities, shell_index, extra)

    untouched_indices = []
    for shell_index in range(len(mesh.shells)):
        if shell_index not in spec.removed_shell_who:
            untouched_indices.append(shell_index)
    traces = [faces_trace(mesh, untouched_indices, NEUTRAL_SHELL, hover, "base"),
              faces_trace(mesh, removed_indices, NEUTRAL_SHELL, hover, "rm:gray", visible=False)]

    for mode, mode_label in MODES:
        visible = mode == "cmp"
        colors = {}
        outline_groups = {}
        if mode == "cmp":
            for shell_index, who in spec.removed_shell_who.items():
                colors[shell_index] = WHO_FILL[who]
            for who in WHO_FILL:
                group_indices = []
                for shell_index in removed_indices:
                    if spec.removed_shell_who[shell_index] == who:
                        group_indices.append(shell_index)
                outline_groups[who] = group_indices
        else:
            for shell_index, who in spec.removed_shell_who.items():
                if removed_by(who, mode):
                    colors[shell_index] = SIDE_FILL
                else:
                    colors[shell_index] = NEUTRAL_SHELL
            side_indices = []
            for shell_index in removed_indices:
                if removed_by(spec.removed_shell_who[shell_index], mode):
                    side_indices.append(shell_index)
            outline_groups["side"] = side_indices

        traces.append(faces_trace(mesh, removed_indices, colors, hover, f"rm:{mode}:fill", visible=visible))
        for who, group_indices in outline_groups.items():
            if who == "side":
                line_color = SIDE_LINE
            else:
                line_color = WHO_LINE[who]
            traces.append(outline_trace(mesh, group_indices, line_color, 5, f"rm:{mode}:fill", visible=visible))

        # 1D
        if mode == "cmp":
            for who in WHO_LINE:
                items = []
                for line_index, line_who in spec.removed_line_who.items():
                    if line_who == who:
                        items.append((line_index, f"{WHO_TEXT[who]} 지운 1D"))
                traces.append(lines_trace(mesh, items, WHO_LINE[who], 10, f"rm:{mode}:line1d", visible=visible))
        else:
            items = []
            for line_index, line_who in spec.removed_line_who.items():
                if removed_by(line_who, mode):
                    items.append((line_index, f"{WHO_TEXT[line_who]} 지운 1D"))
            traces.append(lines_trace(mesh, items, SIDE_LINE, 10, f"rm:{mode}:line1d", visible=visible))

    return traces + common_traces(mesh, qualities)


def result_traces(spec: ResultSpec) -> list:
    """오른쪽 결과 화면. 새로 만든 element 를 초록으로 칠한다."""
    mesh = spec.mesh
    qualities = mesh_qualities(mesh)
    hover = {}
    for shell_index in range(len(mesh.shells)):
        hover[shell_index] = shell_hover(mesh, qualities, shell_index, spec.created.get(shell_index, ""))
    created_indices = sorted(spec.created)
    other_indices = []
    for shell_index in range(len(mesh.shells)):
        if shell_index not in spec.created:
            other_indices.append(shell_index)
    traces = [faces_trace(mesh, other_indices, NEUTRAL_SHELL, hover, "base"),
              faces_trace(mesh, created_indices, CREATED, hover, "cr:fill"),
              outline_trace(mesh, created_indices, CREATED_EDGE, 5, "cr:fill"),
              faces_trace(mesh, created_indices, NEUTRAL_SHELL, hover, "cr:gray", visible=False),
              lines_trace(mesh, list(spec.created_lines.items()), CREATED_EDGE, 10, "cr:line1d")]
    return traces + common_traces(mesh, qualities)


def view_ranges(meshes: list) -> dict:
    """여러 메쉬를 담는 공통 축 범위."""
    all_points = []
    for mesh in meshes:
        for point in mesh.coords.values():
            all_points.append(point)
    points = np.array(all_points, dtype=float)
    lowest = points.min(0)
    highest = points.max(0)
    padding = 0.05 * np.maximum(highest - lowest, 1.0)
    lowest = lowest - padding
    highest = highest + padding
    ranges = {}
    axis_names = ["xaxis", "yaxis", "zaxis"]
    for axis_index in range(3):
        ranges[axis_names[axis_index]] = [float(lowest[axis_index]), float(highest[axis_index])]
    return ranges


def scene_layout(title: str, ranges: dict) -> go.Layout:
    scene = {"aspectmode": "data"}
    for axis_name, axis_range in ranges.items():
        scene[axis_name] = dict(range=axis_range, showticklabels=False, title="", autorange=False)
    return go.Layout(title=dict(text=title, x=0.5, font=dict(size=14)), scene=scene,
                     margin=dict(l=0, r=0, t=30, b=0), showlegend=False, hoverlabel=dict(align="left"))


# ──────────────────────────────────────────────────────────────────────────
# 비교 — 원본 / 모델 / 사람 셋으로 세 화면의 강조와 변경 목록을 정한다
# ──────────────────────────────────────────────────────────────────────────
class Change:
    """변경 목록의 한 줄 — 표와 '변경 위치로' 가 같은 목록을 쓴다."""

    def __init__(self, who: str, what: str, target: str, shared: bool, note: str, at: tuple):
        #: "모델" | "사람"
        self.who = who
        #: "element 삭제" | "element 생성" | "node 삭제" | "node 이동" | "1D 삭제" | "1D 생성"
        self.what = what
        #: element / node 번호
        self.target = target
        #: 상대 쪽도 같은 변경을 했나 (좌표 기준)
        self.shared = shared
        self.note = note
        #: 확대할 중심 좌표
        self.at = at


def shell_key(mesh, shell_index: int) -> frozenset:
    position_keys = []
    for node_id in mesh.shells[shell_index][2]:
        position_keys.append(pkey(mesh.coords[node_id]))
    return frozenset(position_keys)


def line_key(mesh, line_index: int) -> frozenset:
    position_keys = []
    for node_id in mesh.lines[line_index][2]:
        position_keys.append(pkey(mesh.coords[node_id]))
    return frozenset(position_keys)


def who_changed(in_model: bool, in_human: bool) -> str:
    if in_model and in_human:
        return "both"
    if in_model:
        return "model"
    return "human"


def same_change_text(both: bool, other: str) -> str:
    if both:
        return f"{other}도 같은 변경 ✓"
    return f"{other}는 안 한 변경"


def center_of(mesh, node_ids) -> tuple:
    point_list = []
    for node_id in node_ids:
        point_list.append(mesh.coords[node_id])
    return tuple(np.mean(point_list, axis=0))


def node_list_text(node_ids) -> str:
    texts = []
    for node_id in node_ids:
        texts.append(str(node_id))
    return ", ".join(texts)


def build_view(bad, model, human, model_absorbed: dict | None = None, human_absorbed: dict | None = None):
    """세 화면 명세 [원본, 모델 결과, 사람 결과] + 변경 목록 + 요약."""
    model_diff = diff(bad, model, model_absorbed)
    human_diff = diff(bad, human, human_absorbed)

    removed_by_model = set()
    for shell_index in model_diff.removed_shells:
        removed_by_model.add(shell_key(bad, shell_index))
    removed_by_human = set()
    for shell_index in human_diff.removed_shells:
        removed_by_human.add(shell_key(bad, shell_index))
    created_by_model = set()
    for shell_index in model_diff.created_shells:
        created_by_model.add(shell_key(model, shell_index))
    created_by_human = set()
    for shell_index in human_diff.created_shells:
        created_by_human.add(shell_key(human, shell_index))
    lines_removed_by_model = set()
    for line_index in model_diff.removed_lines:
        lines_removed_by_model.add(line_key(bad, line_index))
    lines_removed_by_human = set()
    for line_index in human_diff.removed_lines:
        lines_removed_by_human.add(line_key(bad, line_index))
    lines_created_by_model = set()
    for line_index in model_diff.created_lines:
        lines_created_by_model.add(line_key(model, line_index))
    lines_created_by_human = set()
    for line_index in human_diff.created_lines:
        lines_created_by_human.add(line_key(human, line_index))

    # 사라진 node 자리 — 옮긴 node 의 출발점도 그 자리가 없어진 것이다
    deleted_by_model = set()
    for node_id in model_diff.deleted_nodes:
        deleted_by_model.add(pkey(bad.coords[node_id]))
    for start_point, end_point, explanation in model_diff.moved_nodes.values():
        deleted_by_model.add(pkey(start_point))
    deleted_by_human = set()
    for node_id in human_diff.deleted_nodes:
        deleted_by_human.add(pkey(bad.coords[node_id]))
    for start_point, end_point, explanation in human_diff.moved_nodes.values():
        deleted_by_human.add(pkey(start_point))

    # ── 왼쪽 원본 ──
    original_spec = OrigSpec(bad)
    for shell_index in set(model_diff.removed_shells) | set(human_diff.removed_shells):
        key = shell_key(bad, shell_index)
        original_spec.removed_shell_who[shell_index] = who_changed(key in removed_by_model, key in removed_by_human)
    for line_index in set(model_diff.removed_lines) | set(human_diff.removed_lines):
        key = line_key(bad, line_index)
        original_spec.removed_line_who[line_index] = who_changed(key in lines_removed_by_model,
                                                                 key in lines_removed_by_human)

    # ── 오른쪽 결과 ──
    model_created = {}
    for shell_index in model_diff.created_shells:
        model_created[shell_index] = ("모델이 만듦 · "
                                      + same_change_text(shell_key(model, shell_index) in created_by_human, "사람"))
    model_created_lines = {}
    for line_index in model_diff.created_lines:
        model_created_lines[line_index] = ("모델이 만든 1D · "
                                           + same_change_text(line_key(model, line_index) in lines_created_by_human,
                                                              "사람"))
    model_spec = ResultSpec(model, "model", "모델 재생성 결과", model_created, model_created_lines)

    human_created = {}
    for shell_index in human_diff.created_shells:
        human_created[shell_index] = ("사람이 만듦 · "
                                      + same_change_text(shell_key(human, shell_index) in created_by_model, "모델"))
    human_created_lines = {}
    for line_index in human_diff.created_lines:
        human_created_lines[line_index] = ("사람이 만든 1D · "
                                           + same_change_text(line_key(human, line_index) in lines_created_by_model,
                                                              "모델"))
    human_spec = ResultSpec(human, "human", "사람 수정 결과", human_created, human_created_lines)

    # ── 변경 목록 ──
    changes = []
    sides = [
        ("모델", model_diff, model, removed_by_human, created_by_human, lines_removed_by_human,
         lines_created_by_human, deleted_by_human),
        ("사람", human_diff, human, removed_by_model, created_by_model, lines_removed_by_model,
         lines_created_by_model, deleted_by_model),
    ]
    for (who, mesh_diff, result_mesh, other_removed, other_created, other_lines_removed, other_lines_created,
         other_deleted) in sides:
        for shell_index in mesh_diff.removed_shells:
            element_id, element_type, node_ids = bad.shells[shell_index]
            changes.append(Change(who, "element 삭제", f"{element_type} {element_id}",
                                  shell_key(bad, shell_index) in other_removed,
                                  f"node {node_list_text(node_ids)}", center_of(bad, node_ids)))
        for shell_index in mesh_diff.created_shells:
            element_id, element_type, node_ids = result_mesh.shells[shell_index]
            changes.append(Change(who, "element 생성", f"{element_type} {element_id}",
                                  shell_key(result_mesh, shell_index) in other_created,
                                  f"node {node_list_text(node_ids)}", center_of(result_mesh, node_ids)))
        for node_id, explanation in mesh_diff.deleted_nodes.items():
            changes.append(Change(who, "node 삭제", f"GRID {node_id}", pkey(bad.coords[node_id]) in other_deleted,
                                  explanation, tuple(bad.coords[node_id])))
        for node_id, (start_point, end_point, explanation) in mesh_diff.moved_nodes.items():
            middle = tuple((np.asarray(start_point) + np.asarray(end_point)) / 2)
            changes.append(Change(who, "node 이동", f"GRID {node_id}", pkey(start_point) in other_deleted,
                                  explanation, middle))
        for line_index in mesh_diff.removed_lines:
            element_id, element_type, (node_a, node_b) = bad.lines[line_index]
            changes.append(Change(who, "1D 삭제", f"{element_type} {element_id}",
                                  line_key(bad, line_index) in other_lines_removed, f"node {node_a}-{node_b}",
                                  center_of(bad, (node_a, node_b))))
        for line_index in mesh_diff.created_lines:
            element_id, element_type, (node_a, node_b) = result_mesh.lines[line_index]
            changes.append(Change(who, "1D 생성", f"{element_type} {element_id}",
                                  line_key(result_mesh, line_index) in other_lines_created, f"node {node_a}-{node_b}",
                                  center_of(result_mesh, (node_a, node_b))))

    summary = {
        "model": {"지운 element": len(model_diff.removed_shells), "만든 element": len(model_diff.created_shells),
                  "사라진 node": len(model_diff.deleted_nodes), "옮긴 node": len(model_diff.moved_nodes),
                  "1D 삭제": len(model_diff.removed_lines), "1D 생성": len(model_diff.created_lines)},
        "human": {"지운 element": len(human_diff.removed_shells), "만든 element": len(human_diff.created_shells),
                  "사라진 node": len(human_diff.deleted_nodes), "옮긴 node": len(human_diff.moved_nodes),
                  "1D 삭제": len(human_diff.removed_lines), "1D 생성": len(human_diff.created_lines)},
        "stats": {"원본": stats(bad), "모델": stats(model), "사람": stats(human)},
    }
    return [original_spec, model_spec, human_spec], changes, summary


# ──────────────────────────────────────────────────────────────────────────
# HTML — 아래 문자열은 웹 페이지(HTML/CSS/JavaScript) 틀이다. {이름} 자리에 값을 넣는다.
# ──────────────────────────────────────────────────────────────────────────
PAGE_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<script>{plotlyjs}</script>
<style>
  :root {{ color-scheme: light; }}
  html, body {{ height:100%; }}
  /* 도구막대가 몇 줄로 접히든 남는 높이를 세 화면이 다 쓰도록 세로 flex 로 둔다 */
  body {{ margin:0; font-family:"Malgun Gothic","Segoe UI",system-ui,sans-serif;
         background:#f8fafc; color:#0f172a; font-size:13px;
         display:flex; flex-direction:column; }}
  .bar {{ display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; padding:7px 12px;
          background:#fff; border-bottom:1px solid #e2e8f0; flex:none; }}
  .bar label {{ cursor:pointer; user-select:none; white-space:nowrap; }}
  .bar button, .bar select {{ font:inherit; padding:2px 9px; border:1px solid #cbd5e1;
                 background:#fff; border-radius:5px; cursor:pointer; }}
  .bar button:hover {{ background:#f1f5f9; }}
  .bar select {{ max-width:430px; }}
  .group {{ display:inline-flex; gap:8px; align-items:center; padding:2px 8px;
            border:1px solid #e2e8f0; border-radius:6px; background:#f8fafc; }}
  .group b {{ font-weight:600; color:#334155; }}
  .key {{ display:inline-block; width:11px; height:11px; border-radius:2px;
          vertical-align:-1px; margin-right:4px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr;
           gap:1px; background:#cbd5e1; flex:1; min-height:0; }}
  .pane {{ background:#fff; min-height:0; }}
  .pane.orig {{ grid-row:1 / span 2; }}
  .pane > div {{ width:100%; height:100%; }}
  .sync {{ color:#059669; font-weight:600; }}
</style>
<div class="bar">
  <button id="reset">시점 초기화</button>
  <select id="focus"><option value="">변경 위치로 확대…</option>{focus_opts}</select>
  <span class="group"><b>원본에 표시할 삭제</b>{modes}</span>
  <span class="group">{legend}</span>
  {toggles}
  <span class="sync">↔ 세 화면 시점 연동</span>
</div>
<div class="grid">
  <div class="pane orig"><div id="P0"></div></div>
  <div class="pane"><div id="P1"></div></div>
  <div class="pane"><div id="P2"></div></div>
</div>
<script>
const FIGS = {figs};
const FOCUS = {focus};
const RANGES = {ranges};
const cfg = {{responsive:true, displaylogo:false, modeBarButtonsToRemove:['toImage']}};
const ids = ['P0','P1','P2'];
Promise.all(ids.map((id, i) => Plotly.newPlot(id, FIGS[i].data, FIGS[i].layout, cfg))).then(() => {{
  const G = ids.map(id => document.getElementById(id));

  // ── 카메라 연동 — 매 프레임 비교. 기준값은 '보낸 카메라 그 자체' 로 맞춘다 ──
  const cam = gd => gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.camera;
  const key = c => c ? JSON.stringify([c.eye, c.center, c.up, c.projection && c.projection.type]) : '';
  let last = null, busy = false;
  function tick() {{
    if (busy) return;
    const ks = G.map(g => key(cam(g)));
    if (ks.some(k => !k)) return;
    if (last === null) {{ last = ks; return; }}
    const src = ks.findIndex((k, i) => k !== last[i]);
    if (src < 0) return;
    busy = true;
    const snap = JSON.parse(JSON.stringify(cam(G[src])));
    const settle = () => {{ last = ks.map(() => key(snap)); busy = false; }};
    Promise.all(G.map((g, i) => i === src ? null : Plotly.relayout(g, {{'scene.camera': snap}})))
      .then(settle, settle);
  }}
  (function loop() {{ tick(); requestAnimationFrame(loop); }})();
  setInterval(tick, 150);   // 창이 가려져 rAF 가 멈출 때 대비

  // ── 강조 상태 → 트레이스 보이기 ──────────────────────────────────────
  // 트레이스마다 meta 에 묶음 이름이 있다. 상태 하나에서 모든 트레이스의 보이기를 다시 계산한다.
  const S = {{mode: 'cmp'}};
  document.querySelectorAll('.bar input[data-g]').forEach(b => S[b.dataset.g] = b.checked);
  function shown(meta) {{
    if (meta === 'base' || meta === 'focus') return true;
    if (meta in S) return !!S[meta];
    if (meta === 'rm:gray') return !S.fill_rm;
    if (meta === 'cr:gray') return !S.fill_cr;
    const p = meta.split(':');
    if (p[0] === 'rm') {{
      const want = {{fill:'fill_rm', line1d:'line1d'}}[p[2]];
      return p[1] === S.mode && !!S[want];
    }}
    if (p[0] === 'cr') return !!S[{{fill:'fill_cr', line1d:'line1d'}}[p[1]]];
    return true;
  }}
  function apply() {{
    G.forEach(g => {{
      const vis = g.data.map(t => shown(t.meta));
      Plotly.restyle(g, {{visible: vis}}, g.data.map((_, i) => i));
    }});
  }}
  document.querySelectorAll('.bar input[data-g]').forEach(b =>
    b.addEventListener('change', () => {{ S[b.dataset.g] = b.checked; apply(); }}));
  document.querySelectorAll('.bar input[name=mode]').forEach(r =>
    r.addEventListener('change', () => {{ if (r.checked) {{ S.mode = r.value; apply(); }} }}));
  apply();

  // ── 변경 위치로 확대 / 시점 초기화 ─────────────────────────────────
  function setRanges(r, pt) {{
    busy = true;
    const upd = {{'scene.xaxis.range': r.xaxis, 'scene.yaxis.range': r.yaxis,
                 'scene.zaxis.range': r.zaxis}};
    const jobs = [];
    G.forEach(g => {{
      jobs.push(Plotly.relayout(g, upd));
      const fi = g.data.findIndex(t => t.meta === 'focus');
      if (fi >= 0) jobs.push(Plotly.restyle(g, pt ? {{x:[[pt[0]]], y:[[pt[1]]], z:[[pt[2]]]}}
                                                   : {{x:[[]], y:[[]], z:[[]]}}, [fi]));
    }});
    Promise.all(jobs).then(() => {{ last = null; busy = false; }}, () => {{ busy = false; }});
  }}
  document.getElementById('focus').addEventListener('change', e => {{
    const v = e.target.value;
    if (v === '') {{ setRanges(RANGES, null); return; }}
    const f = FOCUS[+v], h = f.r;
    setRanges({{xaxis:[f.p[0]-h, f.p[0]+h], yaxis:[f.p[1]-h, f.p[1]+h], zaxis:[f.p[2]-h, f.p[2]+h]}}, f.p);
  }});
  document.getElementById('reset').addEventListener('click', () => {{
    document.getElementById('focus').value = '';
    busy = true;
    const eye = {{eye:{{x:1.45,y:1.45,z:1.0}}, center:{{x:0,y:0,z:0}}, up:{{x:0,y:0,z:1}},
                 projection:{{type:'perspective'}}}};
    Promise.all(G.map(g => Plotly.relayout(g, {{'scene.camera': eye,
        'scene.xaxis.range': RANGES.xaxis, 'scene.yaxis.range': RANGES.yaxis,
        'scene.zaxis.range': RANGES.zaxis}})))
      .then(() => {{ G.forEach(g => {{ const fi = g.data.findIndex(t => t.meta === 'focus');
                                       if (fi >= 0) Plotly.restyle(g, {{x:[[]], y:[[]], z:[[]]}}, [fi]); }});
                    last = null; busy = false; }}, () => {{ busy = false; }});
  }});
  window.__regen = {{ready: true, state: () => S,
                     visible: () => G.map(g => g.data.map(t => [t.meta, t.visible]))}};
}});
</script>
"""


def render_html(panes: list, changes: list, focus_radius: float) -> str:
    """세 화면 HTML. ``panes`` = [OrigSpec, ResultSpec(모델), ResultSpec(사람)].

    ``focus_radius`` 는 '변경 위치로' 확대할 때 상자 반폭(mm).
    """
    original_spec, model_spec, human_spec = panes
    ranges = view_ranges([original_spec.mesh, model_spec.mesh, human_spec.mesh])
    figures = [go.Figure(data=orig_traces(original_spec), layout=scene_layout(original_spec.title, ranges)),
               go.Figure(data=result_traces(model_spec), layout=scene_layout(model_spec.title, ranges)),
               go.Figure(data=result_traces(human_spec), layout=scene_layout(human_spec.title, ranges))]

    modes_html = ""
    for value, label in MODES:
        checked = ""
        if value == "cmp":
            checked = " checked"
        modes_html = modes_html + (f'<label><input type="radio" name="mode" value="{value}"{checked}> '
                                   f'{html.escape(label)}</label>')

    legend_html = ""
    legend_items = [(WHO_FILL["both"], "둘 다"), (WHO_FILL["model"], "모델만"),
                    (WHO_FILL["human"], "사람만"), (CREATED, "새로 만든 element")]
    for color, text in legend_items:
        legend_html = legend_html + (f'<span><span class="key" style="background:{color}"></span>'
                                     f'{html.escape(text)}</span>')

    toggles_html = ""
    for group_name, label, color, initially_on in TOGGLES:
        checked = ""
        if initially_on:
            checked = " checked"
        toggles_html = toggles_html + (f'<label><input type="checkbox" data-g="{group_name}"{checked}>'
                                       f'<span class="key" style="background:{color}"></span>'
                                       f'{html.escape(label)}</label>')

    options_html = ""
    for change_index in range(len(changes)):
        change = changes[change_index]
        other_side_text = ""
        if not change.shared:
            other_side_text = "  (상대 쪽에 없음)"
        options_html = options_html + (f'<option value="{change_index}">{html.escape(change.who)} · '
                                       f'{html.escape(change.what)} · {html.escape(change.target)}'
                                       f'{other_side_text}</option>')

    focus_list = []
    for change in changes:
        center = []
        for value in change.at:
            center.append(float(value))
        focus_list.append({"p": center, "r": float(focus_radius)})

    figure_json_list = []
    for figure in figures:
        figure_json_list.append(figure.to_plotly_json())
    return PAGE_TEMPLATE.format(
        plotlyjs=get_plotlyjs(),
        figs=json.dumps(figure_json_list, cls=PlotlyJSONEncoder),
        focus=json.dumps(focus_list), ranges=json.dumps(ranges),
        focus_opts=options_html, modes=modes_html, legend=legend_html, toggles=toggles_html,
    )

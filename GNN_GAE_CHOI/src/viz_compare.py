"""수리 전/후 메쉬를 카메라 연동된 2분할 3D 뷰어로 내보낸다.

한쪽을 돌리거나 이동하면 반대쪽이 똑같이 따라간다 — 같은 각도에서 봐야 무엇이 바뀌었는지
알 수 있기 때문이다. 색으로도 바로 구분된다.

    왼쪽(수리 전)   빨강 = 이 연산으로 없어질 요소
    오른쪽(수리 후) 초록 = 새로 생긴 요소
    양쪽            회색 = 그대로,  검은 테두리 = 불량각 요소(내각<30° 또는 >150°)
                    파랑 = 1D 요소,  주황 = 삼각형 병합 때 같이 지워야 하는 1D 요소
                    하늘색 원 = ext 절점 (움직일 수 없는 절단면)

plotly.js 를 파일 안에 넣어 자체 완결 HTML 로 쓴다 — 네트워크 없이 열린다.

    python src/viz_compare.py --best 3
    python src/viz_compare.py --name SN2592_FWD_Clean_0694
    python src/viz_compare.py --name SN2592_FWD_Clean_0001 --corrupt 6
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

import gae_paths
import features as F
from meshedit import EditableMesh, EditError, merge_candidates, shared_edge_of
from meshrep import Edge
from repair_rule import RepairConfig, repair

# 앱들이 같은 팔레트를 쓴다 — 뷰어를 오가며 봐도 색 의미가 안 바뀌도록.
NEUTRAL_SHELL = "#c7ced9"
#: 쉘 경계선. "요소가 어떻게 나뉘어 있나" 가 제일 중요한 선이라 진하게 둔다.
SHELL_EDGE = "#334155"
NEUTRAL_LINE = "#1d4ed8"
BAD_OUTLINE = "#111827"
EXT_MARKER = "#06b6d4"
REMOVED = "#ef4444"       # 수리 전 화면에서 없어질 요소
CHANGED_NODE = "#a21caf"  # 사라지거나 새로 생긴 절점 — 빨강/초록과 섞이지 않는 자주색
MOVED_NODE = "#eab308"    # 자리를 옮긴 절점 — 사라진 것과 구분되는 노랑
ADDED = "#22c55e"         # 수리 후 화면에서 새로 생긴 요소
BLOCKER = "#f97316"       # 변을 없앨 때 같이 지워야 하는 1D 요소

#: 트레이스 순서. 양쪽 화면이 같은 순서를 쓰므로 체크박스 하나로 둘 다 켜고 끌 수 있다.
GROUPS = ["shells", "edges", "bad", "lines", "blockers", "ext", "nodes", "moves"]


# ──────────────────────────────────────────────────────────────────────────
# 트레이스 (plotly 그림 요소)
# ──────────────────────────────────────────────────────────────────────────
def blocking_line_edges(mesh: EditableMesh) -> set:
    """인접 삼각형 쌍의 공유 변 위에 올라타 있는 1D 요소의 변.

    금지가 아니라 **같이 처리해야 하는 대상**이다 — 사람도 이런 변 151개 중 35개를 없애며
    그 위의 1D 요소를 같이 지웠다.
    """
    edges = set()
    for candidate in merge_candidates(mesh, drop_lines=True):
        if candidate.lines_on_edge > 0:
            edges.add(shared_edge_of(mesh, candidate.si, candidate.sj))
    return edges


def add_segment(x_values: list, y_values: list, z_values: list, point_a, point_b) -> None:
    """선분 하나를 좌표 목록에 붙인다 (None 으로 다음 선분과 끊는다)."""
    x_values.extend([point_a[0], point_b[0], None])
    y_values.extend([point_a[1], point_b[1], None])
    z_values.extend([point_a[2], point_b[2], None])


def marker_points(mesh: EditableMesh, marks: dict):
    """{절점: 설명} 중 메쉬에 있는 절점만 → (좌표 배열 (k,3), 설명 목록)."""
    point_list = []
    text_list = []
    for node_id, text in marks.items():
        if node_id in mesh.coords:
            point_list.append(mesh.coords[node_id])
            text_list.append(text)
    if len(point_list) > 0:
        points = np.array(point_list)
    else:
        points = np.empty((0, 3))
    return points, text_list


def mesh_traces(mesh: EditableMesh, mark: set, mark_color: str, mark_label: str,
                node_marks: dict | None = None, node_color: str = CHANGED_NODE, node_label: str = "바뀐 절점",
                move_marks: dict | None = None) -> list:
    """메쉬 하나를 GROUPS 순서의 트레이스 목록으로 만든다.

    ``mark`` 는 강조할 요소의 절점 집합들 (수리 전이면 없어질 것, 후면 새로 생긴 것).
    비어 있는 트레이스도 반드시 만든다 — 양쪽 화면의 트레이스 인덱스를 맞춰야 한다.
    """
    # ── 쉘 면 ──
    x_values, y_values, z_values = [], [], []
    triangle_i, triangle_j, triangle_k = [], [], []
    face_colors = []
    hover_texts = []
    vertex_count = 0
    for element_id, element_type, node_ids in mesh.shells:
        corner_points = []
        for node_id in node_ids:
            corner_points.append(mesh.coords[node_id])
        first_vertex = vertex_count
        for point in corner_points:
            x_values.append(point[0])
            y_values.append(point[1])
            z_values.append(point[2])
        vertex_count = vertex_count + len(corner_points)
        # 사각형은 삼각형 2개로 그린다
        drawn_triangles = [(first_vertex, first_vertex + 1, first_vertex + 2)]
        if len(corner_points) == 4:
            drawn_triangles.append((first_vertex, first_vertex + 2, first_vertex + 3))

        is_marked = frozenset(node_ids) in mark
        if is_marked:
            color = mark_color
        else:
            color = NEUTRAL_SHELL
        quality = F.shell_quality(np.array(corner_points, dtype=float))
        bad_text = ""
        if F.is_bad_shell(quality):
            bad_text = " <b>불량</b>"
        node_text_list = []
        for node_id in node_ids:
            node_text_list.append(str(node_id))
        hover = (f"<b>{element_type} {element_id}</b><br>"
                 f"내각 {quality.angle_min:.1f}°~{quality.angle_max:.1f}°{bad_text}<br>"
                 f"aspect {quality.aspect:.2f}  skew {quality.skew:.1f}°<br>"
                 f"절점 {', '.join(node_text_list)}")
        if is_marked:
            hover = hover + f"<br><b>{mark_label}</b>"
        for vertex_a, vertex_b, vertex_c in drawn_triangles:
            triangle_i.append(vertex_a)
            triangle_j.append(vertex_b)
            triangle_k.append(vertex_c)
            face_colors.append(color)
        for _ in corner_points:
            hover_texts.append(hover)

    traces = [go.Mesh3d(x=x_values, y=y_values, z=z_values, i=triangle_i, j=triangle_j, k=triangle_k,
                        facecolor=face_colors, flatshading=True, opacity=1.0, text=hover_texts,
                        hoverinfo="text", name="쉘 요소")]

    rep = mesh.rep

    # ── 쉘 경계선 ──
    edge_x, edge_y, edge_z = [], [], []
    for edge in rep.edge_shells:
        node_a, node_b = tuple(edge)
        add_segment(edge_x, edge_y, edge_z, mesh.coords[node_a], mesh.coords[node_b])
    traces.append(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines",
                               line=dict(color=SHELL_EDGE, width=2.5), name="쉘 경계", hoverinfo="skip"))

    # ── 불량각 요소 테두리 ──
    outline_x, outline_y, outline_z = [], [], []
    for element_id, element_type, node_ids in mesh.shells:
        corner_points = []
        for node_id in node_ids:
            corner_points.append(mesh.coords[node_id])
        if not F.is_bad_shell(F.shell_quality(np.array(corner_points, dtype=float))):
            continue
        for point in corner_points + [corner_points[0]]:
            outline_x.append(point[0])
            outline_y.append(point[1])
            outline_z.append(point[2])
        outline_x.append(None)
        outline_y.append(None)
        outline_z.append(None)
    traces.append(go.Scatter3d(x=outline_x, y=outline_y, z=outline_z, mode="lines",
                               line=dict(color=BAD_OUTLINE, width=5), name="불량각 요소", hoverinfo="skip"))

    # ── 1D 요소 — 같이 지워야 하는 것과 아닌 것을 나눈다 ──
    blockers = blocking_line_edges(mesh)
    for want_blocker, color, label in ((False, NEUTRAL_LINE, "1D 요소 (CBAR/CBEAM)"),
                                       (True, BLOCKER, "같이 지워야 하는 1D 요소")):
        line_x, line_y, line_z, line_texts = [], [], [], []
        for element_id, element_type, (node_a, node_b) in mesh.lines:
            is_blocker = Edge((node_a, node_b)) in blockers
            if is_blocker != want_blocker:
                continue
            add_segment(line_x, line_y, line_z, mesh.coords[node_a], mesh.coords[node_b])
            text = f"<b>{element_type} {element_id}</b><br>절점 {node_a}-{node_b}"
            if want_blocker:
                text = text + "<br><b>양옆 삼각형을 합치려면 이 1D 요소도 같이 지워야 함</b>"
            line_texts.extend([text, text, ""])
        if want_blocker:
            width = 7
        else:
            width = 4
        traces.append(go.Scatter3d(x=line_x, y=line_y, z=line_z, mode="lines", line=dict(color=color, width=width),
                                   text=line_texts, hoverinfo="text", name=label))

    # ── ext 절점 — 움직일 수 없는 절단면 ──
    ext_point_list = []
    for node_id in sorted(mesh.coords):
        if mesh.ext.get(node_id, False):
            ext_point_list.append(mesh.coords[node_id])
    ext_points = np.array(ext_point_list)
    if len(ext_points) > 0:
        ext_x, ext_y, ext_z = ext_points[:, 0], ext_points[:, 1], ext_points[:, 2]
    else:
        ext_x, ext_y, ext_z = [], [], []
    traces.append(go.Scatter3d(x=ext_x, y=ext_y, z=ext_z, mode="markers",
                               marker=dict(size=4, color=EXT_MARKER, symbol="circle-open", line=dict(width=2)),
                               name="ext 절점 (고정)", hoverinfo="skip"))

    # ── 사라지거나 새로 생긴 절점 — 쌍 비교에서만 쓴다. 안 줘도 트레이스는 만든다 ──
    # (양쪽 화면의 트레이스 번호가 어긋나면 체크박스가 엉뚱한 걸 켜고 끈다)
    if node_marks is None:
        node_marks = {}
    node_points, node_texts = marker_points(mesh, node_marks)
    traces.append(go.Scatter3d(x=node_points[:, 0], y=node_points[:, 1], z=node_points[:, 2], mode="markers",
                               marker=dict(size=9, color=node_color, symbol="x", line=dict(width=2, color=node_color)),
                               text=node_texts, hoverinfo="text", name=node_label))

    # ── 자리를 옮긴 절점. 사라진 절점과 색·모양을 달리한다 ──
    # 옮겨간 자리가 사라진 절점의 자리와 정확히 겹치는 일이 흔해서(실측 99건), 같은 색이면
    # 좌우 화면에 같은 마커가 찍힌 것처럼 보인다.
    if move_marks is None:
        move_marks = {}
    move_points, move_texts = marker_points(mesh, move_marks)
    traces.append(go.Scatter3d(x=move_points[:, 0], y=move_points[:, 1], z=move_points[:, 2], mode="markers",
                               marker=dict(size=8, color=MOVED_NODE, symbol="diamond",
                                           line=dict(width=1, color="#854d0e")),
                               text=move_texts, hoverinfo="text", name="자리를 옮긴 절점"))
    return traces


def common_ranges(mesh_a: EditableMesh, mesh_b: EditableMesh) -> dict:
    """두 메쉬를 담는 공통 축 범위. 같은 축 범위라야 두 화면이 픽셀 단위로 비교된다."""
    all_points = list(mesh_a.coords.values()) + list(mesh_b.coords.values())
    points = np.array(all_points, dtype=float)
    lowest = points.min(axis=0)
    highest = points.max(axis=0)
    padding = 0.05 * np.maximum(highest - lowest, 1.0)
    lowest = lowest - padding
    highest = highest + padding
    ranges = {}
    axis_names = ["xaxis", "yaxis", "zaxis"]
    for axis_index in range(3):
        ranges[axis_names[axis_index]] = [float(lowest[axis_index]), float(highest[axis_index])]
    return ranges


def scene_layout(title: str, ranges: dict) -> go.Layout:
    return go.Layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        scene=dict(
            aspectmode="data",
            xaxis=dict(range=ranges["xaxis"], showticklabels=False, title=""),
            yaxis=dict(range=ranges["yaxis"], showticklabels=False, title=""),
            zaxis=dict(range=ranges["zaxis"], showticklabels=False, title=""),
        ),
        margin=dict(l=0, r=0, t=34, b=0),
        showlegend=False,
        hoverlabel=dict(align="left"),
    )


# ──────────────────────────────────────────────────────────────────────────
# 통계
# ──────────────────────────────────────────────────────────────────────────
def stats(mesh: EditableMesh) -> dict:
    """화면 표에 쓰는 품질 요약."""
    qualities = []
    for shell_index in range(len(mesh.shells)):
        qualities.append(F.shell_quality(mesh.corners(shell_index)))
    triangle_count = 0
    bad_count = 0
    largest_skew = 0
    largest_aspect = 0
    for quality in qualities:
        if not quality.is_quad:
            triangle_count = triangle_count + 1
        if F.is_bad_shell(quality):
            bad_count = bad_count + 1
    if len(qualities) > 0:
        largest_skew = qualities[0].skew
        largest_aspect = qualities[0].aspect
        for quality in qualities:
            if quality.skew > largest_skew:
                largest_skew = quality.skew
            if quality.aspect > largest_aspect:
                largest_aspect = quality.aspect
    return {
        "쉘 요소": len(qualities),
        "삼각형": triangle_count,
        "삼각형 비율": f"{triangle_count / max(1, len(qualities)):.3f}",
        "불량각 요소": bad_count,
        "최대 skew": f"{largest_skew:.1f}°",
        "최대 aspect": f"{largest_aspect:.2f}",
    }


def operation_text(operation) -> str:
    """편집 연산 하나를 한 줄 설명으로."""
    node_text_list = []
    for node_id in operation.key:
        node_text_list.append(str(node_id))
    node_text = ", ".join(node_text_list)
    if operation.kind == "merge":
        return f"삼각형 2개 → 사각형 1개 (절점 {node_text})"
    if operation.kind == "swap":
        return f"대각선 뒤집기 (절점 {node_text})"
    if operation.kind == "split":
        return f"사각형 → 삼각형 2개 (절점 {node_text})"
    return f"절점 {operation.key[0]} 이동"


# ──────────────────────────────────────────────────────────────────────────
# HTML — 아래 문자열은 웹 페이지(HTML/CSS/JavaScript) 틀이다. {이름} 자리에 값을 넣는다.
# ──────────────────────────────────────────────────────────────────────────
PAGE_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<script>{plotlyjs}</script>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin:0; font-family:"Malgun Gothic","Segoe UI",system-ui,sans-serif;
         background:#f8fafc; color:#0f172a; }}
  header {{ padding:14px 18px 10px; border-bottom:1px solid #e2e8f0; background:#fff; }}
  h1 {{ margin:0 0 4px; font-size:17px; }}
  .sub {{ font-size:13px; color:#475569; }}
  .bar {{ display:flex; flex-wrap:wrap; gap:16px; align-items:center;
          padding:9px 18px; background:#fff; border-bottom:1px solid #e2e8f0;
          font-size:13px; position:sticky; top:0; z-index:5; }}
  .bar label {{ cursor:pointer; user-select:none; }}
  .bar button {{ font:inherit; padding:3px 11px; border:1px solid #cbd5e1;
                 background:#fff; border-radius:5px; cursor:pointer; }}
  .bar button:hover {{ background:#f1f5f9; }}
  .sync {{ color:#059669; font-weight:600; }}
  .views {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#e2e8f0; }}
  .pane {{ background:#fff; }}
  .plot {{ height:min(68vh, 760px); }}
  table {{ border-collapse:collapse; font-size:12.5px; width:100%; }}
  td, th {{ padding:3px 10px; text-align:right; border-top:1px solid #f1f5f9; }}
  th:first-child, td:first-child {{ text-align:left; color:#475569; }}
  .chg {{ color:#059669; font-weight:600; }}
  .neg {{ color:#dc2626; font-weight:600; }}
  footer {{ padding:12px 18px 26px; font-size:13px; }}
  ol {{ margin:6px 0 0; padding-left:22px; }}
  li {{ margin:2px 0; }}
  .key {{ display:inline-block; width:11px; height:11px; border-radius:2px;
          vertical-align:-1px; margin-right:5px; }}
  .empty {{ color:#b45309; }}
</style>
{header_html}
<div class="bar">
  <button id="reset">시점 초기화</button>
  <label><input type="checkbox" data-g="edges" checked> 쉘 경계</label>
  <label><input type="checkbox" data-g="bad" checked>
    <span class="key" style="background:{bad}"></span>불량각 요소</label>
  <label><input type="checkbox" data-g="lines" checked>
    <span class="key" style="background:{line}"></span>1D 요소</label>
  <label><input type="checkbox" data-g="blockers" checked>
    <span class="key" style="background:{blocker}"></span>같이 지울 1D</label>
  <label><input type="checkbox" data-g="ext" checked>
    <span class="key" style="background:{ext}"></span>ext 절점</label>
  <label><input type="checkbox" data-g="nodes" checked>
    <span class="key" style="background:{chnode}"></span>사라짐/생김</label>
  <label><input type="checkbox" data-g="moves" checked>
    <span class="key" style="background:{mvnode}"></span>자리 옮김</label>
  <span class="sync">↔ 두 화면의 시점은 연동됩니다</span>
</div>
<div class="views">
  <div class="pane"><div id="L" class="plot"></div>{tableL}</div>
  <div class="pane"><div id="R" class="plot"></div>{tableR}</div>
</div>
{footer_html}
<script>
const GROUPS = {groups};
const figL = {figL}, figR = {figR};
const cfg = {{responsive:true, displaylogo:false,
              modeBarButtonsToRemove:['toImage']}};

Promise.all([
  Plotly.newPlot('L', figL.data, figL.layout, cfg),
  Plotly.newPlot('R', figR.data, figR.layout, cfg)
]).then(() => {{
  const L = document.getElementById('L'), R = document.getElementById('R');

  // ── 카메라 연동 ────────────────────────────────────────────────────
  // plotly 의 'plotly_relayout' 이벤트는 3D 드래그 때 놓치는 경우가 있어서(plotly 6.5 실측)
  // 매 프레임 두 씬의 카메라를 직접 비교해서, 바뀐 쪽을 반대쪽에 밀어넣는다.
  const cam = gd => gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.camera;
  const key = c => c ? JSON.stringify([c.eye, c.center, c.up,
                                       c.projection && c.projection.type]) : '';
  let lastL = null, lastR = null, busy = false;

  function tick() {{
    const cl = cam(L), cr = cam(R);
    if (!cl || !cr || busy) return;
    const kl = key(cl), kr = key(cr);
    if (lastL === null) {{
      lastL = kl; lastR = kr;
    }} else if (kl !== lastL) {{
      push(L, R, cl);
    }} else if (kr !== lastR) {{
      push(R, L, cr);
    }}
  }}

  // src 의 카메라를 dst 에 복사한다. 기준값(lastL/lastR)을 **보낸 카메라 그 자체**로 맞춘다 —
  // 복사가 끝난 뒤 '현재' 카메라를 기준으로 삼으면 그 사이 더 끈 만큼이 흡수돼 반대쪽이 영구히 뒤처진다.
  function push(src, dst, camera) {{
    busy = true;
    const snap = JSON.parse(JSON.stringify(camera));
    const settle = () => {{ lastL = lastR = key(snap); busy = false; }};
    Plotly.relayout(dst, {{'scene.camera': snap}}).then(settle, settle);
  }}
  (function loop() {{ tick(); requestAnimationFrame(loop); }})();
  // 창이 가려지면 브라우저가 rAF 를 멈춘다 — 타이머로 한 번 더 받쳐준다.
  setInterval(tick, 150);
  window.__sync = {{get ready() {{ return true; }},
                   get state() {{ return {{lastL, lastR, busy}}; }}}};

  document.querySelectorAll('.bar input[data-g]').forEach(box => {{
    box.addEventListener('change', () => {{
      const idx = GROUPS[box.dataset.g];
      if (idx === undefined) return;
      Plotly.restyle(L, {{visible: box.checked ? true : 'legendonly'}}, [idx]);
      Plotly.restyle(R, {{visible: box.checked ? true : 'legendonly'}}, [idx]);
    }});
  }});

  document.getElementById('reset').addEventListener('click', () => {{
    busy = true;
    const eye = {{eye:{{x:1.45,y:1.45,z:1.0}}, center:{{x:0,y:0,z:0}},
                 up:{{x:0,y:0,z:1}}, projection:{{type:'perspective'}}}};
    const settle = () => {{ lastL = lastR = key(eye); busy = false; }};
    Promise.all([Plotly.relayout(L, {{'scene.camera': eye}}),
                 Plotly.relayout(R, {{'scene.camera': eye}})]).then(settle, settle);
  }});
}});
</script>
"""


def stats_table_html(before_stats: dict, after_stats: dict, side: str) -> str:
    """화면 아래 지표 표. side = "L"(수리 전) 또는 "R"(수리 후)."""
    rows = []
    for key in before_stats:
        before_value = before_stats[key]
        after_value = after_stats[key]
        if side == "L":
            rows.append(f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(before_value))}</td></tr>")
        else:
            css_class = ""
            if key in ("삼각형", "불량각 요소") and isinstance(before_value, int) and isinstance(after_value, int):
                if after_value < before_value:
                    css_class = ' class="chg"'
                elif after_value > before_value:
                    css_class = ' class="neg"'
            rows.append(f"<tr><td>{html.escape(key)}</td><td{css_class}>{html.escape(str(after_value))}</td></tr>")
    if side == "L":
        head = "수리 전"
    else:
        head = "수리 후"
    return f"<table><tr><th>지표</th><th>{head}</th></tr>{''.join(rows)}</table>"


def group_index_json() -> str:
    """{"shells": 0, "edges": 1, ...} JSON."""
    index_of_group = {}
    for group_index in range(len(GROUPS)):
        index_of_group[GROUPS[group_index]] = group_index
    return json.dumps(index_of_group)


def render_html(before: EditableMesh, after: EditableMesh, operations: list, subtitle: str, embed: bool = False,
                node_marks_before: dict | None = None, node_marks_after: dict | None = None,
                move_marks_before: dict | None = None, move_marks_after: dict | None = None) -> str:
    """수리 전/후 2분할 뷰어 HTML 을 문자열로 만든다.

    ``embed=True`` 면 제목·통계표·연산목록을 빼고 **3D 화면과 도구막대만** 남긴다.
    Streamlit 앱이 그 부분을 자기 위젯으로 그리기 때문이다.
    """
    before_keys = set()
    for shell in before.shells:
        before_keys.add(frozenset(shell[2]))
    after_keys = set()
    for shell in after.shells:
        after_keys.add(frozenset(shell[2]))
    removed = before_keys - after_keys
    added = after_keys - before_keys

    # 임베드할 때는 메쉬 이름을 Streamlit 쪽에서 이미 보여주므로 화면 제목을 짧게 둔다
    if embed:
        title_left = "수리 전"
        title_right = "수리 후"
    else:
        title_left = f"수리 전 — {before.name}"
        title_right = f"수리 후 — {after.name}"

    ranges = common_ranges(before, after)
    figure_left = go.Figure(
        data=mesh_traces(before, removed, REMOVED, "이 연산으로 없어짐", node_marks=node_marks_before,
                         node_label="사라지는 절점", move_marks=move_marks_before),
        layout=scene_layout(title_left, ranges),
    )
    figure_right = go.Figure(
        data=mesh_traces(after, added, ADDED, "새로 생긴 요소", node_marks=node_marks_after,
                         node_label="새로 생긴 절점", move_marks=move_marks_after),
        layout=scene_layout(title_right, ranges),
    )

    before_stats = stats(before)
    after_stats = stats(after)
    if embed:
        # iframe 높이가 곧 화면 높이다 — 도구막대를 뺀 나머지를 3D 가 다 쓰게 한다.
        header_html = "<style>.plot { height:calc(100vh - 62px); }</style>"
        footer_html = ""
        table_left = ""
        table_right = ""
    else:
        header_html = (f'<header><h1>메쉬 수리 전/후 비교 — {html.escape(before.name)}</h1>'
                       f'<div class="sub">{html.escape(subtitle)}</div></header>')
        if len(operations) > 0:
            list_items = ""
            for operation in operations:
                list_items = list_items + f"<li>{html.escape(operation_text(operation))}</li>"
            body = f"<b>적용한 연산 {len(operations)}회</b><ol>{list_items}</ol>"
        else:
            body = ('<b class="empty">적용된 연산이 없습니다 — 두 화면이 동일합니다.</b><br>'
                    "규칙 기반 수리기가 이 메쉬에서는 합칠 수 있는 삼각형 쌍을 찾지 못했습니다. "
                    "<code>--best</code> 로 실제 변화가 있는 메쉬를 고를 수 있습니다.")
        footer_html = f"<footer>{body}</footer>"
        table_left = stats_table_html(before_stats, after_stats, "L")
        table_right = stats_table_html(before_stats, after_stats, "R")

    return PAGE_TEMPLATE.format(
        title=f"메쉬 수리 전/후 비교 — {before.name}",
        plotlyjs=get_plotlyjs(),
        figL=json.dumps(figure_left, cls=PlotlyJSONEncoder),
        figR=json.dumps(figure_right, cls=PlotlyJSONEncoder),
        groups=group_index_json(),
        header_html=header_html,
        footer_html=footer_html,
        tableL=table_left,
        tableR=table_right,
        bad=BAD_OUTLINE, line=NEUTRAL_LINE, blocker=BLOCKER, ext=EXT_MARKER,
        chnode=CHANGED_NODE, mvnode=MOVED_NODE,
    )


def build_html(before: EditableMesh, after: EditableMesh, operations: list, path: str, subtitle: str) -> str:
    """``render_html`` 결과를 파일로 쓴다. 돌려주는 값은 파일 경로."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        file_handle.write(render_html(before, after, operations, subtitle))
    return path


# ──────────────────────────────────────────────────────────────────────────
def corrupt(mesh: EditableMesh, split_count: int, random_generator: random.Random):
    """정상 메쉬의 사각형을 무작위로 쪼개 일부러 망가뜨린다. (망가뜨린 사본, 실제로 쪼갠 수)."""
    corrupted = mesh.clone()
    done_count = 0
    for attempt in range(split_count * 6):
        if done_count >= split_count:
            break
        quad_indices = []
        for shell_index in range(len(corrupted.shells)):
            if len(corrupted.shells[shell_index][2]) == 4:
                quad_indices.append(shell_index)
        if len(quad_indices) == 0:
            break
        try:
            corrupted.split_quad(random_generator.choice(quad_indices), diag=random_generator.randint(0, 1))
        except EditError:
            continue
        done_count = done_count + 1
    corrupted.log.clear()
    return corrupted, done_count


def main() -> None:
    parser = argparse.ArgumentParser(description="수리 전/후 2분할 3D 뷰어")
    parser.add_argument("--name", help="볼 메쉬 이름")
    parser.add_argument("--best", type=int, metavar="N",
                        help="수리 연산이 가장 많이 일어난 비정상 메쉬 N개를 자동으로 고른다")
    parser.add_argument("--scan", type=int, default=300, help="--best 가 훑을 메쉬 수")
    parser.add_argument("--corrupt", type=int, metavar="N",
                        help="먼저 사각형 N개를 쪼개 망가뜨린 뒤 수리한다 (자기지도 손상 맛보기)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from bdf_io import discover_pairs
    from labels import load_labels

    gae_paths.ensure_dirs()
    mesh_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    random_generator = random.Random(args.seed)
    config = RepairConfig()

    if args.best:
        labels = load_labels(gae_paths.LABEL_XLSX)
        candidate_names = []
        for name in sorted(labels):
            if name in mesh_paths and labels[name].cls == 1:
                candidate_names.append(name)
        candidate_names = candidate_names[:args.scan]
        print(f"비정상 {len(candidate_names)}개를 훑어 변화가 큰 메쉬를 고르는 중...")
        scored = []
        for name in candidate_names:
            mesh = EditableMesh.load(*mesh_paths[name])
            _, operations = repair(mesh, config)
            if len(operations) > 0:
                scored.append((len(operations), name))
        scored.sort(reverse=True)
        targets = []
        for operation_count, name in scored[:args.best]:
            targets.append(name)
        print(f"  수리가 일어난 메쉬 {len(scored)}개 중 상위 {len(targets)}개 선택\n")
    elif args.name:
        targets = [args.name]
    else:
        parser.error("--name 또는 --best 중 하나가 필요하다")

    for name in targets:
        if name not in mesh_paths:
            print(f"[건너뜀] {name}: 데이터에 없음")
            continue
        base_mesh = EditableMesh.load(*mesh_paths[name])
        note = ""
        if args.corrupt:
            base_mesh, done_count = corrupt(base_mesh, args.corrupt, random_generator)
            note = f"사각형 {done_count}개를 일부러 쪼갠 뒤 수리 · "
        fixed_mesh, operations = repair(base_mesh, config)

        before_stats = stats(base_mesh)
        after_stats = stats(fixed_mesh)
        subtitle = (f"{note}연산 {len(operations)}회 · "
                    f"삼각형 {before_stats['삼각형']} → {after_stats['삼각형']} · "
                    f"불량각 요소 {before_stats['불량각 요소']} → {after_stats['불량각 요소']}")
        suffix = ""
        if args.corrupt:
            suffix = f"_corrupt{args.corrupt}"
        out_path = os.path.join(gae_paths.OUT_DIR, f"compare_{name}{suffix}.html")
        build_html(base_mesh, fixed_mesh, operations, out_path, subtitle)
        print(f"  {name}: 연산 {len(operations)}회, 삼각형 {before_stats['삼각형']}→{after_stats['삼각형']}")
        print(f"    {out_path}")


if __name__ == "__main__":
    main()

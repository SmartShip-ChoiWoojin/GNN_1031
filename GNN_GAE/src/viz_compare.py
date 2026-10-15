"""수리 전/후 메쉬를 카메라 연동된 2분할 3D 뷰어로 내보낸다.

한쪽을 돌리거나 이동하면 반대쪽이 똑같이 따라간다 — 같은 각도에서 봐야 무엇이 바뀌었는지
알 수 있기 때문이다. 색으로도 바로 구분된다.

    왼쪽(수리 전)   빨강 = 이 연산으로 없어질 요소
    오른쪽(수리 후) 초록 = 새로 생긴 요소
    양쪽            회색 = 그대로,  검은 테두리 = 불량각 요소(내각<30° 또는 >150°)
                    파랑 = 1D 요소,  주황 = 삼각형 병합을 막고 있는 1D 요소
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

import gae_paths as gp
import features as F
from meshedit import EditableMesh, EditError, Op, merge_candidates
from meshrep import Edge
from repair_rule import RepairConfig, repair

# app.py 와 같은 팔레트를 쓴다 — 두 뷰어를 오가며 봐도 색 의미가 안 바뀌도록.
NEUTRAL_SHELL = "#c7ced9"
#: 쉘 경계선. 여기서 보고 싶은 건 "요소가 어떻게 나뉘어 있나" 라서 경계가 제일 중요한
#: 선이다. 옅은 회색(#6b7280, 폭 1.5, 불투명도 0.55)으로 뒀더니 비슷한 밝기의 면 위에서
#: 거의 안 보였다 — 진하게 하고 불투명하게 둔다. 불량각 테두리(검정, 폭 5)와는 여전히
#: 굵기·색이 뚜렷이 다르다.
SHELL_EDGE = "#334155"
NEUTRAL_LINE = "#1d4ed8"
BAD_OUTLINE = "#111827"
EXT_MARKER = "#06b6d4"
REMOVED = "#ef4444"   # 수리 전 화면에서 없어질 요소
CHANGED_NODE = "#a21caf"  # 사라지거나 새로 생긴 절점 — 빨강/초록과 섞이지 않는 자주색
MOVED_NODE = "#eab308"    # 자리를 옮긴 절점 — 사라진 것과 구분되는 노랑
ADDED = "#22c55e"     # 수리 후 화면에서 새로 생긴 요소
BLOCKER = "#f97316"   # 변을 없앨 때 같이 지워야 하는 1D 요소

#: 트레이스 순서. 양쪽 화면이 같은 순서를 쓰므로 체크박스 하나로 둘 다 켜고 끌 수 있다.
GROUPS = ["shells", "edges", "bad", "lines", "blockers", "ext", "nodes", "moves"]


# ──────────────────────────────────────────────────────────────────────────
# 트레이스
# ──────────────────────────────────────────────────────────────────────────
def _blocking_line_edges(m: EditableMesh) -> set:
    """인접 삼각형 쌍의 공유 변 위에 올라타 있는 1D 요소의 변.

    예전에는 "병합을 막는" 변으로 표시했는데, 사람이 고친 508쌍을 재보니 그렇지 않다 —
    손댄 구역 안의 이런 변 151개 중 35개(23.2%)를 실제로 없앴고, 그때 그 위의 1D
    요소도 같이 지웠다(35건 중 34건). 금지가 아니라 **같이 처리해야 하는 대상**이다.
    """
    out = set()
    for c in merge_candidates(m, drop_lines=True):
        if c.lines_on_edge > 0:
            out.add(Edge(tuple(set(m.shells[c.si][2]) & set(m.shells[c.sj][2]))))
    return out


def mesh_traces(
    m: EditableMesh,
    mark: set,
    mark_color: str,
    mark_label: str,
    node_marks: dict | None = None,
    node_color: str = CHANGED_NODE,
    node_label: str = "바뀐 절점",
    move_marks: dict | None = None,
) -> list:
    """메쉬 하나를 GROUPS 순서의 트레이스 목록으로 만든다.

    ``mark`` 는 강조할 요소의 절점 집합들 (수리 전이면 없어질 것, 후면 새로 생긴 것).
    비어 있는 트레이스도 반드시 만든다 — 양쪽 화면의 트레이스 인덱스를 맞춰야 한다.
    """
    xs, ys, zs, ti, tj, tk, fc, hover = [], [], [], [], [], [], [], []
    vtx = 0
    for eid, etype, nodes in m.shells:
        P = [m.coords[n] for n in nodes]
        base = vtx
        for p in P:
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        vtx += len(P)
        tris = [(base, base + 1, base + 2)]
        if len(P) == 4:
            tris.append((base, base + 2, base + 3))

        key = frozenset(nodes)
        hit = key in mark
        color = mark_color if hit else NEUTRAL_SHELL
        q = F.shell_quality(np.array(P, dtype=float))
        txt = (
            f"<b>{etype} {eid}</b><br>"
            f"내각 {q.angle_min:.1f}°~{q.angle_max:.1f}°"
            f"{' <b>불량</b>' if F.is_bad_shell(q) else ''}<br>"
            f"aspect {q.aspect:.2f}  skew {q.skew:.1f}°<br>"
            f"절점 {', '.join(str(n) for n in nodes)}"
            + (f"<br><b>{mark_label}</b>" if hit else "")
        )
        for a, b, c in tris:
            ti.append(a); tj.append(b); tk.append(c)
            fc.append(color)
        for _ in P:
            hover.append(txt)

    traces = [
        go.Mesh3d(
            x=xs, y=ys, z=zs, i=ti, j=tj, k=tk, facecolor=fc,
            flatshading=True, opacity=1.0, text=hover, hoverinfo="text", name="쉘 요소",
        )
    ]

    rep = m.rep

    # 쉘 경계선 — 요소 하나하나가 구분돼 보이도록 항상 옅게 긋는다
    sx, sy, sz = [], [], []
    for e in rep.edge_shells:
        a, b = tuple(e)
        pa, pb = m.coords[a], m.coords[b]
        sx += [pa[0], pb[0], None]; sy += [pa[1], pb[1], None]; sz += [pa[2], pb[2], None]
    traces.append(go.Scatter3d(
        x=sx, y=sy, z=sz, mode="lines", line=dict(color=SHELL_EDGE, width=2.5),
        name="쉘 경계", hoverinfo="skip",
    ))

    # 불량각 요소 테두리
    ex, ey, ez = [], [], []
    for _, _, nodes in m.shells:
        P = [m.coords[n] for n in nodes]
        if not F.is_bad_shell(F.shell_quality(np.array(P, dtype=float))):
            continue
        for p in P + [P[0]]:
            ex.append(p[0]); ey.append(p[1]); ez.append(p[2])
        ex.append(None); ey.append(None); ez.append(None)
    traces.append(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines", line=dict(color=BAD_OUTLINE, width=5),
        name="불량각 요소", hoverinfo="skip",
    ))

    # 1D 요소 — 병합을 막는 것과 아닌 것을 나눈다
    blockers = _blocking_line_edges(m)
    for want_block, color, label in (
        (False, NEUTRAL_LINE, "1D 요소 (CBAR/CBEAM)"),
        (True, BLOCKER, "같이 지워야 하는 1D 요소"),
    ):
        lx, ly, lz, lt = [], [], [], []
        for eid, etype, (a, b) in m.lines:
            if (Edge((a, b)) in blockers) != want_block:
                continue
            pa, pb = m.coords[a], m.coords[b]
            lx += [pa[0], pb[0], None]; ly += [pa[1], pb[1], None]; lz += [pa[2], pb[2], None]
            t = f"<b>{etype} {eid}</b><br>절점 {a}-{b}" + (
                "<br><b>양옆 삼각형을 합치려면 이 1D 요소도 같이 지워야 함</b>"
                if want_block else "")
            lt += [t, t, ""]
        traces.append(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="lines",
            line=dict(color=color, width=7 if want_block else 4),
            text=lt, hoverinfo="text", name=label,
        ))

    # ext 절점 — 움직일 수 없는 절단면
    P = np.array([m.coords[n] for n in sorted(m.coords) if m.ext.get(n, False)])
    traces.append(go.Scatter3d(
        x=P[:, 0] if len(P) else [], y=P[:, 1] if len(P) else [],
        z=P[:, 2] if len(P) else [], mode="markers",
        marker=dict(size=4, color=EXT_MARKER, symbol="circle-open", line=dict(width=2)),
        name="ext 절점 (고정)", hoverinfo="skip",
    ))

    # 사라지거나 새로 생긴 절점 — 쌍 비교에서만 쓴다. 안 줘도 트레이스는 만든다
    # (양쪽 화면의 트레이스 번호가 어긋나면 체크박스가 엉뚱한 걸 켜고 끈다).
    nmk = node_marks or {}
    pts = [(n, t) for n, t in nmk.items() if n in m.coords]
    Pn = np.array([m.coords[n] for n, _ in pts]) if pts else np.empty((0, 3))
    traces.append(go.Scatter3d(
        x=Pn[:, 0], y=Pn[:, 1], z=Pn[:, 2], mode="markers",
        marker=dict(size=9, color=node_color, symbol="x",
                    line=dict(width=2, color=node_color)),
        text=[t for _, t in pts], hoverinfo="text", name=node_label,
    ))

    # 자리를 옮긴 절점. 사라진 절점과 색·모양을 달리한다 — 옮겨간 자리가 사라진
    # 절점의 자리와 정확히 겹치는 일이 흔해서(실측 99건), 같은 색이면 좌우 화면에
    # 같은 마커가 찍힌 것처럼 보인다.
    mmk = move_marks or {}
    mp = [(n, t) for n, t in mmk.items() if n in m.coords]
    Pm = np.array([m.coords[n] for n, _ in mp]) if mp else np.empty((0, 3))
    traces.append(go.Scatter3d(
        x=Pm[:, 0], y=Pm[:, 1], z=Pm[:, 2], mode="markers",
        marker=dict(size=8, color=MOVED_NODE, symbol="diamond",
                    line=dict(width=1, color="#854d0e")),
        text=[t for _, t in mp], hoverinfo="text", name="자리를 옮긴 절점",
    ))
    return traces


def _common_ranges(a: EditableMesh, b: EditableMesh) -> dict:
    """두 메쉬를 담는 공통 축 범위. 같은 축 범위라야 두 화면이 픽셀 단위로 비교된다."""
    P = np.array([*a.coords.values(), *b.coords.values()], dtype=float)
    lo, hi = P.min(axis=0), P.max(axis=0)
    pad = 0.05 * np.maximum(hi - lo, 1.0)
    lo, hi = lo - pad, hi + pad
    return {
        ax: [float(lo[i]), float(hi[i])] for i, ax in enumerate(("xaxis", "yaxis", "zaxis"))
    }


def _layout(title: str, ranges: dict) -> go.Layout:
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
def stats(m: EditableMesh) -> dict:
    Q = [F.shell_quality(m.corners(i)) for i in range(len(m.shells))]
    n_tri = sum(1 for q in Q if not q.is_quad)
    return {
        "쉘 요소": len(Q),
        "삼각형": n_tri,
        "삼각형 비율": f"{n_tri / max(1, len(Q)):.3f}",
        "불량각 요소": sum(1 for q in Q if F.is_bad_shell(q)),
        "최대 skew": f"{max((q.skew for q in Q), default=0):.1f}°",
        "최대 aspect": f"{max((q.aspect for q in Q), default=0):.2f}",
    }


def _op_text(op: Op) -> str:
    if op.kind == "merge":
        return f"삼각형 2개 → 사각형 1개 (절점 {', '.join(str(k) for k in op.key)})"
    if op.kind == "swap":
        return f"대각선 뒤집기 (절점 {', '.join(str(k) for k in op.key)})"
    if op.kind == "split":
        return f"사각형 → 삼각형 2개 (절점 {', '.join(str(k) for k in op.key)})"
    return f"절점 {op.key[0]} 이동"


# ──────────────────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────────────────
_PAGE = """<!doctype html>
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
  // plotly 의 'plotly_relayout' 이벤트에 걸지 않는다. 3D 씬에서는 이 이벤트가
  // 드래그 때 뜨기도 하고 안 뜨기도 해서(plotly 6.5 에서 실측) 놓치는 경우가 있다.
  // 대신 매 프레임 두 씬의 카메라를 직접 비교해서, 바뀐 쪽을 반대쪽에 밀어넣는다.
  // 버전에 안 휘둘리고, 드래그가 끝나기 전에도 실시간으로 따라온다.
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

  // src 의 카메라를 dst 에 복사한다.
  //
  // 기준값(lastL/lastR)을 **보낸 카메라 그 자체**로 맞추는 게 핵심이다. 예전에는 복사가
  // 끝난 뒤 양쪽의 '현재' 카메라를 다시 읽어서 기준으로 삼았는데, 그러면 relayout 이
  // 도는 동안 사용자가 계속 끌어서 생긴 움직임이 기준값에 그대로 흡수돼 버린다. 다음
  // 틱에서 '안 바뀌었다'고 판정하고 넘어가므로 그 마지막 움직임만큼 반대쪽이 영구히
  // 뒤처진다 — 드래그를 멈추면 좌우 각도가 다른 채로 굳는 증상이 이것이었다.
  // 보낸 값으로 맞춰두면, 그 사이 더 움직인 만큼은 다음 틱에서 다시 잡혀 따라붙는다.
  function push(src, dst, camera) {{
    busy = true;
    // 원본 객체를 그대로 넘기면 plotly 가 제자리에서 고쳐 쓰며 서로 얽힌다
    const snap = JSON.parse(JSON.stringify(camera));
    const settle = () => {{ lastL = lastR = key(snap); busy = false; }};
    // 실패해도 busy 를 반드시 풀어준다 — 한 번 걸리면 연동이 통째로 멈춘다
    Plotly.relayout(dst, {{'scene.camera': snap}}).then(settle, settle);
  }}
  // 평소에는 rAF 로 화면 갱신에 맞춰 돌린다.
  (function loop() {{ tick(); requestAnimationFrame(loop); }})();
  // 창이 가려지거나 백그라운드로 가면 브라우저가 rAF 를 아예 멈춘다 — 그때도
  // 연동이 끊기지 않도록 타이머로 한 번 더 받쳐준다. busy 가 중복 실행을 막는다.
  setInterval(tick, 150);
  // 연동이 안 될 때 콘솔에서 바로 확인할 수 있는 표식
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
    // projection 까지 넣어야 key() 가 relayout 뒤의 카메라와 똑같이 나온다
    const eye = {{eye:{{x:1.45,y:1.45,z:1.0}}, center:{{x:0,y:0,z:0}},
                 up:{{x:0,y:0,z:1}}, projection:{{type:'perspective'}}}};
    const settle = () => {{ lastL = lastR = key(eye); busy = false; }};
    Promise.all([Plotly.relayout(L, {{'scene.camera': eye}}),
                 Plotly.relayout(R, {{'scene.camera': eye}})]).then(settle, settle);
  }});
}});
</script>
"""


def _table(before: dict, after: dict, side: str) -> str:
    rows = []
    for k in before:
        b, a = before[k], after[k]
        if side == "L":
            rows.append(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(b))}</td></tr>")
        else:
            cls = ""
            if k in ("삼각형", "불량각 요소") and isinstance(b, int) and isinstance(a, int):
                cls = ' class="chg"' if a < b else (' class="neg"' if a > b else "")
            rows.append(
                f"<tr><td>{html.escape(k)}</td><td{cls}>{html.escape(str(a))}</td></tr>"
            )
    head = "수리 전" if side == "L" else "수리 후"
    return f"<table><tr><th>지표</th><th>{head}</th></tr>{''.join(rows)}</table>"


def render_html(
    before: EditableMesh,
    after: EditableMesh,
    ops: list[Op],
    subtitle: str,
    embed: bool = False,
    node_marks_before: dict | None = None,
    node_marks_after: dict | None = None,
    move_marks_before: dict | None = None,
    move_marks_after: dict | None = None,
) -> str:
    """수리 전/후 2분할 뷰어 HTML 을 문자열로 만든다.

    ``embed=True`` 면 제목·통계표·연산목록을 빼고 **3D 화면과 도구막대만** 남긴다.
    Streamlit 앱(`app.py`)이 그 부분을 자기 위젯으로 더 보기 좋게 그리기 때문이다.
    카메라 연동 스크립트는 iframe 안에서 돌아야 하므로 어느 쪽이든 그대로 들어간다.
    """
    bkeys = {frozenset(s[2]) for s in before.shells}
    akeys = {frozenset(s[2]) for s in after.shells}
    removed, added = bkeys - akeys, akeys - bkeys

    # 임베드할 때는 메쉬 이름을 Streamlit 쪽에서 이미 보여주므로 화면 제목을 짧게 둔다.
    # 긴 제목을 그대로 두면 좁아진 두 화면 사이에서 글자가 서로 맞닿는다.
    tL, tR = ("수리 전", "수리 후") if embed else (
        f"수리 전 — {before.name}", f"수리 후 — {after.name}")

    ranges = _common_ranges(before, after)
    figL = go.Figure(
        data=mesh_traces(before, removed, REMOVED, "이 연산으로 없어짐",
                         node_marks=node_marks_before, node_label="사라지는 절점",
                         move_marks=move_marks_before),
        layout=_layout(tL, ranges),
    )
    figR = go.Figure(
        data=mesh_traces(after, added, ADDED, "새로 생긴 요소",
                         node_marks=node_marks_after, node_label="새로 생긴 절점",
                         move_marks=move_marks_after),
        layout=_layout(tR, ranges),
    )

    sb, sa = stats(before), stats(after)
    if embed:
        # iframe 높이가 곧 화면 높이다 — 도구막대를 뺀 나머지를 3D 가 다 쓰게 한다.
        # (기본 min(68vh,760px) 를 그대로 두면 아래가 허옇게 남는다)
        # 이 문자열은 .format() 의 인자로 들어가므로 중괄호를 이스케이프하지 않는다
        header_html = "<style>.plot { height:calc(100vh - 62px); }</style>"
        footer_html = ""
        tableL = tableR = ""
    else:
        header_html = (
            f'<header><h1>메쉬 수리 전/후 비교 — {html.escape(before.name)}</h1>'
            f'<div class="sub">{html.escape(subtitle)}</div></header>'
        )
        if ops:
            items = "".join(f"<li>{html.escape(_op_text(o))}</li>" for o in ops)
            body = f"<b>적용한 연산 {len(ops)}회</b><ol>{items}</ol>"
        else:
            body = (
                '<b class="empty">적용된 연산이 없습니다 — 두 화면이 동일합니다.</b><br>'
                "규칙 기반 수리기가 이 메쉬에서는 합칠 수 있는 삼각형 쌍을 찾지 못했습니다. "
                "<code>--best</code> 로 실제 변화가 있는 메쉬를 고를 수 있습니다."
            )
        footer_html = f"<footer>{body}</footer>"
        tableL, tableR = _table(sb, sa, "L"), _table(sb, sa, "R")

    return _PAGE.format(
        title=f"메쉬 수리 전/후 비교 — {before.name}",
        plotlyjs=get_plotlyjs(),
        figL=json.dumps(figL, cls=PlotlyJSONEncoder),
        figR=json.dumps(figR, cls=PlotlyJSONEncoder),
        groups=json.dumps({g: i for i, g in enumerate(GROUPS)}),
        header_html=header_html,
        footer_html=footer_html,
        tableL=tableL,
        tableR=tableR,
        bad=BAD_OUTLINE, line=NEUTRAL_LINE, blocker=BLOCKER, ext=EXT_MARKER,
        chnode=CHANGED_NODE, mvnode=MOVED_NODE,
    )


def build_html(
    before: EditableMesh, after: EditableMesh, ops: list[Op], path: str, subtitle: str
) -> str:
    """``render_html`` 결과를 파일로 쓴다. 돌려주는 값은 파일 경로."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(before, after, ops, subtitle))
    return path


# ──────────────────────────────────────────────────────────────────────────
def corrupt(m: EditableMesh, n: int, rng: random.Random) -> tuple[EditableMesh, int]:
    """정상 메쉬의 사각형을 무작위로 쪼개 일부러 망가뜨린다 (Stage 1 손상 생성의 맛보기)."""
    out = m.clone()
    done = 0
    for _ in range(n * 6):
        if done >= n:
            break
        quads = [i for i, s in enumerate(out.shells) if len(s[2]) == 4]
        if not quads:
            break
        try:
            out.split_quad(rng.choice(quads), diag=rng.randint(0, 1))
        except EditError:
            continue
        done += 1
    out.log.clear()
    return out, done


def main() -> None:
    ap = argparse.ArgumentParser(description="수리 전/후 2분할 3D 뷰어")
    ap.add_argument("--name", help="볼 메쉬 이름")
    ap.add_argument("--best", type=int, metavar="N",
                    help="수리 연산이 가장 많이 일어난 비정상 메쉬 N개를 자동으로 고른다")
    ap.add_argument("--scan", type=int, default=300, help="--best 가 훑을 메쉬 수")
    ap.add_argument("--corrupt", type=int, metavar="N",
                    help="먼저 사각형 N개를 쪼개 망가뜨린 뒤 수리한다 (자기지도 손상 맛보기)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bdf_io import discover_pairs
    from labels import load_labels

    gp.ensure_dirs()
    pairs, _, _ = discover_pairs(gp.FE_DIR)
    rng = random.Random(args.seed)
    cfg = RepairConfig()

    if args.best:
        lab = load_labels(gp.LABEL_XLSX)
        cands = [n for n in sorted(lab) if n in pairs and lab[n].cls == 1][: args.scan]
        print(f"비정상 {len(cands)}개를 훑어 변화가 큰 메쉬를 고르는 중...")
        scored = []
        for nm in cands:
            m = EditableMesh.load(*pairs[nm])
            _, ops = repair(m, cfg)
            if ops:
                scored.append((len(ops), nm))
        scored.sort(reverse=True)
        targets = [nm for _, nm in scored[: args.best]]
        print(f"  수리가 일어난 메쉬 {len(scored)}개 중 상위 {len(targets)}개 선택\n")
    elif args.name:
        targets = [args.name]
    else:
        ap.error("--name 또는 --best 중 하나가 필요하다")

    for nm in targets:
        if nm not in pairs:
            print(f"[건너뜀] {nm}: 데이터에 없음")
            continue
        base = EditableMesh.load(*pairs[nm])
        note = ""
        if args.corrupt:
            base, done = corrupt(base, args.corrupt, rng)
            note = f"사각형 {done}개를 일부러 쪼갠 뒤 수리 · "
        fixed, ops = repair(base, cfg)

        sb, sa = stats(base), stats(fixed)
        sub = (
            f"{note}연산 {len(ops)}회 · "
            f"삼각형 {sb['삼각형']} → {sa['삼각형']} · "
            f"불량각 요소 {sb['불량각 요소']} → {sa['불량각 요소']}"
        )
        suffix = f"_corrupt{args.corrupt}" if args.corrupt else ""
        out = os.path.join(gp.OUT_DIR, f"compare_{nm}{suffix}.html")
        build_html(base, fixed, ops, out, sub)
        print(f"  {nm}: 연산 {len(ops)}회, 삼각형 {sb['삼각형']}→{sa['삼각형']}")
        print(f"    {out}")


if __name__ == "__main__":
    main()

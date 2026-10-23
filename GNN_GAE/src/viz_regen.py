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

색 규칙은 세 화면에서 하나다:
    빨강   모델과 사람이 둘 다 지운 element       (왼쪽, '비교' 보기)
    주황   **모델만** 삭제·변형한 element (왼쪽)
    보라   **사람만** 삭제·변형한 element (왼쪽)
    초록   새로 만든 element (오른쪽)
    (node 삭제·이동은 3D 위에 따로 표시하지 않는다 — 화면 아래 변경 목록과 '변경 위치로' 에 있다)

왼쪽 원본은 "원본에 표시할 삭제" 로 보기를 바꾼다:
    비교   누가 지웠는지 색으로 (빨강/주황/보라)
    모델   모델이 지운 것만 빨강
    사람   사람이 지운 것만 빨강

강조를 끄면 그 element 는 회색으로 돌아간다. 같은 자리에 면을 두 장 겹치면 깜빡이므로
(z-fighting), 강조할 element 는 기본 면에서 빼고 "색칠한 것 / 회색 사본" 을 따로 두어 하나만
보이게 한다. 선(테두리·1D)은 겹쳐 그려도 되므로 위에 덧그린다.

"변경 위치로" 목록에서 고르면 세 화면이 그 자리로 확대된다 (축 범위를 좁히는 방식 —
카메라 계산 없이 aspectmode='data' 에서 그대로 맞는다).
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

import features as F
from meshdiff import MeshDiff, diff, pkey
from viz_compare import (BAD_OUTLINE, EXT_MARKER, NEUTRAL_LINE, NEUTRAL_SHELL,
                         SHELL_EDGE, stats)

CREATED = "#22c55e"          # 새로 만든 element
CREATED_EDGE = "#15803d"
FOCUS = "#2563eb"            # '변경 위치로' 표시

#: 누가 바꿨나 — 면 색 / 선·X 색
WHO_FILL = {"both": "#ef4444", "model": "#fb923c", "human": "#a78bfa"}
WHO_LINE = {"both": "#b91c1c", "model": "#ea580c", "human": "#7c3aed"}
WHO_TEXT = {"both": "모델·사람 둘 다", "model": "모델만", "human": "사람만"}
#: '모델' / '사람' 보기에서 그쪽이 지운 element 의 색
SIDE_FILL, SIDE_LINE = "#ef4444", "#b91c1c"

#: 원본에 표시할 삭제 보기
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
@dataclass
class OrigSpec:
    """왼쪽 원본 화면. 지워진 element / 1D 를 '누가 지웠나' 와 함께 갖는다."""

    mesh: object
    #: 쉘 인덱스 -> "both" | "model" | "human"
    rm_shell: dict[int, str] = field(default_factory=dict)
    #: 1D 인덱스 -> 누가
    rm_line: dict[int, str] = field(default_factory=dict)
    title: str = "원본 (비정상 메쉬)"


@dataclass
class ResultSpec:
    """오른쪽 결과 화면 (모델 또는 사람). 지우기·만들기가 반영된 최종 메쉬."""

    mesh: object
    who: str                     # "model" | "human"
    title: str
    #: 새로 만든 쉘 인덱스 -> 설명
    created: dict[int, str] = field(default_factory=dict)
    #: 새로 생긴 1D 인덱스 -> 설명
    created_lines: dict[int, str] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# 트레이스 부품
# ──────────────────────────────────────────────────────────────────────────
def _hover(m, Q, i, extra: str = "") -> str:
    eid, etype, nodes = m.shells[i]
    q = Q[i]
    return (f"<b>{etype} {eid}</b><br>내각 {q.angle_min:.1f}°~{q.angle_max:.1f}°"
            f"{' <b>불량</b>' if F.is_bad_shell(q) else ''}<br>"
            f"aspect {q.aspect:.2f}  skew {q.skew:.1f}°<br>"
            f"node {', '.join(map(str, nodes))}" + (f"<br><b>{extra}</b>" if extra else ""))


def _mesh3d(m, idxs, colors, hover, meta, visible=True):
    """쉘 묶음 하나를 Mesh3d 로. ``colors`` 는 쉘마다 색 (문자열 하나면 전부 그 색)."""
    xs, ys, zs, ti, tj, tk, fc, txt = [], [], [], [], [], [], [], []
    v = 0
    for i in idxs:
        nodes = m.shells[i][2]
        c = colors if isinstance(colors, str) else colors[i]
        for n in nodes:
            p = m.coords[n]
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
            txt.append(hover[i])
        ti.append(v); tj.append(v + 1); tk.append(v + 2); fc.append(c)
        if len(nodes) == 4:
            ti.append(v); tj.append(v + 2); tk.append(v + 3); fc.append(c)
        v += len(nodes)
    return go.Mesh3d(x=xs, y=ys, z=zs, i=ti, j=tj, k=tk, facecolor=fc, flatshading=True,
                     opacity=1.0, text=txt, hoverinfo="text", meta=meta, visible=visible)


def _outline(m, idxs, color, width, meta, visible=True):
    xs, ys, zs = [], [], []
    for i in idxs:
        P = [m.coords[n] for n in m.shells[i][2]]
        for p in P + [P[0]]:
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        xs.append(None); ys.append(None); zs.append(None)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width),
                        hoverinfo="skip", meta=meta, visible=visible)


def _lines(m, items, color, width, meta, visible=True):
    """``items`` = [(1D 인덱스, 설명)]."""
    lx, ly, lz, lt = [], [], [], []
    for li, why in items:
        eid, etype, (a, b) = m.lines[li]
        pa, pb = m.coords[a], m.coords[b]
        lx += [pa[0], pb[0], None]; ly += [pa[1], pb[1], None]; lz += [pa[2], pb[2], None]
        t = f"<b>{etype} {eid}</b><br>node {a}-{b}" + (f"<br><b>{why}</b>" if why else "")
        lt += [t, t, ""]
    return go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color=color, width=width),
                        text=lt, hoverinfo="text", meta=meta, visible=visible)


def _points(P, texts, marker, meta, visible=True):
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    return go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", marker=marker,
                        text=texts, hoverinfo="text" if texts else "skip", meta=meta,
                        visible=visible)


def _common(m, Q) -> list:
    """세 화면 공통 — 쉘 경계, 불량각 테두리, 1D element, ext node, '변경 위치로' 표시."""
    rep = m.rep
    sx, sy, sz = [], [], []
    for e in rep.edge_shells:
        a, b = tuple(e)
        pa, pb = m.coords[a], m.coords[b]
        sx += [pa[0], pb[0], None]; sy += [pa[1], pb[1], None]; sz += [pa[2], pb[2], None]
    out = [
        go.Scatter3d(x=sx, y=sy, z=sz, mode="lines", line=dict(color=SHELL_EDGE, width=2.5),
                     hoverinfo="skip", meta="edges"),
        _outline(m, [i for i, q in enumerate(Q) if F.is_bad_shell(q)], BAD_OUTLINE, 5, "bad",
                 visible=False),
        _lines(m, [(li, "") for li in range(len(m.lines))], NEUTRAL_LINE, 4, "lines"),
        _points([m.coords[n] for n in sorted(m.coords) if m.ext.get(n, False)], None,
                dict(size=4, color=EXT_MARKER, symbol="circle-open", line=dict(width=2)),
                "ext", visible=False),
        go.Scatter3d(x=[], y=[], z=[], mode="markers",
                     marker=dict(size=34, color=FOCUS, opacity=0.25, symbol="circle"),
                     hoverinfo="skip", meta="focus"),
    ]
    return out


# ──────────────────────────────────────────────────────────────────────────
# 화면별 트레이스
# ──────────────────────────────────────────────────────────────────────────
def orig_traces(s: OrigSpec) -> list:
    """왼쪽 원본. 지워진 element 는 보기(비교/모델/사람)마다 따로 칠한 트레이스를 둔다."""
    m = s.mesh
    Q = [F.shell_quality(m.corners(i)) for i in range(len(m.shells))]
    union = sorted(s.rm_shell)
    hov = {i: _hover(m, Q, i, (f"{WHO_TEXT[s.rm_shell[i]]} 지움" if i in s.rm_shell else ""))
           for i in range(len(m.shells))}
    tr = [_mesh3d(m, [i for i in range(len(m.shells)) if i not in s.rm_shell], NEUTRAL_SHELL,
                  hov, "base"),
          _mesh3d(m, union, NEUTRAL_SHELL, hov, "rm:gray", visible=False)]

    def mine(who: str, side: str) -> bool:
        return who == "both" or who == side

    for mode, _ in MODES:
        if mode == "cmp":
            colors = {i: WHO_FILL[w] for i, w in s.rm_shell.items()}
            out_idx = {w: [i for i in union if s.rm_shell[i] == w] for w in WHO_FILL}
        else:
            colors = {i: (SIDE_FILL if mine(w, mode) else NEUTRAL_SHELL) for i, w in s.rm_shell.items()}
            out_idx = {"side": [i for i in union if mine(s.rm_shell[i], mode)]}
        vis = mode == "cmp"
        tr.append(_mesh3d(m, union, colors, hov, f"rm:{mode}:fill", visible=vis))
        for w, idx in out_idx.items():
            col = SIDE_LINE if w == "side" else WHO_LINE[w]
            tr.append(_outline(m, idx, col, 5, f"rm:{mode}:fill", visible=vis))

        # 1D
        if mode == "cmp":
            for w in WHO_LINE:
                items = [(li, f"{WHO_TEXT[w]} 지운 1D") for li, ww in s.rm_line.items() if ww == w]
                tr.append(_lines(m, items, WHO_LINE[w], 10, f"rm:{mode}:line1d", visible=vis))
        else:
            items = [(li, f"{WHO_TEXT[w]} 지운 1D") for li, w in s.rm_line.items() if mine(w, mode)]
            tr.append(_lines(m, items, SIDE_LINE, 10, f"rm:{mode}:line1d", visible=vis))

    return tr + _common(m, Q)


def result_traces(s: ResultSpec) -> list:
    """오른쪽 결과 화면. 새로 만든 element 를 초록으로 칠한다."""
    m = s.mesh
    Q = [F.shell_quality(m.corners(i)) for i in range(len(m.shells))]
    hov = {i: _hover(m, Q, i, s.created.get(i, "")) for i in range(len(m.shells))}
    made = sorted(s.created)
    tr = [_mesh3d(m, [i for i in range(len(m.shells)) if i not in s.created], NEUTRAL_SHELL,
                  hov, "base"),
          _mesh3d(m, made, CREATED, hov, "cr:fill"),
          _outline(m, made, CREATED_EDGE, 5, "cr:fill"),
          _mesh3d(m, made, NEUTRAL_SHELL, hov, "cr:gray", visible=False),
          _lines(m, list(s.created_lines.items()), CREATED_EDGE, 10, "cr:line1d")]
    return tr + _common(m, Q)


def _ranges(meshes) -> dict:
    P = np.array([c for m in meshes for c in m.coords.values()], dtype=float)
    lo, hi = P.min(0), P.max(0)
    pad = 0.05 * np.maximum(hi - lo, 1.0)
    lo, hi = lo - pad, hi + pad
    return {ax: [float(lo[i]), float(hi[i])] for i, ax in enumerate(("xaxis", "yaxis", "zaxis"))}


def _layout(title: str, ranges: dict) -> go.Layout:
    return go.Layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        scene=dict(aspectmode="data",
                   **{ax: dict(range=r, showticklabels=False, title="", autorange=False)
                      for ax, r in ranges.items()}),
        margin=dict(l=0, r=0, t=30, b=0), showlegend=False, hoverlabel=dict(align="left"),
    )


# ──────────────────────────────────────────────────────────────────────────
# 비교 — 원본 / 모델 / 사람 셋으로 세 화면의 강조와 변경 목록을 정한다
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Change:
    """변경 목록의 한 줄 — 표와 '변경 위치로' 가 같은 목록을 쓴다."""

    who: str          # "모델" | "사람"
    what: str         # "element 삭제" | "element 생성" | "node 삭제" | "node 이동" | "1D 삭제" | "1D 생성"
    target: str       # element / node 번호
    shared: bool      # 상대 쪽도 같은 변경을 했나 (좌표 기준)
    note: str
    at: tuple         # 확대할 중심 좌표


def _skey(m, i) -> frozenset:
    return frozenset(pkey(m.coords[n]) for n in m.shells[i][2])


def _lkey(m, i) -> frozenset:
    return frozenset(pkey(m.coords[n]) for n in m.lines[i][2])


def _who(in_model: bool, in_human: bool) -> str:
    return "both" if in_model and in_human else ("model" if in_model else "human")


def build_view(bad, model, human, model_absorbed: dict | None = None,
               human_absorbed: dict | None = None) -> tuple[list, list[Change], dict]:
    """세 화면 명세 [원본, 모델 결과, 사람 결과] + 변경 목록 + 요약."""
    dm: MeshDiff = diff(bad, model, model_absorbed)
    dh: MeshDiff = diff(bad, human, human_absorbed)
    rm_m = {_skey(bad, i) for i in dm.removed_shells}
    rm_h = {_skey(bad, i) for i in dh.removed_shells}
    cr_m = {_skey(model, i) for i in dm.created_shells}
    cr_h = {_skey(human, i) for i in dh.created_shells}
    lrm_m = {_lkey(bad, i) for i in dm.removed_lines}
    lrm_h = {_lkey(bad, i) for i in dh.removed_lines}
    lcr_m = {_lkey(model, i) for i in dm.created_lines}
    lcr_h = {_lkey(human, i) for i in dh.created_lines}
    # 사라진 node 자리 — 옮긴 node 의 출발점도 그 자리가 없어진 것이다
    del_m = {pkey(bad.coords[n]) for n in dm.deleted_nodes} | {pkey(p0) for p0, _, _ in dm.moved_nodes.values()}
    del_h = {pkey(bad.coords[n]) for n in dh.deleted_nodes} | {pkey(p0) for p0, _, _ in dh.moved_nodes.values()}

    orig = OrigSpec(bad)
    for i in set(dm.removed_shells) | set(dh.removed_shells):
        k = _skey(bad, i)
        orig.rm_shell[i] = _who(k in rm_m, k in rm_h)
    for li in set(dm.removed_lines) | set(dh.removed_lines):
        k = _lkey(bad, li)
        orig.rm_line[li] = _who(k in lrm_m, k in lrm_h)

    def tag(both: bool, other: str) -> str:
        return f"{other}도 같은 변경 ✓" if both else f"{other}는 안 한 변경"

    res_m = ResultSpec(model, "model", "모델 재생성 결과",
                       created={i: f"모델이 만듦 · {tag(_skey(model, i) in cr_h, '사람')}"
                                for i in dm.created_shells},
                       created_lines={i: f"모델이 만든 1D · {tag(_lkey(model, i) in lcr_h, '사람')}"
                                      for i in dm.created_lines})
    res_h = ResultSpec(human, "human", "사람 수정 결과",
                       created={i: f"사람이 만듦 · {tag(_skey(human, i) in cr_m, '모델')}"
                                for i in dh.created_shells},
                       created_lines={i: f"사람이 만든 1D · {tag(_lkey(human, i) in lcr_m, '모델')}"
                                      for i in dh.created_lines})

    # ── 변경 목록 ──────────────────────────────────────────────────────
    ch: list[Change] = []
    cen = lambda m, i: tuple(np.mean([m.coords[n] for n in m.shells[i][2]], axis=0))
    lcen = lambda m, i: tuple(np.mean([m.coords[n] for n in m.lines[i][2]], axis=0))
    for who, d, res, rm_o, cr_o, lrm_o, lcr_o, del_o in (
            ("모델", dm, model, rm_h, cr_h, lrm_h, lcr_h, del_h),
            ("사람", dh, human, rm_m, cr_m, lrm_m, lcr_m, del_m)):
        for i in d.removed_shells:
            eid, t, ns = bad.shells[i]
            ch.append(Change(who, "element 삭제", f"{t} {eid}", _skey(bad, i) in rm_o,
                             f"node {', '.join(map(str, ns))}", cen(bad, i)))
        for i in d.created_shells:
            eid, t, ns = res.shells[i]
            ch.append(Change(who, "element 생성", f"{t} {eid}", _skey(res, i) in cr_o,
                             f"node {', '.join(map(str, ns))}", cen(res, i)))
        for n, why in d.deleted_nodes.items():
            ch.append(Change(who, "node 삭제", f"GRID {n}", pkey(bad.coords[n]) in del_o, why,
                             tuple(bad.coords[n])))
        for n, (p0, p1, why) in d.moved_nodes.items():
            ch.append(Change(who, "node 이동", f"GRID {n}", pkey(p0) in del_o, why,
                             tuple((np.asarray(p0) + np.asarray(p1)) / 2)))
        for i in d.removed_lines:
            eid, t, (a, b) = bad.lines[i]
            ch.append(Change(who, "1D 삭제", f"{t} {eid}", _lkey(bad, i) in lrm_o, f"node {a}-{b}",
                             lcen(bad, i)))
        for i in d.created_lines:
            eid, t, (a, b) = res.lines[i]
            ch.append(Change(who, "1D 생성", f"{t} {eid}", _lkey(res, i) in lcr_o, f"node {a}-{b}",
                             lcen(res, i)))

    summary = {
        "model": {"지운 element": len(dm.removed_shells), "만든 element": len(dm.created_shells),
                  "사라진 node": len(dm.deleted_nodes), "옮긴 node": len(dm.moved_nodes),
                  "1D 삭제": len(dm.removed_lines), "1D 생성": len(dm.created_lines)},
        "human": {"지운 element": len(dh.removed_shells), "만든 element": len(dh.created_shells),
                  "사라진 node": len(dh.deleted_nodes), "옮긴 node": len(dh.moved_nodes),
                  "1D 삭제": len(dh.removed_lines), "1D 생성": len(dh.created_lines)},
        "stats": {"원본": stats(bad), "모델": stats(model), "사람": stats(human)},
    }
    return [orig, res_m, res_h], ch, summary


# ──────────────────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────────────────
_PAGE = """<!doctype html>
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

  // ── 카메라 연동 ────────────────────────────────────────────────────
  // plotly_relayout 이벤트는 3D 드래그 때 놓치는 경우가 있어(6.5 실측) 매 프레임 비교한다.
  // 기준값은 '보낸 카메라 그 자체' 로 맞춘다 — 복사 뒤 현재값을 읽으면 그 사이 더 끈
  // 만큼이 흡수돼 반대쪽이 영구히 뒤처진다.
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
  // 트레이스마다 meta 에 묶음 이름이 있다. 상태 하나에서 모든 트레이스의 보이기를 다시 계산한다
  // (보기 전환·체크박스가 서로 얽혀 있어서, 하나씩 켜고 끄면 조합이 어긋난다).
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


def render_html(panes: list, changes: list[Change], focus_radius: float) -> str:
    """세 화면 HTML. ``panes`` = [OrigSpec, ResultSpec(모델), ResultSpec(사람)].

    ``focus_radius`` 는 '변경 위치로' 확대할 때 상자 반폭(mm).
    """
    orig, rm, rh = panes
    ranges = _ranges([orig.mesh, rm.mesh, rh.mesh])
    figs = [go.Figure(data=orig_traces(orig), layout=_layout(orig.title, ranges)),
            go.Figure(data=result_traces(rm), layout=_layout(rm.title, ranges)),
            go.Figure(data=result_traces(rh), layout=_layout(rh.title, ranges))]
    modes = "".join(
        f'<label><input type="radio" name="mode" value="{v}"{" checked" if v == "cmp" else ""}> '
        f'{html.escape(lbl)}</label>' for v, lbl in MODES)
    legend = "".join(
        f'<span><span class="key" style="background:{c}"></span>{html.escape(t)}</span>'
        for c, t in ((WHO_FILL["both"], "둘 다"), (WHO_FILL["model"], "모델만"),
                     (WHO_FILL["human"], "사람만"), (CREATED, "새로 만든 element")))
    toggles = "".join(
        f'<label><input type="checkbox" data-g="{g}"{" checked" if on else ""}>'
        f'<span class="key" style="background:{c}"></span>{html.escape(lbl)}</label>'
        for g, lbl, c, on in TOGGLES)
    opts = "".join(
        f'<option value="{i}">{html.escape(c.who)} · {html.escape(c.what)} · {html.escape(c.target)}'
        f'{"" if c.shared else "  (상대 쪽에 없음)"}</option>'
        for i, c in enumerate(changes))
    focus = [{"p": [float(v) for v in c.at], "r": float(focus_radius)} for c in changes]
    return _PAGE.format(
        plotlyjs=get_plotlyjs(),
        figs=json.dumps([f.to_plotly_json() for f in figs], cls=PlotlyJSONEncoder),
        focus=json.dumps(focus), ranges=json.dumps(ranges),
        focus_opts=opts, modes=modes, legend=legend, toggles=toggles,
    )

"""재생성 결과 뷰어 — 원본 → 모델 / 원본 → 사람 을 2×2 로, 네 화면 시점 연동.

    ┌──────────────────────┬──────────────────────┐
    │ 원본 — 모델이 지운 것 │ 모델 재생성 — 만든 것 │
    ├──────────────────────┼──────────────────────┤
    │ 원본 — 사람이 지운 것 │ 사람 수정 — 만든 것   │
    └──────────────────────┴──────────────────────┘

강조 (도구막대에서 하나씩 끄고 켤 수 있다):
    빨강 면/테두리    지운 요소              (왼쪽 화면)
    초록 면/테두리    새로 만든 요소         (오른쪽 화면)
    자주 X           사라진 절점 자리       (왼쪽 화면)
    노랑 마름모+화살표 자리를 옮긴 절점 (출발 → 도착)
    굵은 빨강/초록 선  지운 1D / 새로 생긴 1D
    주황 굵은 테두리   **상대 쪽에는 없는** 변경 — 모델만 한 것, 사람만 한 것

강조를 끄면 그 요소는 회색으로 돌아간다(색만 빠지고 형상은 그대로). 면 색을 트레이스
하나에 섞지 않고 강조 묶음마다 트레이스를 따로 두는 이유가 이것이다 — 같은 자리에 면을
두 장 겹치면 깜빡이므로(z-fighting) "색칠한 것 / 회색 사본" 두 장을 두고 하나만 보인다.

"변경 위치로" 목록에서 고르면 네 화면이 그 자리로 확대된다 (축 범위를 좁히는 방식 —
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
from viz_compare import (BAD_OUTLINE, CHANGED_NODE, EXT_MARKER, MOVED_NODE, NEUTRAL_LINE,
                         NEUTRAL_SHELL, SHELL_EDGE, stats)

DELETED = "#ef4444"      # 지운 요소
CREATED = "#22c55e"      # 만든 요소
ONLY_ONE = "#f97316"     # 상대 쪽에 없는 변경 (테두리)
FOCUS = "#2563eb"        # '변경 위치로' 표시

#: 도구막대 체크박스 → 강조 묶음. 값이 False 면 처음에 꺼진 채로 시작한다.
TOGGLES = [
    ("fill_rm", "지운 요소", DELETED, True),
    ("fill_cr", "만든 요소", CREATED, True),
    ("delnode", "사라진 절점", CHANGED_NODE, True),
    ("moved", "옮긴 절점", MOVED_NODE, True),
    ("line1d", "1D 변경", DELETED, True),
    ("diff", "상대 쪽에 없는 변경", ONLY_ONE, True),
    ("edges", "쉘 경계", SHELL_EDGE, True),
    ("bad", "불량각 요소", BAD_OUTLINE, False),
    ("lines", "1D 요소", NEUTRAL_LINE, True),
    ("ext", "ext 절점", EXT_MARKER, False),
]


# ──────────────────────────────────────────────────────────────────────────
# 무엇을 강조할지 — 한 화면 분량
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PaneSpec:
    mesh: object
    title: str
    #: 강조할 쉘 인덱스 -> 설명 (지운 것이면 'fill_rm', 만든 것이면 'fill_cr' 묶음)
    fill: dict[int, str] = field(default_factory=dict)
    fill_group: str = "fill_rm"
    #: 상대 쪽에는 없는 변경 — 주황 테두리를 두를 쉘 인덱스
    only: set = field(default_factory=set)
    #: 강조할 1D 인덱스 -> 설명
    lines: dict[int, str] = field(default_factory=dict)
    line_color: str = DELETED
    #: 사라진 절점 자리 (id -> 설명)
    delnodes: dict[int, str] = field(default_factory=dict)
    #: 옮긴 절점 [(출발, 도착, 설명)] — 출발점만 찍을지(왼쪽) 화살표까지(오른쪽)
    moves: list = field(default_factory=list)
    move_arrow: bool = False


def _outline(m, idxs) -> tuple[list, list, list]:
    xs, ys, zs = [], [], []
    for i in idxs:
        P = [m.coords[n] for n in m.shells[i][2]]
        for p in P + [P[0]]:
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


def _mesh3d(m, idxs, color, hover, group, visible=True, name=""):
    xs, ys, zs, ti, tj, tk, txt = [], [], [], [], [], [], []
    v = 0
    for i in idxs:
        eid, etype, nodes = m.shells[i]
        P = [m.coords[n] for n in nodes]
        for p in P:
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
            txt.append(hover[i])
        ti.append(v); tj.append(v + 1); tk.append(v + 2)
        if len(P) == 4:
            ti.append(v); tj.append(v + 2); tk.append(v + 3)
        v += len(P)
    return go.Mesh3d(x=xs, y=ys, z=zs, i=ti, j=tj, k=tk, color=color, flatshading=True,
                     opacity=1.0, text=txt, hoverinfo="text", name=name, meta=group,
                     visible=visible)


def pane_traces(s: PaneSpec) -> list:
    m = s.mesh
    rep = m.rep
    Q = [F.shell_quality(m.corners(i)) for i in range(len(m.shells))]
    hover = {}
    for i, (eid, etype, nodes) in enumerate(m.shells):
        q = Q[i]
        hover[i] = (f"<b>{etype} {eid}</b><br>내각 {q.angle_min:.1f}°~{q.angle_max:.1f}°"
                    f"{' <b>불량</b>' if F.is_bad_shell(q) else ''}<br>"
                    f"aspect {q.aspect:.2f}  skew {q.skew:.1f}°<br>"
                    f"절점 {', '.join(map(str, nodes))}"
                    + (f"<br><b>{s.fill[i]}</b>" if i in s.fill else ""))
    plain = [i for i in range(len(m.shells)) if i not in s.fill]
    hl = sorted(s.fill)
    tr = [_mesh3d(m, plain, NEUTRAL_SHELL, hover, "base", name="쉘 요소")]
    fill_color = DELETED if s.fill_group == "fill_rm" else CREATED
    tr.append(_mesh3d(m, hl, fill_color, hover, s.fill_group, name="강조 요소"))
    tr.append(_mesh3d(m, hl, NEUTRAL_SHELL, hover, "n:" + s.fill_group, visible=False))
    # 강조 요소 테두리 — 면 색만으로는 경계가 붙어 보여서 몇 개인지 안 읽힌다
    x, y, z = _outline(m, hl)
    tr.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=fill_color, width=6),
                           hoverinfo="skip", meta=s.fill_group))

    # 쉘 경계
    sx, sy, sz = [], [], []
    for e in rep.edge_shells:
        a, b = tuple(e)
        pa, pb = m.coords[a], m.coords[b]
        sx += [pa[0], pb[0], None]; sy += [pa[1], pb[1], None]; sz += [pa[2], pb[2], None]
    tr.append(go.Scatter3d(x=sx, y=sy, z=sz, mode="lines", line=dict(color=SHELL_EDGE, width=2.5),
                           hoverinfo="skip", meta="edges"))
    # 불량각
    x, y, z = _outline(m, [i for i, q in enumerate(Q) if F.is_bad_shell(q)])
    tr.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=BAD_OUTLINE, width=5),
                           hoverinfo="skip", meta="bad", visible="legendonly"))
    # 상대 쪽에 없는 변경 — 주황 굵은 테두리
    x, y, z = _outline(m, sorted(s.only))
    tr.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=ONLY_ONE, width=11),
                           hoverinfo="skip", meta="diff"))

    # 1D — 바뀐 것은 굵게, 나머지는 파랑
    for want, color, width, group in ((False, NEUTRAL_LINE, 4, "lines"), (True, s.line_color, 10, "line1d")):
        lx, ly, lz, lt = [], [], [], []
        for li, (eid, etype, (a, b)) in enumerate(m.lines):
            if (li in s.lines) != want:
                continue
            pa, pb = m.coords[a], m.coords[b]
            lx += [pa[0], pb[0], None]; ly += [pa[1], pb[1], None]; lz += [pa[2], pb[2], None]
            t = f"<b>{etype} {eid}</b><br>절점 {a}-{b}" + (f"<br><b>{s.lines[li]}</b>" if want else "")
            lt += [t, t, ""]
        tr.append(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color=color, width=width),
                               text=lt, hoverinfo="text", meta=group))

    # ext 절점
    P = np.array([m.coords[n] for n in sorted(m.coords) if m.ext.get(n, False)]).reshape(-1, 3)
    tr.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                           marker=dict(size=4, color=EXT_MARKER, symbol="circle-open", line=dict(width=2)),
                           hoverinfo="skip", meta="ext", visible="legendonly"))
    # 사라진 절점 자리
    pts = [(n, t) for n, t in s.delnodes.items() if n in m.coords]
    Pn = np.array([m.coords[n] for n, _ in pts]).reshape(-1, 3)
    tr.append(go.Scatter3d(x=Pn[:, 0], y=Pn[:, 1], z=Pn[:, 2], mode="markers",
                           marker=dict(size=9, color=CHANGED_NODE, symbol="x",
                                       line=dict(width=2, color=CHANGED_NODE)),
                           text=[t for _, t in pts], hoverinfo="text", meta="delnode"))
    # 옮긴 절점 — 출발(왼쪽) 또는 도착+화살표(오른쪽)
    if s.moves:
        ends = np.array([(p1 if s.move_arrow else p0) for p0, p1, _ in s.moves]).reshape(-1, 3)
        tr.append(go.Scatter3d(x=ends[:, 0], y=ends[:, 1], z=ends[:, 2], mode="markers",
                               marker=dict(size=8, color=MOVED_NODE, symbol="diamond",
                                           line=dict(width=1, color="#854d0e")),
                               text=[("도착: " if s.move_arrow else "출발: ") + t for _, _, t in s.moves],
                               hoverinfo="text", meta="moved"))
        if s.move_arrow:
            ax, ay, az = [], [], []
            for p0, p1, _ in s.moves:
                ax += [p0[0], p1[0], None]; ay += [p0[1], p1[1], None]; az += [p0[2], p1[2], None]
            tr.append(go.Scatter3d(x=ax, y=ay, z=az, mode="lines",
                                   line=dict(color=MOVED_NODE, width=7, dash="dash"),
                                   hoverinfo="skip", meta="moved"))
            P0 = np.array([p0 for p0, _, _ in s.moves]).reshape(-1, 3)
            tr.append(go.Scatter3d(x=P0[:, 0], y=P0[:, 1], z=P0[:, 2], mode="markers",
                                   marker=dict(size=5, color="#ffffff", symbol="circle",
                                               line=dict(width=2, color="#854d0e")),
                                   text=["출발: " + t for _, _, t in s.moves], hoverinfo="text",
                                   meta="moved"))
    # '변경 위치로' 표시 — 처음엔 비어 있고 고르면 채워진다
    tr.append(go.Scatter3d(x=[], y=[], z=[], mode="markers",
                           marker=dict(size=34, color=FOCUS, opacity=0.25, symbol="circle"),
                           hoverinfo="skip", meta="focus"))
    return tr


def _ranges(meshes) -> dict:
    P = np.array([c for m in meshes for c in m.coords.values()], dtype=float)
    lo, hi = P.min(0), P.max(0)
    pad = 0.05 * np.maximum(hi - lo, 1.0)
    lo, hi = lo - pad, hi + pad
    return {ax: [float(lo[i]), float(hi[i])] for i, ax in enumerate(("xaxis", "yaxis", "zaxis"))}


def _layout(title: str, ranges: dict) -> go.Layout:
    return go.Layout(
        title=dict(text=title, x=0.5, font=dict(size=13)),
        scene=dict(aspectmode="data",
                   **{ax: dict(range=r, showticklabels=False, title="", autorange=False)
                      for ax, r in ranges.items()}),
        margin=dict(l=0, r=0, t=28, b=0), showlegend=False, hoverlabel=dict(align="left"),
    )


# ──────────────────────────────────────────────────────────────────────────
# 비교 — 원본 / 모델 / 사람 셋으로 네 화면의 강조를 정한다
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Change:
    """변경 목록의 한 줄 — 표와 '변경 위치로' 가 같은 목록을 쓴다."""

    who: str          # "모델" | "사람"
    what: str         # "요소 삭제" | "요소 생성" | "절점 삭제" | "절점 이동" | "1D 삭제" | "1D 생성"
    target: str       # 요소/절점 번호
    shared: bool      # 상대 쪽도 같은 변경을 했나 (좌표 기준)
    note: str
    at: tuple         # 확대할 중심 좌표


def _skey(m, i) -> frozenset:
    return frozenset(pkey(m.coords[n]) for n in m.shells[i][2])


def _lkey(m, i) -> frozenset:
    return frozenset(pkey(m.coords[n]) for n in m.lines[i][2])


def build_view(bad, model, human, model_absorbed: dict | None = None,
               human_absorbed: dict | None = None) -> tuple[list[PaneSpec], list[Change], dict]:
    """네 화면 명세 + 변경 목록 + 요약."""
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
    del_pos_h = {pkey(bad.coords[n]) for n in dh.deleted_nodes} | \
                {pkey(p0) for p0, _, _ in dh.moved_nodes.values()}
    del_pos_m = {pkey(bad.coords[n]) for n in dm.deleted_nodes} | \
                {pkey(p0) for p0, _, _ in dm.moved_nodes.values()}

    def tag(both: bool, other: str) -> str:
        return f"{other}도 같은 변경 ✓" if both else f"{other}는 안 한 변경"

    panes = [
        PaneSpec(bad, "원본 — 모델이 지운 것",
                 fill={i: f"모델이 지움 · {tag(_skey(bad, i) in rm_h, '사람')}" for i in dm.removed_shells},
                 fill_group="fill_rm",
                 only={i for i in dm.removed_shells if _skey(bad, i) not in rm_h},
                 lines={i: f"모델이 지운 1D · {tag(_lkey(bad, i) in lrm_h, '사람')}" for i in dm.removed_lines},
                 line_color=DELETED, delnodes=dict(dm.deleted_nodes),
                 moves=list(dm.moved_nodes.values())),
        PaneSpec(model, "모델 재생성 — 모델이 만든 것",
                 fill={i: f"모델이 만듦 · {tag(_skey(model, i) in cr_h, '사람')}" for i in dm.created_shells},
                 fill_group="fill_cr",
                 only={i for i in dm.created_shells if _skey(model, i) not in cr_h},
                 lines={i: f"모델이 만든 1D · {tag(_lkey(model, i) in lcr_h, '사람')}" for i in dm.created_lines},
                 line_color=CREATED, moves=list(dm.moved_nodes.values()), move_arrow=True),
        PaneSpec(bad, "원본 — 사람이 지운 것",
                 fill={i: f"사람이 지움 · {tag(_skey(bad, i) in rm_m, '모델')}" for i in dh.removed_shells},
                 fill_group="fill_rm",
                 only={i for i in dh.removed_shells if _skey(bad, i) not in rm_m},
                 lines={i: f"사람이 지운 1D · {tag(_lkey(bad, i) in lrm_m, '모델')}" for i in dh.removed_lines},
                 line_color=DELETED, delnodes=dict(dh.deleted_nodes),
                 moves=list(dh.moved_nodes.values())),
        PaneSpec(human, "사람 수정 — 사람이 만든 것",
                 fill={i: f"사람이 만듦 · {tag(_skey(human, i) in cr_m, '모델')}" for i in dh.created_shells},
                 fill_group="fill_cr",
                 only={i for i in dh.created_shells if _skey(human, i) not in cr_m},
                 lines={i: f"사람이 만든 1D · {tag(_lkey(human, i) in lcr_m, '모델')}" for i in dh.created_lines},
                 line_color=CREATED, moves=list(dh.moved_nodes.values()), move_arrow=True),
    ]

    # ── 변경 목록 ──────────────────────────────────────────────────────
    ch: list[Change] = []
    cen = lambda m, i: tuple(np.mean([m.coords[n] for n in m.shells[i][2]], axis=0))
    lcen = lambda m, i: tuple(np.mean([m.coords[n] for n in m.lines[i][2]], axis=0))
    for who, d, res, rm_o, cr_o, lrm_o, lcr_o, del_o in (
            ("모델", dm, model, rm_h, cr_h, lrm_h, lcr_h, del_pos_h),
            ("사람", dh, human, rm_m, cr_m, lrm_m, lcr_m, del_pos_m)):
        for i in d.removed_shells:
            eid, t, ns = bad.shells[i]
            ch.append(Change(who, "요소 삭제", f"{t} {eid}", _skey(bad, i) in rm_o,
                             f"절점 {', '.join(map(str, ns))}", cen(bad, i)))
        for i in d.created_shells:
            eid, t, ns = res.shells[i]
            ch.append(Change(who, "요소 생성", f"{t} {eid}", _skey(res, i) in cr_o,
                             f"절점 {', '.join(map(str, ns))}", cen(res, i)))
        for n, why in d.deleted_nodes.items():
            ch.append(Change(who, "절점 삭제", f"GRID {n}", pkey(bad.coords[n]) in del_o, why,
                             tuple(bad.coords[n])))
        for n, (p0, p1, why) in d.moved_nodes.items():
            ch.append(Change(who, "절점 이동", f"GRID {n}", pkey(p0) in del_o, why,
                             tuple((np.asarray(p0) + np.asarray(p1)) / 2)))
        for i in d.removed_lines:
            eid, t, (a, b) = bad.lines[i]
            ch.append(Change(who, "1D 삭제", f"{t} {eid}", _lkey(bad, i) in lrm_o, f"절점 {a}-{b}",
                             lcen(bad, i)))
        for i in d.created_lines:
            eid, t, (a, b) = res.lines[i]
            ch.append(Change(who, "1D 생성", f"{t} {eid}", _lkey(res, i) in lcr_o, f"절점 {a}-{b}",
                             lcen(res, i)))

    summary = {
        "model": {"지운 요소": len(dm.removed_shells), "만든 요소": len(dm.created_shells),
                  "사라진 절점": len(dm.deleted_nodes), "옮긴 절점": len(dm.moved_nodes),
                  "1D 삭제": len(dm.removed_lines), "1D 생성": len(dm.created_lines)},
        "human": {"지운 요소": len(dh.removed_shells), "만든 요소": len(dh.created_shells),
                  "사라진 절점": len(dh.deleted_nodes), "옮긴 절점": len(dh.moved_nodes),
                  "1D 삭제": len(dh.removed_lines), "1D 생성": len(dh.created_lines)},
        "stats": {"원본": stats(bad), "모델": stats(model), "사람": stats(human)},
    }
    return panes, ch, summary


# ──────────────────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────────────────
_PAGE = """<!doctype html>
<meta charset="utf-8">
<script>{plotlyjs}</script>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin:0; font-family:"Malgun Gothic","Segoe UI",system-ui,sans-serif;
         background:#f8fafc; color:#0f172a; font-size:13px; }}
  .bar {{ display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; padding:7px 12px;
          background:#fff; border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:5; }}
  .bar label {{ cursor:pointer; user-select:none; white-space:nowrap; }}
  .bar button, .bar select {{ font:inherit; padding:2px 9px; border:1px solid #cbd5e1;
                 background:#fff; border-radius:5px; cursor:pointer; }}
  .bar button:hover {{ background:#f1f5f9; }}
  .bar select {{ max-width:430px; }}
  .key {{ display:inline-block; width:11px; height:11px; border-radius:2px;
          vertical-align:-1px; margin-right:4px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr;
           gap:1px; background:#cbd5e1; height:calc(100vh - {bar_h}px); }}
  .pane {{ background:#fff; min-height:0; }}
  .pane > div {{ width:100%; height:100%; }}
  .sync {{ color:#059669; font-weight:600; }}
</style>
<div class="bar">
  <button id="reset">시점 초기화</button>
  <select id="focus"><option value="">변경 위치로 확대…</option>{focus_opts}</select>
  {toggles}
  <span class="sync">↔ 네 화면 시점 연동</span>
</div>
<div class="grid">
  <div class="pane"><div id="P0"></div></div><div class="pane"><div id="P1"></div></div>
  <div class="pane"><div id="P2"></div></div><div class="pane"><div id="P3"></div></div>
</div>
<script>
const FIGS = {figs};
const FOCUS = {focus};
const RANGES = {ranges};
const cfg = {{responsive:true, displaylogo:false, modeBarButtonsToRemove:['toImage']}};
const ids = ['P0','P1','P2','P3'];
Promise.all(ids.map((id, i) => Plotly.newPlot(id, FIGS[i].data, FIGS[i].layout, cfg))).then(() => {{
  const G = ids.map(id => document.getElementById(id));

  // ── 카메라 연동 (viz_compare 와 같은 방식, 화면 4개로 일반화) ──────────
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

  // ── 강조 켜고 끄기 — 트레이스의 meta 로 묶음을 찾는다 ──────────────────
  function setGroup(group, on) {{
    G.forEach(g => {{
      const show = [], hide = [];
      g.data.forEach((t, i) => {{
        if (t.meta === group) (on ? show : hide).push(i);
        else if (t.meta === 'n:' + group) (on ? hide : show).push(i);
      }});
      if (show.length) Plotly.restyle(g, {{visible: true}}, show);
      if (hide.length) Plotly.restyle(g, {{visible: false}}, hide);
    }});
  }}
  document.querySelectorAll('.bar input[data-g]').forEach(box => {{
    if (!box.checked) setGroup(box.dataset.g, false);
    box.addEventListener('change', () => setGroup(box.dataset.g, box.checked));
  }});

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
  window.__regen = {{ready: true, groups: () => G.map(g => g.data.map(t => [t.meta, t.visible]))}};
}});
</script>
"""


def render_html(panes: list[PaneSpec], changes: list[Change], focus_radius: float,
                bar_h: int = 64) -> str:
    """네 화면 HTML. ``focus_radius`` 는 '변경 위치로' 확대할 때 상자 반폭(mm)."""
    ranges = _ranges([p.mesh for p in panes])
    figs = [go.Figure(data=pane_traces(p), layout=_layout(p.title, ranges)) for p in panes]
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
        focus_opts=opts, toggles=toggles, bar_h=bar_h,
    )

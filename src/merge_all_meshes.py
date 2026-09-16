"""폴더 안의 모든 .bdf 를 절점 id 기준으로 병합해 하나의 메쉬로 시각화한다.

패치들이 절점 id 를 공유한다는 사실(한 id 가 평균 3개 파일, 최대 13개 파일에 등장)이
실제로 이어진 하나의 형상을 이루는지, 아니면 서로 겹치는 국소 구역들의 모음인지
눈으로 직접 확인하기 위한 일회성 탐색 스크립트. 학습 파이프라인과는 무관하다.

    python src/merge_all_meshes.py
    python src/merge_all_meshes.py --limit 300      # 빠르게 미리보기

절점 id 를 좌표의 유일한 키로 보고 합친다 — 같은 id 가 파일마다 좌표가 다르면
"공유"가 아니라 우연히 id 가 겹친 것이므로 그 개수를 세서 알려준다.

사이드바에서 파일 이름을 클릭하면 그 bdf 가 차지하는 쉘/1D/가장자리만 노란색으로,
나머지는 회색으로 흐려진다 (Plotly.restyle 로 클라이언트 사이드에서 즉시 갱신 —
파일마다 페이지를 새로 만들지 않는다).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths
from bdf_io import read_bdf

DEFAULT_SHELL = "#d6dbe3"
DIM_GRAY = "#c3c8d1"
DIM_LINE_GRAY = "#b6bcc6"
YELLOW = "#facc15"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="폴더 전체 bdf 를 절점 id 로 병합해 시각화")
    ap.add_argument("--folder", default=paths.FE_DIR, help="bdf 폴더 (기본: paths.FE_DIR)")
    ap.add_argument("--out", default=os.path.join(paths.ROOT, "merged_mesh.html"))
    ap.add_argument("--limit", type=int, default=None, help="앞 N개 파일만 (미리보기용)")
    ap.add_argument("--no-lines", action="store_true", help="1D 요소(CBAR/CBEAM) 제외")
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.folder, "*.bdf")))
    if a.limit:
        files = files[: a.limit]
    if not files:
        print(f"bdf 파일을 찾지 못함: {a.folder}")
        return 1
    print(f"{len(files)}개 파일 병합 중...")

    coords: dict[int, tuple[float, float, float]] = {}
    node_file_count: dict[int, int] = {}
    coord_conflicts = 0

    shells: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    lines: dict[tuple[str, tuple[int, int]], tuple[int, int]] = {}
    patch_boundary_edges: set[tuple[int, int]] = set()

    # 파일별로 자기가 가진 요소의 "키"를 모아둔다 — 전역 인덱스가 확정된 뒤
    # (모든 파일을 다 읽고 중복 제거가 끝난 뒤) 인덱스로 변환한다.
    file_shell_keys: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
    file_line_keys: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    file_boundary_raw: dict[str, list[tuple[int, int]]] = {}

    t0 = time.time()
    for i, p in enumerate(files, start=1):
        name = os.path.splitext(os.path.basename(p))[0]
        c, sh, ln = read_bdf(p)
        for nid, xyz in c.items():
            node_file_count[nid] = node_file_count.get(nid, 0) + 1
            prev = coords.get(nid)
            if prev is None:
                coords[nid] = xyz
            elif prev != xyz:
                coord_conflicts += 1

        sk = []
        for eid, etype, nodes in sh:
            key = (etype, tuple(sorted(nodes)))
            shells[key] = nodes  # 절점 순서를 보존해야 면 분할이 올바름
            sk.append(key)
        file_shell_keys[name] = sk

        lk = []
        if not a.no_lines:
            for eid, etype, nodes in ln:
                key = (etype, (min(nodes), max(nodes)))
                lines[key] = nodes
                lk.append(key)
        file_line_keys[name] = lk

        file_edge_count: dict[tuple[int, int], int] = {}
        for eid, etype, nodes in sh:
            k = len(nodes)
            for ii in range(k):
                a1, b1 = nodes[ii], nodes[(ii + 1) % k]
                key2 = (min(a1, b1), max(a1, b1))
                file_edge_count[key2] = file_edge_count.get(key2, 0) + 1
        fb = [key2 for key2, cnt in file_edge_count.items() if cnt == 1]
        patch_boundary_edges.update(fb)
        file_boundary_raw[name] = fb

        if i % 500 == 0:
            print(f"  ... {i}/{len(files)}  ({time.time() - t0:.1f}s)")

    print(f"\n절점 {len(coords)}개 (좌표 불일치 {coord_conflicts}건"
          f"{' — 같은 id 인데 파일마다 위치가 다름' if coord_conflicts else ''})")
    print(f"쉘 {len(shells)}개 (원본 합계 대비 중복 제거됨)")
    if not a.no_lines:
        print(f"1D 요소 {len(lines)}개 (중복 제거됨)")
    reuse = np.array(list(node_file_count.values()))
    print(f"절점당 등장 파일 수: 평균 {reuse.mean():.2f}, 최대 {reuse.max()}, "
          f"단독(1개 파일에만 등장) {int((reuse == 1).sum())}개")

    # ── 좌표/면 배열 구성 ────────────────────────────────────────────
    node_ids = sorted(coords)
    idx = {nid: i for i, nid in enumerate(node_ids)}
    P = np.array([coords[n] for n in node_ids])

    ti, tj, tk = [], [], []
    shell_range: dict[tuple[str, tuple[int, ...]], tuple[int, int]] = {}
    for key, nodes in shells.items():
        ids = [idx[n] for n in nodes]
        start = len(ti)
        ti.append(ids[0]); tj.append(ids[1]); tk.append(ids[2])
        if len(ids) == 4:
            ti.append(ids[0]); tj.append(ids[2]); tk.append(ids[3])
        shell_range[key] = (start, len(ti) - start)

    line_keys = list(lines.keys())
    line_index = {key: i for i, key in enumerate(line_keys)}
    lineA = [idx[lines[key][0]] for key in line_keys]
    lineB = [idx[lines[key][1]] for key in line_keys]

    boundary_list = sorted(patch_boundary_edges)
    boundary_index = {e: i for i, e in enumerate(boundary_list)}
    boundA = [idx[e[0]] for e in boundary_list]
    boundB = [idx[e[1]] for e in boundary_list]

    print(f"\n형상 좌표 범위 (mm): "
          f"x[{P[:,0].min():.0f},{P[:,0].max():.0f}]  "
          f"y[{P[:,1].min():.0f},{P[:,1].max():.0f}]  "
          f"z[{P[:,2].min():.0f},{P[:,2].max():.0f}]")
    print(f"bdf 패치 가장자리 {len(boundary_list)}개")

    # ── 파일별 인덱스로 변환 ─────────────────────────────────────────
    file_names = sorted(file_shell_keys)
    file_tri_idx: dict[str, list[int]] = {}
    for name, keys in file_shell_keys.items():
        out: list[int] = []
        for key in keys:
            start, cnt = shell_range[key]
            out.extend(range(start, start + cnt))
        file_tri_idx[name] = out

    file_line_idx = {
        name: [line_index[key] for key in keys] for name, keys in file_line_keys.items()
    }
    file_boundary_idx = {
        name: [boundary_index[e] for e in raw] for name, raw in file_boundary_raw.items()
    }

    # ── Plotly figure ────────────────────────────────────────────────
    # 트레이스 순서를 고정해야 JS 에서 인덱스로 restyle 할 수 있다.
    #   0=쉘  1=가장자리  2=CBAR선  3=CBAR점  4=CBEAM선  5=CBEAM점
    #   6=강조 1D(노랑, 빈 상태로 시작)  7=강조 가장자리(노랑, 빈 상태로 시작)
    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], i=ti, j=tj, k=tk,
        facecolor=[DEFAULT_SHELL] * len(ti), opacity=0.5,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0),
        name="병합된 쉘 요소", hoverinfo="skip",
    ))

    ex, ey, ez = [], [], []
    for a1, b1 in boundary_list:
        pa, pb = coords[a1], coords[b1]
        ex += [pa[0], pb[0], None]; ey += [pa[1], pb[1], None]; ez += [pa[2], pb[2], None]
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="#000000", width=2), opacity=0.8,
        name=f"bdf 패치 가장자리 ({len(boundary_list)}개)", hoverinfo="skip",
    ))

    if not a.no_lines and lines:
        for etype, color in (("CBAR", "#dc2626"), ("CBEAM", "#7c3aed")):
            pairs = [lines[key] for key in line_keys if key[0] == etype]
            lx, ly, lz, mx, my, mz = [], [], [], [], [], []
            for a1, b1 in pairs:
                pa, pb = coords[a1], coords[b1]
                lx += [pa[0], pb[0], None]; ly += [pa[1], pb[1], None]; lz += [pa[2], pb[2], None]
                mx += [pa[0], pb[0]]; my += [pa[1], pb[1]]; mz += [pa[2], pb[2]]
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz, mode="lines",
                line=dict(color=color, width=4), opacity=1.0,
                name=f"{etype} ({len(pairs)}개)", hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter3d(
                x=mx, y=my, z=mz, mode="markers",
                marker=dict(size=1, color=color),
                showlegend=False, hoverinfo="skip",
            ))
    else:
        # 인덱스를 6,7 로 맞추기 위해 자리만 채워둔다 (숨김)
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", visible=False))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", visible=False))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", visible=False))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", visible=False))

    # 6: 선택 파일의 1D 요소 강조(노랑, 처음엔 빈 트레이스)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line=dict(color=YELLOW, width=7), opacity=1.0,
        name="선택한 bdf — 1D 요소", hoverinfo="skip",
    ))
    # 7: 선택 파일의 가장자리 강조(노랑, 처음엔 빈 트레이스)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line=dict(color=YELLOW, width=6), opacity=1.0,
        name="선택한 bdf — 가장자리", hoverinfo="skip",
    ))

    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
    )

    plot_html = fig.to_html(full_html=False, include_plotlyjs=True, div_id="plotDiv")

    payload = {
        "nodeX": np.round(P[:, 0], 2).tolist(),
        "nodeY": np.round(P[:, 1], 2).tolist(),
        "nodeZ": np.round(P[:, 2], 2).tolist(),
        "lineA": lineA, "lineB": lineB,
        "boundA": boundA, "boundB": boundB,
        "nTri": len(ti),
        "files": file_names,
        "fileTriIdx": file_tri_idx,
        "fileLineIdx": file_line_idx,
        "fileBoundaryIdx": file_boundary_idx,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    print(f"\n임베딩 데이터 크기: {len(payload_json) / 1e6:.1f} MB")

    html = _PAGE_TEMPLATE.format(
        title=f"{os.path.basename(a.folder)} — {len(files)}개 파일 병합 "
              f"({len(node_ids)}절점, {len(shells)}쉘)",
        plot_html=plot_html,
        payload_json=payload_json,
        default_shell=DEFAULT_SHELL, dim_gray=DIM_GRAY,
        dim_line_gray=DIM_LINE_GRAY, yellow=YELLOW,
    )
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"저장: {a.out}  (브라우저로 열어서 왼쪽 목록에서 파일을 클릭)")
    return 0


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  html, body {{ margin: 0; height: 100%; font-family: -apple-system, "Malgun Gothic", sans-serif; }}
  body {{ display: flex; }}
  #sidebar {{
    width: 280px; flex-shrink: 0; border-right: 1px solid #d9dce1;
    display: flex; flex-direction: column; box-sizing: border-box; padding: 10px;
  }}
  #sidebar h3 {{ margin: 0 0 8px; font-size: 14px; color: #1f2937; }}
  #search {{
    width: 100%; box-sizing: border-box; padding: 6px 8px; margin-bottom: 6px;
    border: 1px solid #cbd2db; border-radius: 4px; font-size: 13px;
  }}
  #clearBtn {{
    width: 100%; padding: 6px; margin-bottom: 8px; border: 1px solid #cbd2db;
    border-radius: 4px; background: #f3f4f6; cursor: pointer; font-size: 13px;
  }}
  #clearBtn:hover {{ background: #e5e7eb; }}
  #info {{ font-size: 12px; color: #4b5563; margin-bottom: 8px; min-height: 34px; }}
  #fileList {{ flex: 1; overflow-y: auto; border-top: 1px solid #eee; }}
  .file-item {{
    padding: 3px 6px; font-size: 12px; cursor: pointer; color: #374151;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .file-item:hover {{ background: #f3f4f6; }}
  .file-item.selected {{ background: {yellow}; color: #111827; font-weight: 600; }}
  #plotWrap {{ flex: 1; min-width: 0; }}
</style>
</head>
<body>
<div id="sidebar">
  <h3>bdf 파일</h3>
  <input id="search" placeholder="파일 검색 (예: 0001)">
  <button id="clearBtn">선택 해제</button>
  <div id="info">파일을 클릭하면 병합된 메쉬 위에서 노란색으로 강조됩니다.</div>
  <div id="fileList"></div>
</div>
<div id="plotWrap">
{plot_html}
</div>
<script>
const DATA = {payload_json};
const COLOR = {{
  shellDefault: "{default_shell}", shellDim: "{dim_gray}",
  lineDim: "{dim_line_gray}", yellow: "{yellow}",
}};

function buildSegments(idxList, endA, endB) {{
  const x = [], y = [], z = [];
  for (const i of idxList) {{
    const a = endA[i], b = endB[i];
    x.push(DATA.nodeX[a], DATA.nodeX[b], null);
    y.push(DATA.nodeY[a], DATA.nodeY[b], null);
    z.push(DATA.nodeZ[a], DATA.nodeZ[b], null);
  }}
  return {{x, y, z}};
}}

let current = null;
let gd = null;

function clearHighlight() {{
  current = null;
  const faceColor = new Array(DATA.nTri).fill(COLOR.shellDefault);
  Plotly.restyle(gd, {{facecolor: [faceColor]}}, [0]);
  Plotly.restyle(gd, {{"line.color": "#000000", opacity: 0.8}}, [1]);
  Plotly.restyle(gd, {{"line.color": "#dc2626", opacity: 1.0}}, [2]);
  Plotly.restyle(gd, {{"marker.color": "#dc2626", opacity: 1.0}}, [3]);
  Plotly.restyle(gd, {{"line.color": "#7c3aed", opacity: 1.0}}, [4]);
  Plotly.restyle(gd, {{"marker.color": "#7c3aed", opacity: 1.0}}, [5]);
  Plotly.restyle(gd, {{x: [[]], y: [[]], z: [[]]}}, [6]);
  Plotly.restyle(gd, {{x: [[]], y: [[]], z: [[]]}}, [7]);
  document.getElementById("info").textContent =
    "파일을 클릭하면 병합된 메쉬 위에서 노란색으로 강조됩니다.";
  document.querySelectorAll(".file-item.selected").forEach(el => el.classList.remove("selected"));
}}

function highlightFile(name, el) {{
  current = name;
  const triIdx = DATA.fileTriIdx[name] || [];
  const faceColor = new Array(DATA.nTri).fill(COLOR.shellDim);
  for (const t of triIdx) faceColor[t] = COLOR.yellow;
  Plotly.restyle(gd, {{facecolor: [faceColor]}}, [0]);

  Plotly.restyle(gd, {{"line.color": COLOR.lineDim, opacity: 0.35}}, [1]);
  Plotly.restyle(gd, {{"line.color": COLOR.lineDim, opacity: 0.35}}, [2]);
  Plotly.restyle(gd, {{"marker.color": COLOR.lineDim, opacity: 0.35}}, [3]);
  Plotly.restyle(gd, {{"line.color": COLOR.lineDim, opacity: 0.35}}, [4]);
  Plotly.restyle(gd, {{"marker.color": COLOR.lineDim, opacity: 0.35}}, [5]);

  const segL = buildSegments(DATA.fileLineIdx[name] || [], DATA.lineA, DATA.lineB);
  Plotly.restyle(gd, {{x: [segL.x], y: [segL.y], z: [segL.z]}}, [6]);
  const segB = buildSegments(DATA.fileBoundaryIdx[name] || [], DATA.boundA, DATA.boundB);
  Plotly.restyle(gd, {{x: [segB.x], y: [segB.y], z: [segB.z]}}, [7]);

  document.getElementById("info").textContent =
    `${{name}} — 쉘 ${{triIdx.length}}면(삼각분할 기준), `
    + `1D ${{(DATA.fileLineIdx[name]||[]).length}}개, `
    + `가장자리 ${{(DATA.fileBoundaryIdx[name]||[]).length}}개`;

  document.querySelectorAll(".file-item.selected").forEach(el2 => el2.classList.remove("selected"));
  if (el) el.classList.add("selected");
}}

function initSidebar() {{
  gd = document.getElementById("plotDiv");
  document.querySelector("#sidebar h3").textContent = `bdf 파일 (${{DATA.files.length}}개)`;

  const listEl = document.getElementById("fileList");
  const frag = document.createDocumentFragment();
  DATA.files.forEach(name => {{
    const div = document.createElement("div");
    div.className = "file-item";
    div.textContent = name;
    div.addEventListener("click", () => highlightFile(name, div));
    frag.appendChild(div);
  }});
  listEl.appendChild(frag);

  document.getElementById("search").addEventListener("input", (e) => {{
    const q = e.target.value.trim().toLowerCase();
    listEl.querySelectorAll(".file-item").forEach(el => {{
      el.style.display = el.textContent.toLowerCase().includes(q) ? "" : "none";
    }});
  }});
  document.getElementById("clearBtn").addEventListener("click", clearHighlight);
}}

window.addEventListener("load", initSidebar);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # 사내 PC 는 라벨 엑셀·데이터 파일에 암호가 걸리므로 파일을 열기 전에 해제한다.
    from fasoo_drm import setting_fasoo

    setting_fasoo()
    sys.exit(main())

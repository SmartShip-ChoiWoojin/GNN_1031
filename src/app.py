"""인터랙티브 3D 메쉬 뷰어 — bdf 파일을 불러와 회전/줌/패닝하며 요소별 어텐션을 본다.

    streamlit run src/app.py

기능
  - 샘플 폴더에서 모델 선택, 오분류(FP/FN) 목록에서 선택, 또는 .bdf(+.json) 직접 업로드
  - 오분류 목록은 train.py 가 저장한 out-of-fold 예측(oof.csv)을 그대로 써서
    재추론 없이 즉시 FP/FN 을 걸러내고, 확신도(0.5 에서 먼 정도) 순으로 정렬한다
  - Plotly 3D — 마우스로 자유롭게 회전·확대·이동, 요소에 마우스 올리면 상세정보
  - 초기 화면은 채색 없이 메쉬 형상만 명확히 표시. "중요도 강조"를 켜면
    임계값 이상인 요소만 단일 색 계열(빨강)로, 나머지는 중립 회색으로 표시
  - 학습된 GNN(runs/*)으로 정상/비정상 확률 + 요소별 어텐션(문제 요소 지목)
  - "그래프 구조" 탭 — 메쉬가 GNN 입력 그래프로 어떻게 변환됐는지
    (요소기반이면 요소 중심점=노드, 관계 종류별 색의 엣지) 시각화
  - 학습셋에 있는 이름이면 그 요소를 본 적 없는 fold 모델로 정직하게 예측,
    없는 이름(직접 업로드)이면 5-fold 앙상블 평균으로 예측
"""

from __future__ import annotations

import glob
import hashlib
import os
import sys
import tempfile
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset as DS
import features as F
import graphs
import paths
from bdf_io import discover_pairs, load_raw_mesh, read_bdf
from explain import (
    attention_for,
    fold_assignment,
    load_fold_model,
    load_result,
    relative_importance,
)
from labels import load_label_table
from meshrep import MeshRep

st.set_page_config(page_title="메쉬 정상/비정상 뷰어", layout="wide")

# ── 색상 상수 ────────────────────────────────────────────────────────────
NEUTRAL_SHELL = "#c7ced9"     # 강조 안 된 쉘 — 중립 회색, 형상이 잘 보이도록
NEUTRAL_LINE = "#1d4ed8"      # 강조 안 된 1D 요소 — 쉘과 구분되는 진한 파랑
RED_LIGHT = "#fcae91"         # 강조 요소 — 빨강 계열 (밝음=경계, 진함=핵심 원인)
RED_DARK = "#99000d"
BAD_OUTLINE = "#111827"       # 규칙 기반 불량각도 요소 테두리 — 검정 (빨강 강조와 구분)
EXT_MARKER = "#06b6d4"


# ──────────────────────────────────────────────────────────────────────────
# 캐시된 로딩
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def available_runs() -> list[str]:
    out = []
    for d in sorted(glob.glob(os.path.join(paths.RUNS_DIR, "*"))):
        if os.path.isfile(os.path.join(d, "result.json")):
            out.append(os.path.basename(d))
    return out


@st.cache_data
def available_samples() -> list[str]:
    paired, _, _ = discover_pairs(paths.FE_DIR)
    return sorted(paired)


@st.cache_data
def label_lookup() -> dict[str, tuple[int, int]]:
    df = load_label_table(paths.LABEL_XLSX)
    return {r.name: (int(r.cls), int(r.ambig)) for r in df.itertuples()}


@st.cache_resource(show_spinner=False)
def get_run_context(run_dir: str):
    """(args, names, labels, arrays, fold_of, opts) — 학습 캐시와 fold 배정을 한 번만 계산."""
    res = load_result(run_dir)
    args = SimpleNamespace(**res["args"])
    opts = {}
    if args.graph == "elem":
        opts = {"corner_edges": not args.no_corner_edges, "include_lines": not args.no_lines}
    names, labels, arrays = DS.build_cache(args.graph, opts=opts, verbose=False)
    fold_of = fold_assignment(names, labels, args)
    return args, names, labels, arrays, fold_of, opts


@st.cache_resource(show_spinner=False)
def get_fold_model(run_dir: str, fold: int, device_str: str):
    device = torch.device(device_str)
    return load_fold_model(run_dir, fold, device)


# ──────────────────────────────────────────────────────────────────────────
# 파일 로딩 (업로드 or 샘플 폴더)
# ──────────────────────────────────────────────────────────────────────────
def parse_from_bytes(bdf_bytes: bytes, json_bytes: bytes | None, name_hint: str) -> MeshRep:
    """업로드된 바이트를 임시 파일로 내려쓴 뒤 MeshRep 으로 파싱한다."""
    with tempfile.TemporaryDirectory() as td:
        bdf_path = os.path.join(td, f"{name_hint}.bdf")
        with open(bdf_path, "wb") as fh:
            fh.write(bdf_bytes)

        if json_bytes is not None:
            json_path = os.path.join(td, f"{name_hint}.json")
            with open(json_path, "wb") as fh:
                fh.write(json_bytes)
        else:
            # json 이 없으면 전 절점을 '외부연결 없음'으로 간주하고 알린다
            coords, _, _ = read_bdf(bdf_path)
            json_path = os.path.join(td, f"{name_hint}.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                import json as _json
                _json.dump({str(k): False for k in coords}, fh)

        return MeshRep(load_raw_mesh(bdf_path, json_path))


@st.cache_resource(show_spinner=False)
def parse_sample(name: str) -> MeshRep:
    bdf_path = os.path.join(paths.FE_DIR, f"{name}.bdf")
    json_path = os.path.join(paths.FE_DIR, f"{name}.json")
    return MeshRep(load_raw_mesh(bdf_path, json_path))


@st.cache_data(show_spinner=False)
def load_oof_errors(run_dir: str) -> pd.DataFrame:
    """train.py 가 저장해둔 out-of-fold 예측(oof.csv/oof_names.txt)에서 FP/FN 만 추린다.

    재추론 없이 그대로 쓴다 — 각 샘플이 자기를 학습에 안 쓴 fold 로 이미 정직하게
    평가된 결과이므로, "왜 틀렸는지" 훑어볼 목록으로 그대로 신뢰할 수 있다.
    """
    oof_path = os.path.join(run_dir, "oof.csv")
    names_path = os.path.join(run_dir, "oof_names.txt")
    if not (os.path.isfile(oof_path) and os.path.isfile(names_path)):
        return pd.DataFrame(columns=["name", "y", "prob", "kind", "conf"])

    arr = np.loadtxt(oof_path, delimiter=",", skiprows=1)
    with open(names_path, "r", encoding="utf-8") as fh:
        names = fh.read().splitlines()

    y = arr[:, 1].astype(int)
    prob = arr[:, 2]
    pred = (prob > 0.5).astype(int)

    df = pd.DataFrame({"name": names, "y": y, "prob": prob})
    df["kind"] = np.select(
        [(y == 0) & (pred == 1), (y == 1) & (pred == 0)],
        ["FP", "FN"], default="정답",
    )
    # 틀린 정도(0.5 에서 얼마나 확신에 차서 틀렸는지) — 큰 순으로 보면
    # "모델이 왜 이렇게 확신했는데 틀렸지?"를 먼저 보게 된다.
    df["conf"] = np.abs(df["prob"] - 0.5)
    return df


# ──────────────────────────────────────────────────────────────────────────
# 예측
# ──────────────────────────────────────────────────────────────────────────
def predict(rep: MeshRep, run_dir: str, device_str: str) -> dict:
    """예측 확률 + 요소별 중요도. 학습셋에 있는 이름이면 그 fold 모델(정직한 평가),
    없으면 5-fold 앙상블 평균을 쓴다."""
    args, names, labels, arrays, fold_of, opts = get_run_context(run_dir)
    device = torch.device(device_str)

    ga = graphs.build(rep, args.graph, **opts)

    if rep.name in names:
        idx = names.index(rep.name)
        fold = int(fold_of[idx])
        model, norm, _, platt = get_fold_model(run_dir, fold, device_str)
        logit, alpha = attention_for(ga, rep.name, labels[idx], model, norm, device)
        prob = 1.0 / (1.0 + np.exp(-(platt[0] * logit + platt[1])))
        mode = f"fold {fold} 모델 (이 요소를 학습에 쓰지 않음)"
        truth = labels[idx]
    else:
        probs, alphas = [], []
        for f in range(1, args.folds + 1):
            model, norm, _, platt = get_fold_model(run_dir, f, device_str)
            logit, alpha = attention_for(ga, rep.name, 0, model, norm, device)
            probs.append(1.0 / (1.0 + np.exp(-(platt[0] * logit + platt[1]))))
            alphas.append(alpha)
        prob = float(np.mean(probs))
        alpha = np.mean(np.stack(alphas), axis=0)
        mode = f"{args.folds}-fold 앙상블 평균 (학습셋에 없는 파일)"
        truth = None

    imp = relative_importance(alpha)
    return dict(args=args, ga=ga, prob=prob, imp=imp, mode=mode, truth=truth)


def per_element_importance(rep: MeshRep, imp: np.ndarray, graph_kind: str) -> np.ndarray:
    """쉘 요소별 중요도(길이 n_shell). 절점기반 그래프면 인접 절점 중요도 평균으로 환산."""
    n_shell = len(rep.shells)
    if graph_kind == "elem":
        return imp[:n_shell]
    return np.array([
        np.mean([imp[rep.node_index[n]] for n in sh.nodes]) for sh in rep.shells
    ])


# ──────────────────────────────────────────────────────────────────────────
# 3D 메쉬 뷰
# ──────────────────────────────────────────────────────────────────────────
def build_mesh_figure(
    rep: MeshRep,
    ga,
    imp: np.ndarray,
    graph_kind: str,
    opacity: float,
    show_lines: bool,
    show_ext: bool,
    show_bad_outline: bool,
    show_shell_edges: bool,
    highlight: bool,
    threshold: float,
) -> go.Figure:
    """메쉬 3D 뷰.

    ``highlight=False`` 면 전 요소를 중립 회색으로 칠해 형상만 보여준다(초기 화면).
    ``highlight=True`` 면 중요도가 ``threshold`` 배 이상인 요소만 빨강 계열로,
    나머지는 회색 그대로 남겨 "이 요소 때문에 비정상"인 곳만 눈에 띄게 한다.
    색은 항상 빨강 한 계열(밝은→진한)만 쓴다 — 무지개색 컬러스케일은 쓰지 않는다.
    """
    fig = go.Figure()
    n_shell = len(rep.shells)
    shell_imp = per_element_importance(rep, imp, graph_kind)
    vmax = max(float(shell_imp.max()), threshold + 0.5, 1e-6)

    # ── 쉘 요소: 요소마다 정점을 따로 둬서 요소 단위로 균일하게 칠한다(faceted look) ──
    xs, ys, zs, ti, tj, tk, intensity, hover = [], [], [], [], [], [], [], []
    vtx = 0
    for si, sh in enumerate(rep.shells):
        pts = [rep.raw.coords[n] for n in sh.nodes]
        base = vtx
        for p in pts:
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        vtx += len(pts)

        tris = [(base, base + 1, base + 2)]
        if len(pts) == 4:
            tris.append((base, base + 2, base + 3))
        for a, b, c in tris:
            ti.append(a); tj.append(b); tk.append(c)

        val = float(shell_imp[si])
        q = sh.quality
        badge = "불량" if F.is_bad_shell(q) else "정상"
        txt = (
            f"{sh.etype} {sh.eid}<br>중요도 {val:.2f}x<br>"
            f"내각 {q.angle_min:.1f}°~{q.angle_max:.1f}° ({badge})<br>"
            f"aspect {q.aspect:.2f}  skew {q.skew:.1f}°"
        )
        for _ in pts:
            intensity.append(val if highlight else 0.0)
            hover.append(txt)

    mesh_kwargs = dict(
        x=xs, y=ys, z=zs, i=ti, j=tj, k=tk,
        flatshading=True, opacity=opacity,
        text=hover, hoverinfo="text", name="쉘 요소",
    )
    if highlight:
        gray_frac = float(np.clip(threshold / vmax, 0.02, 0.97))
        colorscale = [
            [0.0, NEUTRAL_SHELL], [gray_frac, NEUTRAL_SHELL],
            [min(gray_frac + 1e-3, 1.0), RED_LIGHT], [1.0, RED_DARK],
        ]
        fig.add_trace(go.Mesh3d(
            **mesh_kwargs, intensity=intensity, intensitymode="vertex",
            colorscale=colorscale, cmin=0, cmax=vmax, showscale=True,
            colorbar=dict(
                title="중요도(x평균)",
                tickvals=[0, threshold, vmax],
                ticktext=["평균 이하", f"{threshold:.1f}x (임계값)", f"{vmax:.1f}x"],
            ),
        ))
    else:
        fig.add_trace(go.Mesh3d(**mesh_kwargs, color=NEUTRAL_SHELL))

    # ── 쉘 경계선 — 옅은 회색. 인접한 쉘끼리 색이 비슷하면 뭉개져 보이므로,
    # 요소 하나하나의 경계를 항상 그어서 구분되게 한다 (진하게는 하지 않는다).
    if show_shell_edges:
        sx, sy, sz = [], [], []
        for e in rep.edge_shells:
            a, b = tuple(e)
            pa, pb = rep.raw.coords[a], rep.raw.coords[b]
            sx += [pa[0], pb[0], None]; sy += [pa[1], pb[1], None]; sz += [pa[2], pb[2], None]
        if sx:
            fig.add_trace(go.Scatter3d(
                x=sx, y=sy, z=sz, mode="lines",
                line=dict(color="#6b7280", width=1.5), opacity=0.55,
                name="쉘 경계", hoverinfo="skip",
            ))

    # ── 규칙 기반 불량각도 요소 테두리 (검정 — 빨강 강조와 구분되는 별도 신호) ──
    if show_bad_outline:
        ex, ey, ez = [], [], []
        for sh in rep.shells:
            if not F.is_bad_shell(sh.quality):
                continue
            pts = [rep.raw.coords[n] for n in sh.nodes] + [rep.raw.coords[sh.nodes[0]]]
            for p in pts:
                ex.append(p[0]); ey.append(p[1]); ez.append(p[2])
            ex.append(None); ey.append(None); ez.append(None)
        if ex:
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez, mode="lines",
                line=dict(color=BAD_OUTLINE, width=5),
                name="규칙 기반 불량각도 요소", hoverinfo="skip",
            ))

    # ── 1D 요소 (CBAR/CBEAM) — 선을 굵게, 끝점 마커를 찍어 연결을 명확히 ──
    if show_lines and rep.lines:
        has_line_imp = highlight and graph_kind == "elem" and ga.n_node > n_shell
        mx, my, mz, mcolor, mtext = [], [], [], [], []
        for li, ln in enumerate(rep.lines):
            a, b = rep.raw.coords[ln.nodes[0]], rep.raw.coords[ln.nodes[1]]
            if has_line_imp:
                v = float(imp[n_shell + li])
                if v >= threshold:
                    frac = float(np.clip((v - threshold) / max(vmax - threshold, 1e-6), 0, 1))
                    color = _blend(RED_LIGHT, RED_DARK, frac)
                    width = 5 + 4 * frac
                else:
                    color, width = NEUTRAL_LINE, 4
                txt = f"{ln.etype} {ln.eid}<br>길이 {ln.length:.1f}<br>중요도 {v:.2f}x"
            else:
                color, width = NEUTRAL_LINE, 4
                txt = f"{ln.etype} {ln.eid}<br>길이 {ln.length:.1f}"
            fig.add_trace(go.Scatter3d(
                x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
                mode="lines", line=dict(color=color, width=width),
                text=[txt, txt], hoverinfo="text", showlegend=False,
            ))
            for p in (a, b):
                mx.append(p[0]); my.append(p[1]); mz.append(p[2])
            mcolor += [color, color]
            mtext += [txt, txt]
        fig.add_trace(go.Scatter3d(
            x=mx, y=my, z=mz, mode="markers",
            marker=dict(size=3.5, color=mcolor),
            text=mtext, hoverinfo="text",
            name=f"1D 요소 (CBAR/CBEAM, {len(rep.lines)}개)",
        ))

    # ── 외부연결(ext) 절점 ────────────────────────────────────────────
    if show_ext:
        pts = [rep.raw.coords[n] for n in rep.node_ids if rep.ext.get(n, False)]
        if pts:
            P = np.array(pts)
            fig.add_trace(go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                marker=dict(size=4, color=EXT_MARKER, symbol="circle-open", line=dict(width=2)),
                name="외부연결 절점 (json ext=true)",
            ))

    _apply_layout(fig)
    return fig


def _blend(c1: str, c2: str, t: float) -> str:
    """두 hex 색을 t(0~1)로 선형 보간 — 강조 요소 안에서도 진하기 차등."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t); g = int(g1 + (g2 - g1) * t); b = int(b1 + (b2 - b1) * t)
    return f"rgb({r},{g},{b})"


def _apply_layout(fig: go.Figure) -> None:
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        height=720,
    )


def top_elements_table(
    rep: MeshRep, imp: np.ndarray, graph_kind: str, k: int = 10, only_shells: bool = False
):
    """중요도 상위 요소 표.

    ``only_shells=True`` 면 1D 요소(CBAR/CBEAM)를 순위에서 아예 뺀다. 1D 는 형상
    결함(각도/찌그러짐) 판정 대상이 아니라 보강재 위상 정보일 뿐이라 "판정" 칸이
    항상 "-"인데, 그런데도 어텐션 점수가 커서 종합 순위에서 1등을 차지하면
    "왜 비정상인지" 보여주려는 목적과 어긋난다. 쉘/1D 를 분리해서 보여주는 쪽을
    기본으로 쓴다 (아래 UI 에서 두 표로 나눠 호출).
    """
    n_shell = len(rep.shells)
    rows = []
    pool = np.arange(n_shell) if (graph_kind == "elem" and only_shells) else np.arange(len(imp))
    order = pool[np.argsort(-imp[pool])][:k]
    for i in order:
        if graph_kind == "elem" and i >= n_shell:
            ln = rep.lines[i - n_shell]
            rows.append({"중요도": round(float(imp[i]), 2), "종류": ln.etype,
                        "id": ln.eid, "내각/길이": f"길이 {ln.length:.1f}",
                        "판정": "-"})
        elif graph_kind == "elem":
            sh = rep.shells[i]
            q = sh.quality
            rows.append({"중요도": round(float(imp[i]), 2), "종류": sh.etype,
                        "id": sh.eid,
                        "내각/길이": f"{q.angle_min:.1f}°~{q.angle_max:.1f}°",
                        "판정": "불량" if F.is_bad_shell(q) else "정상"})
        else:
            nid = rep.node_ids[i]
            rows.append({"중요도": round(float(imp[i]), 2), "종류": "GRID",
                        "id": nid, "내각/길이": "-",
                        "판정": "ext" if rep.ext.get(nid, False) else "내부"})
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 그래프 구조 뷰 — 메쉬가 GNN 입력 그래프로 어떻게 바뀌었는지
# ──────────────────────────────────────────────────────────────────────────
def elem_graph_edges(rep: MeshRep) -> dict[str, list[tuple[int, int]]]:
    """요소기반 그래프의 관계 4종을 (그래프 노드 인덱스) 쌍 목록으로.

    graph_elem.build() 와 같은 인덱싱: 쉘은 0..n_shell-1, 1D 는 n_shell..n_shell+n_line-1.
    학습에 실제로 쓰이는 정규화된 피처와는 무관하게, 순수히 "무엇과 무엇이 연결됐는지"만
    다시 계산한다 — 시각화 전용이라 graph_elem 내부 구현에 의존하지 않는다.
    """
    n_shell = len(rep.shells)
    edges: dict[str, list[tuple[int, int]]] = {
        "shell_edge": [], "shell_corner": [], "line_shell": [], "line_line": [],
    }

    seen = set()
    for e, shells in rep.edge_shells.items():
        if len(shells) < 2:
            continue
        for a, b in combinations(sorted(set(shells)), 2):
            if (a, b) not in seen:
                seen.add((a, b))
                edges["shell_edge"].append((a, b))

    for nid, shells in rep.node_shells.items():
        for a, b in combinations(sorted(set(shells)), 2):
            if (a, b) not in seen:
                seen.add((a, b))
                edges["shell_corner"].append((a, b))

    for li, ln in enumerate(rep.lines):
        for si in rep.edge_shells.get(ln.edge, ()):
            edges["line_shell"].append((n_shell + li, si))

    seen_ll = set()
    for nid, lines in rep.node_lines.items():
        for a, b in combinations(sorted(set(lines)), 2):
            key = (n_shell + a, n_shell + b)
            if key not in seen_ll:
                seen_ll.add(key)
                edges["line_line"].append(key)

    return edges


REL_STYLE = {
    "shell_edge": ("쉘-쉘 변 공유", "#2563eb"),
    "shell_corner": ("쉘-쉘 꼭짓점 공유", "#9ca3af"),
    "line_shell": ("1D-쉘 연결", "#f97316"),
    "line_line": ("1D-1D 연결", "#dc2626"),
}


def build_graph_figure_elem(rep: MeshRep, show_rel: set[str]) -> go.Figure:
    """노드=요소(쉘/1D), 엣지=관계 4종. 원본 메쉬는 옅게 배경으로 깔아 대응관계를 보여준다."""
    fig = go.Figure()
    n_shell = len(rep.shells)

    # 배경: 원본 메쉬 윤곽 (옅게)
    ex, ey, ez = [], [], []
    for sh in rep.shells:
        pts = [rep.raw.coords[n] for n in sh.nodes] + [rep.raw.coords[sh.nodes[0]]]
        for p in pts:
            ex.append(p[0]); ey.append(p[1]); ez.append(p[2])
        ex.append(None); ey.append(None); ez.append(None)
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="#e5e7eb", width=1.5),
        name="원본 메쉬 (배경)", hoverinfo="skip",
    ))

    centroids = np.array(
        [s.centroid for s in rep.shells] + [l.centroid for l in rep.lines]
    ) if rep.lines else np.array([s.centroid for s in rep.shells])

    sc = centroids[:n_shell]
    fig.add_trace(go.Scatter3d(
        x=sc[:, 0], y=sc[:, 1], z=sc[:, 2], mode="markers",
        marker=dict(size=5, color="#1f2937"),
        text=[f"쉘 {s.etype} {s.eid}" for s in rep.shells], hoverinfo="text",
        name=f"그래프 노드 · 쉘 요소 ({n_shell}개)",
    ))
    if rep.lines:
        lc = centroids[n_shell:]
        fig.add_trace(go.Scatter3d(
            x=lc[:, 0], y=lc[:, 1], z=lc[:, 2], mode="markers",
            marker=dict(size=5, color="#f97316", symbol="diamond"),
            text=[f"1D {l.etype} {l.eid}" for l in rep.lines], hoverinfo="text",
            name=f"그래프 노드 · 1D 요소 ({len(rep.lines)}개)",
        ))

    edges = elem_graph_edges(rep)
    for key, (label, color) in REL_STYLE.items():
        if key not in show_rel:
            continue
        pairs = edges[key]
        if not pairs:
            continue
        lx, ly, lz = [], [], []
        for a, b in pairs:
            pa, pb = centroids[a], centroids[b]
            lx += [pa[0], pb[0], None]
            ly += [pa[1], pb[1], None]
            lz += [pa[2], pb[2], None]
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="lines",
            line=dict(color=color, width=2.5), opacity=0.75, hoverinfo="skip",
            name=f"엣지 · {label} ({len(pairs)}개)",
        ))

    _apply_layout(fig)
    return fig


def build_graph_figure_node(rep: MeshRep) -> go.Figure:
    """절점기반 그래프 — 노드=GRID 절점, 엣지=쉘 요소의 변."""
    fig = go.Figure()
    P = rep.points
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
        marker=dict(size=4, color="#1f2937"),
        text=[f"GRID {n}" for n in rep.node_ids], hoverinfo="text",
        name=f"그래프 노드 · 절점 ({len(P)}개)",
    ))
    ex, ey, ez = [], [], []
    for e in rep.edge_shells:
        a, b = tuple(e)
        pa, pb = rep.raw.coords[a], rep.raw.coords[b]
        ex += [pa[0], pb[0], None]; ey += [pa[1], pb[1], None]; ez += [pa[2], pb[2], None]
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="#2563eb", width=2.5), opacity=0.75, hoverinfo="skip",
        name=f"엣지 · 쉘 변 공유 ({len(rep.edge_shells)}개)",
    ))
    _apply_layout(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────
st.title("FE 메쉬 정상/비정상 뷰어")

runs = available_runs()
if not runs:
    st.error(f"학습된 실행이 없습니다. 먼저 `python src/train.py` 로 학습하세요. ({paths.RUNS_DIR})")
    st.stop()

with st.sidebar:
    st.header("설정")
    default_run = "final" if "final" in runs else runs[0]
    run_name = st.selectbox("모델 실행", runs, index=runs.index(default_run))
    run_dir = os.path.join(paths.RUNS_DIR, run_name)

    device_str = st.radio("연산 장치", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
                          index=0)

    st.divider()
    st.subheader("파일 선택")
    source = st.radio(
        "불러올 방식",
        ["샘플 폴더에서 선택", "오분류(FP/FN) 목록에서 선택", "직접 업로드"],
    )

    rep: MeshRep | None = None
    if source == "샘플 폴더에서 선택":
        samples = available_samples()
        query = st.text_input(
            "검색 (부분 문자열, 예: 0250)", key="search_sample",
            help="입력한 글자를 이름 어디에든 포함하는 파일만 드롭다운에 남습니다.",
        )
        filtered = [s for s in samples if query.lower() in s.lower()] if query else samples
        if not filtered:
            st.warning(f"'{query}' 를 포함하는 파일이 없습니다.")
        else:
            name = st.selectbox(f"모델 이름 ({len(filtered)}개)", filtered, index=0)
            rep = parse_sample(name)

    elif source == "오분류(FP/FN) 목록에서 선택":
        err_df = load_oof_errors(run_dir)
        if err_df.empty:
            st.warning(
                "이 실행에는 oof.csv 가 없습니다 — train.py 로 학습한 실행만 가능합니다."
            )
        else:
            kind = st.radio(
                "무엇을 볼지", ["FP (정상을 비정상으로 오판)", "FN (비정상을 정상으로 오판)", "둘 다"],
                help="FP=실제 정상인데 모델이 비정상이라 한 것. FN=실제 비정상인데 정상이라 한 것.",
            )
            want = {"FP (정상을 비정상으로 오판)": ["FP"],
                   "FN (비정상을 정상으로 오판)": ["FN"],
                   "둘 다": ["FP", "FN"]}[kind]
            sub = err_df[err_df["kind"].isin(want)].sort_values("conf", ascending=False)

            n_fp = int((err_df["kind"] == "FP").sum())
            n_fn = int((err_df["kind"] == "FN").sum())
            st.caption(f"전체 {len(err_df)}개 중 FP {n_fp}개 / FN {n_fn}개 "
                      f"(현재 선택 {len(sub)}개, 확신도 높은 오분류부터 정렬)")

            if sub.empty:
                st.info("해당하는 오분류 사례가 없습니다.")
            else:
                labels_ui = [
                    f"[{r.kind}] {r.name}  (실제 {'비정상' if r.y == 1 else '정상'}, "
                    f"예측확률 {r.prob:.3f})"
                    for r in sub.itertuples()
                ]
                query2 = st.text_input(
                    "검색 (이름 부분 문자열, 예: 0250)", key="search_err",
                )
                labels_ui_f = (
                    [l for l in labels_ui if query2.lower() in l.lower()]
                    if query2 else labels_ui
                )
                if not labels_ui_f:
                    st.warning(f"'{query2}' 를 포함하는 사례가 없습니다.")
                else:
                    pick = st.selectbox(
                        f"오분류 사례 ({len(labels_ui_f)}개)", labels_ui_f, index=0
                    )
                    sel_name = sub.iloc[labels_ui.index(pick)]["name"]
                    rep = parse_sample(sel_name)

    else:
        bdf_file = st.file_uploader(".bdf 파일", type=["bdf"])
        json_file = st.file_uploader(".json 파일 (선택 — 없으면 전 절점 외부연결=False)",
                                     type=["json"])
        if bdf_file is not None:
            name_hint = os.path.splitext(bdf_file.name)[0]
            rep = parse_from_bytes(
                bdf_file.getvalue(),
                json_file.getvalue() if json_file is not None else None,
                name_hint,
            )
            if json_file is None:
                st.warning("json 파일 없이 불러왔습니다 — 전 절점을 '외부연결 없음'으로 처리합니다.")

    st.divider()
    st.subheader("메쉬 표시")
    opacity = st.slider("쉘 투명도", 0.3, 1.0, 0.95)
    show_lines = st.checkbox("1D 요소(CBAR/CBEAM) 표시", value=True)
    show_ext = st.checkbox("외부연결 절점 표시", value=True)
    show_bad_outline = st.checkbox("규칙 기반 불량각도 요소 테두리", value=True)
    show_shell_edges = st.checkbox("쉘 경계선 표시 (요소끼리 구분)", value=True)

    st.divider()
    st.subheader("중요도 강조")
    highlight = st.checkbox("중요도 강조 켜기 (기본: 꺼짐 — 형상만 보기)", value=False)
    threshold = st.slider(
        "강조 임계값 (평균 대비 배수)", 1.0, 6.0, 2.0, 0.1,
        disabled=not highlight,
        help="이 배수 이상인 요소만 빨강으로 칠합니다. 나머지는 회색 그대로 둡니다.",
    )

if rep is None:
    st.info("왼쪽에서 모델을 선택하거나 .bdf 파일을 업로드하세요.")
    st.stop()

with st.spinner("예측 계산 중..."):
    result = predict(rep, run_dir, device_str)

col1, col2, col3 = st.columns(3)
col1.metric("예측", "비정상" if result["prob"] > 0.5 else "정상", f"p={result['prob']:.3f}")
if result["truth"] is not None:
    truth_str = "비정상" if result["truth"] == 1 else "정상"
    correct = (result["truth"] == 1) == (result["prob"] > 0.5)
    col2.metric("실제 라벨", truth_str, "일치" if correct else "불일치",
               delta_color="normal" if correct else "inverse")
else:
    lab = label_lookup().get(rep.name)
    if lab:
        cls, ambig = lab
        tag = {0: "정상", 1: "비정상", 2: "분류어려움"}[cls]
        col2.metric("라벨 엑셀 기록", tag, "애매" if ambig else "확실")
    else:
        col2.metric("실제 라벨", "정보 없음")
col3.metric("요소 수", f"{len(rep.shells)}쉘 + {len(rep.lines)}개 1D")

st.caption(f"평가 방식: {result['mode']}")

tab_mesh, tab_graph = st.tabs(["3D 메쉬", "그래프 구조"])

with tab_mesh:
    fig = build_mesh_figure(
        rep, result["ga"], result["imp"], result["args"].graph,
        opacity, show_lines, show_ext, show_bad_outline, show_shell_edges,
        highlight, threshold,
    )
    st.plotly_chart(fig, width="stretch", key="mesh_fig")

    st.subheader("중요도 상위 요소")
    st.caption(
        "1D 요소(CBAR/CBEAM)는 형상 결함 판정 대상이 아니라 보강재 위상 정보라서 "
        "'판정' 칸이 항상 '-' 입니다. **왜 비정상인지는 아래 '쉘' 표를 먼저 보세요.**"
    )
    col_shell, col_line = st.columns(2)
    with col_shell:
        st.markdown("**쉘 요소 (형상 결함 후보)**")
        st.dataframe(
            top_elements_table(rep, result["imp"], result["args"].graph, k=8, only_shells=True),
            width="stretch", hide_index=True,
        )
    with col_line:
        st.markdown("**1D 요소 (참고용 — 결함 판정 아님)**")
        if result["args"].graph == "elem" and rep.lines:
            full_ranked = top_elements_table(
                rep, result["imp"], result["args"].graph, k=len(result["imp"])
            )
            line_rows = [r for r in full_ranked if r["종류"] in ("CBAR", "CBEAM")][:8]
            st.dataframe(line_rows, width="stretch", hide_index=True)
        else:
            st.caption("이 실행은 1D 요소를 그래프에 포함하지 않습니다.")

    n_bad = sum(1 for s in rep.shells if F.is_bad_shell(s.quality))
    st.caption(
        f"불량 각도 요소(최소내각<30° 또는 최대내각>150°): {n_bad}개  ·  "
        f"외부연결 절점: {sum(1 for v in rep.ext.values() if v)}/{len(rep.ext)}"
    )

with tab_graph:
    st.caption(
        "이 메쉬가 GNN 에 들어가기 전에 어떤 그래프로 바뀌는지 보여줍니다. "
        f"현재 실행({run_name})은 **{result['args'].graph}** 그래프를 씁니다."
    )
    if result["args"].graph == "elem":
        st.markdown(
            "노드 = 요소(쉘 검정 원 / 1D 주황 다이아몬드), "
            "엣지 = 요소 사이의 연결 관계. 관계 종류별로 표시 여부를 고를 수 있습니다."
        )
        cols = st.columns(4)
        show_rel = set()
        defaults = {"shell_edge": True, "shell_corner": False,
                   "line_shell": True, "line_line": True}
        for c, (key, (label, color)) in zip(cols, REL_STYLE.items()):
            if c.checkbox(label, value=defaults[key], key=f"rel_{key}"):
                show_rel.add(key)
        fig2 = build_graph_figure_elem(rep, show_rel)
    else:
        st.markdown("노드 = GRID 절점, 엣지 = 쉘 요소의 변 — 원본 메쉬와 거의 같은 구조입니다.")
        fig2 = build_graph_figure_node(rep)

    st.plotly_chart(fig2, width="stretch", key="graph_fig")

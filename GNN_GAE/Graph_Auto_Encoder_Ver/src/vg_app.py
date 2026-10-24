"""VGAE 재생성 결과 화면 — 왼쪽 변경 전, 오른쪽 변경 후. 사람이 보고 판정을 남긴다.

    streamlit run GNN_GAE/Graph_Auto_Encoder_Ver/src/vg_app.py

    ┌──────────────────────────┬──────────────────────────┐
    │ 변경 전 (원본)             │ 변경 후 (모델 재생성)       │
    │ 빨강 = 지운 요소           │ 초록 = 새로 만든 요소       │
    │ 빨간 X = 없어진 절점       │                          │
    └──────────────────────────┴──────────────────────────┘
    편집 목록 / 모델별 비교 / 판정 (좋음·나쁨·애매 + 메모)

모델을 여기서 돌리지 않는다. vg_test.py(테스트)·vg_decode.py(학습)가 저장한 .bdf 를 읽고,
원본과 **좌표로** 비교해 지운 것·만든 것을 칠한다. 그래서 torch 없이 열린다.

    테스트 901개   out/test/<모델>/*.bdf, runs/test/<모델>/regen.csv    정답 없음 — 사람이 판정
    학습 508쌍     out/<모델>/*.bdf,      runs/<모델>/decode.csv         사람 결과(Good)도 볼 수 있음

판정은 runs/test/judgments.csv 에 (메쉬, 모델)마다 한 줄로 저장한다. 다시 저장하면 덮어쓴다.
"""

from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import vg_paths as vp  # noqa: E402
import features as F  # noqa: E402
from meshdiff import diff  # noqa: E402
from meshedit import EditableMesh  # noqa: E402

MODELS = ["vgae", "vgae_skip", "vgae_near", "vgae_near_l64",
          "gatv2_l64", "transformer_l64", "nnconv_l64"]
MODEL_TEXT = {
    "vgae": "vgae — GINE 기본",
    "vgae_skip": "vgae_skip — GINE, 디코더에 Bad 연결 더함",
    "vgae_near": "vgae_near — GINE, 두 칸 쌍 가중",
    "vgae_near_l64": "vgae_near_l64 — GINE, 두 칸 쌍 가중 + 잠재 64 + 800 epoch",
    "gatv2_l64": "gatv2_l64 — GATv2, 설정은 vgae_near_l64 와 같음",
    "transformer_l64": "transformer_l64 — TransformerConv, 설정 같음",
    "nnconv_l64": "nnconv_l64 — NNConv(ECC), 설정 같음",
}
TEST_DIR = os.path.join(vp.RUNS_DIR, "test")
JUDGE_CSV = os.path.join(TEST_DIR, "judgments.csv")
VERDICTS = ["좋음", "나쁨", "애매"]

SHELL = "#cfd8e3"
SHELL_EDGE = "#4a5568"
REMOVED = "#e53e3e"
CREATED = "#38a169"
LINE1D = "#2b6cb0"
BAD_OUTLINE = "#d69e2e"
EXT = "#805ad5"

st.set_page_config(page_title="VGAE 재생성 판정", layout="wide")


# ──────────────────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_sets():
    """{'테스트': {이름: (원본 RawMesh, None)}, '학습': {이름: (원본, Good)}} + 테스트 메타."""
    out = {"테스트": {}, "학습": {}}
    meta = pd.DataFrame()
    p = os.path.join(vp.CACHE_DIR, "test_node.pkl")
    if os.path.exists(p):
        with open(p, "rb") as fh:
            for d in pickle.load(fh):
                out["테스트"][d["name"]] = (d["before"], None)
    m = os.path.join(TEST_DIR, "test_meshes.csv")
    if os.path.exists(m):
        meta = pd.read_csv(m).set_index("name")
    if os.path.exists(vp.DATA_PATH):
        with open(vp.DATA_PATH, "rb") as fh:
            for d in pickle.load(fh):
                out["학습"][d["name"]] = (d["before"], d["after"])
    return out, meta


def _results_path(dataset: str, model: str) -> str:
    return (os.path.join(TEST_DIR, model, "regen.csv") if dataset == "테스트"
            else os.path.join(vp.RUNS_DIR, model, "decode.csv"))


@st.cache_data
def _read_results(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path).set_index("name")


def load_results(dataset: str, model: str) -> pd.DataFrame:
    """결과 표. 파일 수정 시각을 캐시 키에 넣어, 재생성을 다시 돌리면 새로 읽는다."""
    path = _results_path(dataset, model)
    if not os.path.exists(path):
        return pd.DataFrame()
    return _read_results(path, os.path.getmtime(path))


def result_path(dataset: str, model: str, name: str) -> str:
    base = os.path.join(vp.OUT_DIR, "test", model) if dataset == "테스트" else os.path.join(vp.OUT_DIR, model)
    return os.path.join(base, f"{name}.bdf")


def load_judgments() -> pd.DataFrame:
    if os.path.exists(JUDGE_CSV):
        return pd.read_csv(JUDGE_CSV, dtype=str).fillna("")
    return pd.DataFrame(columns=["name", "model", "verdict", "memo", "time"])


def save_judgment(name: str, model: str, verdict: str, memo: str) -> None:
    j = load_judgments()
    j = j[~((j.name == name) & (j.model == model))]
    row = pd.DataFrame([dict(name=name, model=model, verdict=verdict, memo=memo,
                             time=datetime.now().strftime("%Y-%m-%d %H:%M"))])
    os.makedirs(TEST_DIR, exist_ok=True)
    pd.concat([j, row], ignore_index=True).to_csv(JUDGE_CSV, index=False, encoding="utf-8-sig")


# ──────────────────────────────────────────────────────────────────────────
# 그리기
# ──────────────────────────────────────────────────────────────────────────
def _tris(m, idxs):
    xs, ys, zs, I, J, K = [], [], [], [], [], []
    v = 0
    for i in idxs:
        ns = m.shells[i][2]
        for n in ns:
            x, y, z = m.coords[n]
            xs.append(x); ys.append(y); zs.append(z)
        I.append(v); J.append(v + 1); K.append(v + 2)
        if len(ns) == 4:
            I.append(v); J.append(v + 2); K.append(v + 3)
        v += len(ns)
    return xs, ys, zs, I, J, K


def _mesh3d(m, idxs, color, hover, opacity=1.0):
    if not idxs:
        return None
    xs, ys, zs, I, J, K = _tris(m, idxs)
    # 면마다 같은 설명이 뜨도록 삼각형 단위 hover 를 만든다
    text = []
    for i in idxs:
        text += [hover(i)] * (2 if len(m.shells[i][2]) == 4 else 1)
    return go.Mesh3d(x=xs, y=ys, z=zs, i=I, j=J, k=K, color=color, opacity=opacity,
                     flatshading=True, hovertext=text, hoverinfo="text",
                     lighting=dict(ambient=0.85, diffuse=0.3))


def _segments(m, pairs, color, width, name=""):
    xs, ys, zs = [], [], []
    for a, b in pairs:
        pa, pb = m.coords[a], m.coords[b]
        xs += [pa[0], pb[0], None]; ys += [pa[1], pb[1], None]; zs += [pa[2], pb[2], None]
    if not xs:
        return None
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width),
                        hoverinfo="skip", name=name)


def _shell_edges(m, idxs):
    seen = set()
    for i in idxs:
        ns = m.shells[i][2]
        for a, b in zip(ns, ns[1:] + ns[:1]):
            seen.add((min(a, b), max(a, b)))
    return sorted(seen)


def figure(m, highlight: dict[int, str], color: str, *, lines_hi=(), gone_nodes=(),
           show_bad=False, show_ext=False, ranges=None, title=""):
    """메쉬 하나. ``highlight`` = {쉘 인덱스: 설명} 을 ``color`` 로 칠한다."""
    Q = [F.shell_quality(np.array([m.coords[n] for n in s[2]], dtype=float)) for s in m.shells]

    def hover(i):
        eid, t, ns = m.shells[i]
        q = Q[i]
        bad = " <b>불량각</b>" if F.is_bad_shell(q) else ""
        extra = f"<br><b>{highlight[i]}</b>" if i in highlight else ""
        return (f"<b>{t} {eid}</b>{bad}<br>내각 {q.angle_min:.1f}°~{q.angle_max:.1f}°"
                f"<br>node {', '.join(map(str, ns))}{extra}")

    base = [i for i in range(len(m.shells)) if i not in highlight]
    tr = [_mesh3d(m, base, SHELL, hover), _mesh3d(m, sorted(highlight), color, hover),
          _segments(m, _shell_edges(m, range(len(m.shells))), SHELL_EDGE, 2),
          _segments(m, [tuple(l[2]) for l in m.lines], LINE1D, 5, "1D")]
    if lines_hi:
        tr.append(_segments(m, lines_hi, color, 10))
    if show_bad:
        tr.append(_segments(m, _shell_edges(m, [i for i, q in enumerate(Q) if F.is_bad_shell(q)]),
                            BAD_OUTLINE, 6))
    if gone_nodes:
        P = np.array([m.coords[n] for n in gone_nodes])
        tr.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text",
                               marker=dict(size=7, color=REMOVED, symbol="x"),
                               text=[str(n) for n in gone_nodes], textposition="top center",
                               hovertext=[f"절점 {n} 없어짐" for n in gone_nodes], hoverinfo="text"))
    if show_ext:
        ext = [n for n in m.coords if m.ext.get(n, False)]
        if ext:
            P = np.array([m.coords[n] for n in ext])
            tr.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                                   marker=dict(size=4, color=EXT, symbol="circle-open"),
                                   hovertext=[f"ext 절점 {n}" for n in ext], hoverinfo="text"))
    fig = go.Figure([t for t in tr if t is not None])
    scene = dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                 zaxis=dict(visible=False))
    if ranges:
        scene["aspectmode"] = "cube" if ranges.get("_cube") else "data"
        for ax in ("xaxis", "yaxis", "zaxis"):
            scene[ax] = dict(visible=False, range=ranges[ax], autorange=False)
    fig.update_layout(scene=scene, showlegend=False, height=620,
                      margin=dict(l=0, r=0, t=30, b=0), title=dict(text=title, x=0.5),
                      uirevision="keep")
    return fig


def ranges_of(meshes, center=None, radius=None) -> dict:
    """두 화면이 같은 범위를 보도록. ``center`` 를 주면 그 둘레만."""
    if center is not None:
        c = np.asarray(center, dtype=float)
        return {ax: [float(c[i] - radius), float(c[i] + radius)]
                for i, ax in enumerate(("xaxis", "yaxis", "zaxis"))} | {"_cube": True}
    P = np.array([c for m in meshes for c in m.coords.values()], dtype=float)
    lo, hi = P.min(0), P.max(0)
    pad = 0.05 * np.maximum(hi - lo, 1.0)
    return {ax: [float(lo[i] - pad[i]), float(hi[i] + pad[i])]
            for i, ax in enumerate(("xaxis", "yaxis", "zaxis"))}


def op_text(op: str, gain: str | None) -> str:
    kind, key = op.split(":", 1)
    g = f"  (점수 +{float(gain):.2f})" if gain else ""
    if kind == "collapse":
        r, k = key.split(">")
        return f"절점 병합 — 절점 {r} 을 {k} 에 합침 ({r} 없어짐){g}"
    a, b = key.split("-")
    return f"삼각형 쌍 병합 — 공유 변 {a}–{b} 를 없애 사각형 하나로{g}"


def _ops(row) -> list[tuple[str, str | None]]:
    ops = row.get("ops")
    if not isinstance(ops, str) or not ops:
        return []
    gains = str(row.get("gains", "")).split(";") if isinstance(row.get("gains"), str) else []
    return [(o, gains[i] if i < len(gains) else None) for i, o in enumerate(ops.split(";"))]


# ──────────────────────────────────────────────────────────────────────────
# 화면
# ──────────────────────────────────────────────────────────────────────────
sets, meta = load_sets()
st.sidebar.title("VGAE 재생성 판정")
dataset = st.sidebar.radio("데이터", ["테스트", "학습"], horizontal=True,
                           help="테스트 = 0918 확실·비정상 중 학습에 안 쓴 901개 (정답 없음). "
                                "학습 = 학습에 쓴 508쌍 (사람 결과 있음).")
model = st.sidebar.selectbox("모델", MODELS, index=MODELS.index("vgae_near_l64"),
                             format_func=lambda m: MODEL_TEXT[m])
res = load_results(dataset, model)
names = sorted(sets[dataset])
if not names or res.empty:
    st.warning(f"{dataset} 결과가 없습니다. 먼저 "
               + ("`python src/vg_test.py`" if dataset == "테스트" else f"`python src/vg_decode.py --run {model}`")
               + " 를 실행하세요.")
    st.stop()

judg = load_judgments()
st.sidebar.markdown("**목록 거르기**")
only_edit = st.sidebar.checkbox("편집한 메쉬만", value=True)
if dataset == "테스트":
    judged = set(judg[judg.model == model].name)
    jfilter = st.sidebar.radio("판정", ["전부", "미판정만", "판정한 것만"], horizontal=True)
    max_ov = st.sidebar.slider("학습 메쉬와 겹치는 요소 비율 최대", 0.0, 1.0, 1.0, 0.05,
                               help="테스트 메쉬의 요소 중 학습 508쌍 원본에도 있는 요소의 비율. "
                                    "낮을수록 모델이 처음 보는 자리.")
else:
    exact_f = st.sidebar.radio("사람 결과와", ["전부", "일치만", "불일치만"], horizontal=True)

sel = [n for n in names if n in res.index]
if only_edit:
    sel = [n for n in sel if int(res.loc[n, "n_ops"]) > 0]
if dataset == "테스트":
    if jfilter == "미판정만":
        sel = [n for n in sel if n not in judged]
    elif jfilter == "판정한 것만":
        sel = [n for n in sel if n in judged]
    if not meta.empty:
        sel = [n for n in sel if float(meta.loc[n, "overlap_train"]) <= max_ov + 1e-9]
else:
    if exact_f != "전부":
        sel = [n for n in sel if bool(res.loc[n, "exact"]) == (exact_f == "일치만")]
if not sel:
    st.info("조건에 맞는 메쉬가 없습니다.")
    st.stop()

if "idx" not in st.session_state:
    st.session_state.idx = 0
st.session_state.idx = min(st.session_state.idx, len(sel) - 1)
c1, c2, c3 = st.sidebar.columns([1, 2, 1])
if c1.button("◀"):
    st.session_state.idx = max(0, st.session_state.idx - 1)
if c3.button("▶"):
    st.session_state.idx = min(len(sel) - 1, st.session_state.idx + 1)
c2.markdown(f"<div style='text-align:center;padding-top:6px'>{st.session_state.idx + 1} / {len(sel)}</div>",
            unsafe_allow_html=True)
name = st.sidebar.selectbox("메쉬", sel, index=st.session_state.idx)
st.session_state.idx = sel.index(name)

show_bad = st.sidebar.checkbox("불량각 요소 테두리 (노랑)", value=False)
show_ext = st.sidebar.checkbox("ext 절점 (보라)", value=False)
if dataset == "테스트":
    n_j = judg[judg.model == model].verdict.value_counts()
    st.sidebar.caption(f"이 모델 판정: " + ", ".join(f"{v} {int(n_j.get(v, 0))}" for v in VERDICTS)
                       + f" / 테스트 {len(names)}개")

# ── 메쉬 불러오기, 차이 계산 ─────────────────────────────────────────────
bad_raw, good_raw = sets[dataset][name]
bad = EditableMesh(bad_raw)
rp = result_path(dataset, model, name)
after = EditableMesh.load(rp, rp[:-4] + ".json")
dd = diff(bad, after)
rm = {i: "지운 요소" for i in dd.removed_shells}
cr = {i: "새로 만든 요소" for i in dd.created_shells}
rm_lines = [tuple(bad.lines[i][2]) for i in dd.removed_lines]
cr_lines = [tuple(after.lines[i][2]) for i in dd.created_lines]
gone = sorted(dd.deleted_nodes)
row = res.loc[name]
ops = _ops(row)

# ── 머리 ────────────────────────────────────────────────────────────────
st.markdown(f"### {name}")
info = [f"모델 **{model}**", f"편집 **{len(ops)}개**",
        f"요소 {len(bad.shells)} → {len(after.shells)}",
        f"삼각형 {sum(len(s[2]) == 3 for s in bad.shells)} → {sum(len(s[2]) == 3 for s in after.shells)}"]
if "n_bad_before" in row:
    info.append(f"불량각 요소 {int(row.n_bad_before)} → {int(row.n_bad_after)}")
if dataset == "테스트" and not meta.empty and name in meta.index:
    info.append(f"학습 메쉬와 겹침 {float(meta.loc[name, 'overlap_train']):.0%}")
if dataset == "학습":
    info.append("사람 결과와 **일치**" if bool(row.exact) else "사람 결과와 **불일치**")
st.markdown(" · ".join(info))

# ── 확대 위치 ───────────────────────────────────────────────────────────
spots = [("전체", None)]
for i in dd.removed_shells:
    spots.append((f"지운 요소 {bad.shells[i][0]}", bad.corners(i).mean(0)))
for n in gone:
    spots.append((f"없어진 절점 {n}", bad.P(n)))
zoom = st.selectbox("확대 위치 (두 화면 같이)", range(len(spots)), format_func=lambda k: spots[k][0])
if spots[zoom][1] is None:
    rng = ranges_of([bad, after])
else:
    med = float(np.median([np.linalg.norm(bad.P(a) - bad.P(b)) for a, b in _shell_edges(bad, range(len(bad.shells)))]))
    rng = ranges_of([], center=spots[zoom][1], radius=3.5 * med)

# ── 두 화면 ─────────────────────────────────────────────────────────────
show_human = dataset == "학습" and st.checkbox("사람 결과(Good)도 보기", value=False)
cols = st.columns(3 if show_human else 2)
with cols[0]:
    st.plotly_chart(figure(bad, rm, REMOVED, lines_hi=rm_lines, gone_nodes=gone, show_bad=show_bad,
                           show_ext=show_ext, ranges=rng, title="변경 전 (원본)"),
                    width="stretch", key="before")
    st.caption(f"빨강 = 지운 요소 {len(rm)}개 · 빨간 X = 없어진 절점 {len(gone)}개"
               + (f" · 굵은 빨강 선 = 지운 1D {len(rm_lines)}개" if rm_lines else ""))
with cols[1]:
    st.plotly_chart(figure(after, cr, CREATED, lines_hi=cr_lines, show_bad=show_bad, show_ext=show_ext,
                           ranges=rng, title=f"변경 후 ({model})"),
                    width="stretch", key="after")
    st.caption(f"초록 = 새로 만든 요소 {len(cr)}개" + (f" · 굵은 초록 선 = 새 1D {len(cr_lines)}개" if cr_lines else ""))
if show_human:
    good = EditableMesh(good_raw)
    dg = diff(bad, good)
    with cols[2]:
        st.plotly_chart(figure(good, {i: "사람이 만든 요소" for i in dg.created_shells}, CREATED,
                               show_bad=show_bad, show_ext=show_ext, ranges=rng, title="사람 결과 (Good)"),
                        width="stretch", key="human")

st.caption("두 화면은 같은 축 범위로 그린다. 회전은 화면마다 따로 된다 — 다시 맞추려면 '확대 위치' 를 바꾸거나 "
           "화면 오른쪽 위 집 모양 버튼을 누른다.")

# ── 편집 목록 ───────────────────────────────────────────────────────────
left, right = st.columns([3, 2])
with left:
    st.markdown("#### 적용한 편집 (순서대로)")
    if ops:
        for k, (o, g) in enumerate(ops, 1):
            st.markdown(f"{k}. {op_text(o, g)}")
    else:
        st.markdown("편집 없음 — 모델이 고칠 곳이 없다고 봤거나, 고르려던 편집이 검사에서 막혔다.")

    st.markdown("#### 모델 비교 (이 메쉬)")
    comp = []
    for mdl in MODELS:
        r = load_results(dataset, mdl)
        if name not in r.index:
            continue
        rr = r.loc[name]
        jj = judg[(judg.name == name) & (judg.model == mdl)]
        comp.append({
            "모델": mdl, "편집 수": int(rr.n_ops), "절점 병합": int(rr.n_collapse), "삼각형 쌍": int(rr.n_merge),
            **({"불량각 후": int(rr.n_bad_after)} if "n_bad_after" in rr else {}),
            **({"사람과 일치": bool(rr.exact)} if dataset == "학습" else {}),
            **({"판정": jj.verdict.iloc[0] if len(jj) else ""} if dataset == "테스트" else {}),
            "편집": str(rr.ops) if isinstance(rr.ops, str) else "",
        })
    st.dataframe(pd.DataFrame(comp), hide_index=True, width="stretch")

# ── 판정 ────────────────────────────────────────────────────────────────
with right:
    if dataset == "테스트":
        st.markdown(f"#### 판정 — {model}")
        cur = judg[(judg.name == name) & (judg.model == model)]
        cur_v = cur.verdict.iloc[0] if len(cur) else None
        cur_m = cur.memo.iloc[0] if len(cur) else ""
        with st.form(f"judge_{name}_{model}"):
            v = st.radio("이 재생성 결과는", VERDICTS, horizontal=True,
                         index=VERDICTS.index(cur_v) if cur_v in VERDICTS else None)
            memo = st.text_input("메모 (선택)", value=cur_m)
            go_next = st.checkbox("저장하고 다음 메쉬로", value=True)
            if st.form_submit_button("판정 저장"):
                if v is None:
                    st.warning("좋음 / 나쁨 / 애매 중 하나를 고르세요.")
                else:
                    save_judgment(name, model, v, memo)
                    # '미판정만' 이면 방금 판정한 메쉬가 목록에서 빠지므로 같은 번호가 곧 다음 메쉬다
                    if go_next and jfilter != "미판정만":
                        st.session_state.idx = min(len(sel) - 1, st.session_state.idx + 1)
                    st.rerun()
        if cur_v:
            st.caption(f"저장된 판정: {cur_v} ({cur.time.iloc[0]})")
        st.caption(f"저장 위치: {JUDGE_CSV}")
    else:
        st.markdown("#### 학습 데이터")
        st.markdown("학습에 쓴 쌍이라 판정은 받지 않는다. 사람 결과와의 일치 여부만 표시한다.")

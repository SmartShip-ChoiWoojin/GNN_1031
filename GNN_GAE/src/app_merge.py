"""재생성 결과 뷰어 — 모델이 고친 메쉬와 사람이 고친 메쉬를 나란히 본다.

`merge_gnn.py` 가 저장한 out-of-fold 예측(`out/merge_pred.csv`)을 읽는다. 각 메쉬의
점수는 **그 메쉬를 학습에 쓰지 않은** 모델이 매긴 것이다.

    python src/merge_gnn.py              # 예측을 먼저 만든다
    streamlit run GNN_GAE/src/app_merge.py

왼쪽이 모델, 오른쪽이 사람. 둘 다 같은 원본에서 출발하므로 초록(새로 생긴 사각형)의
위치가 다르면 모델이 다른 곳을 합친 것이다.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import gae_paths as gp
from bdf_io import discover_pairs
from fasoo_drm import setting_fasoo
from meshedit import EditableMesh, EditError, merge_candidates
from meshrep import Edge
from pairs import discover_good_pairs, load_pair
from viz_compare import render_html, stats

setting_fasoo()
st.set_page_config(page_title="재생성 결과 뷰어", layout="wide")

PRED = os.path.join(gp.OUT_DIR, "merge_pred.csv")


def show_table(df, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def show_html(html: str, height: int) -> None:
    fn = getattr(st, "iframe", None)
    if fn is not None:
        fn(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)


@st.cache_data
def load_pred() -> pd.DataFrame | None:
    return pd.read_csv(PRED) if os.path.exists(PRED) else None


@st.cache_data
def path_maps():
    return discover_good_pairs(), discover_pairs(gp.FE_DIR)[0]


def apply_merges(base: EditableMesh, edges: list[Edge]) -> tuple[EditableMesh, list[str]]:
    """주어진 변들을 차례로 합친다. 합칠 때마다 쉘 번호가 밀리므로 매번 다시 찾는다."""
    m = base.clone()
    notes = []
    for e in edges:
        hit = next((c for c in merge_candidates(m, drop_lines=True)
                    if Edge(tuple(set(m.shells[c.si][2]) & set(m.shells[c.sj][2]))) == e), None)
        if hit is None:
            notes.append(f"{sorted(e)}: 후보가 아니어서 건너뜀")
            continue
        try:
            m.merge_tris(hit.si, hit.sj, drop_lines=True)
        except EditError as exc:
            notes.append(f"{sorted(e)}: {exc}")
    return m, notes


@st.cache_data(show_spinner=False)
def build_view(name: str, ver: float) -> dict:
    """모델 결과 / 사람 결과를 만들어 뷰어 HTML 과 요약을 돌려준다."""
    good, orig = path_maps()
    p = load_pair(name, good, orig)
    base = EditableMesh(p.before)

    d = load_pred()
    g = d[d.name == name].copy()
    pick_e = [Edge(tuple(int(x) for x in s.split("-"))) for s in g[g.picked == 1].edge]
    truth_e = [Edge(tuple(int(x) for x in s.split("-"))) for s in g[g.truth == 1].edge]

    model_mesh, notes = apply_merges(base, pick_e)
    human_mesh = EditableMesh(p.after)

    bkeys = {frozenset(s[2]) for s in base.shells}
    mk = {frozenset(s[2]) for s in model_mesh.shells} - bkeys
    hk = {frozenset(s[2]) for s in human_mesh.shells} - bkeys

    html = render_html(
        model_mesh, human_mesh, [], "", embed=True,
        node_marks_before={}, node_marks_after={},
    )
    # render_html 은 (전, 후) 의 차이를 칠한다. 여기서는 양쪽 다 '원본 대비 새로 생긴
    # 것' 을 보여주고 싶으므로 직접 만든 마크를 쓴다 — 아래 figure 를 다시 만든다.
    from viz_compare import ADDED, mesh_traces, _common_ranges, _layout
    import plotly.graph_objects as go
    import json
    from plotly.utils import PlotlyJSONEncoder
    from plotly.offline import get_plotlyjs
    import viz_compare as V

    ranges = _common_ranges(model_mesh, human_mesh)
    figL = go.Figure(data=mesh_traces(model_mesh, mk, ADDED, "모델이 새로 만든 사각형"),
                     layout=_layout("모델", ranges))
    figR = go.Figure(data=mesh_traces(human_mesh, hk, ADDED, "사람이 만든 사각형"),
                     layout=_layout("사람", ranges))
    html = V._PAGE.format(
        title=f"모델 vs 사람 — {name}",
        plotlyjs=get_plotlyjs(),
        figL=json.dumps(figL, cls=PlotlyJSONEncoder),
        figR=json.dumps(figR, cls=PlotlyJSONEncoder),
        groups=json.dumps({x: i for i, x in enumerate(V.GROUPS)}),
        header_html="<style>.plot { height:calc(100vh - 62px); }</style>",
        footer_html="", tableL="", tableR="",
        bad=V.BAD_OUTLINE, line=V.NEUTRAL_LINE, blocker=V.BLOCKER,
        ext=V.EXT_MARKER, chnode=V.CHANGED_NODE, mvnode=V.MOVED_NODE,
    )

    same = {frozenset(x[2]) for x in model_mesh.shells} == \
           {frozenset(x[2]) for x in human_mesh.shells}
    return dict(
        html=html, same=same, notes=notes,
        n_pick=len(pick_e), n_truth=len(truth_e),
        before=stats(base), model=stats(model_mesh), human=stats(human_mesh),
        table=g[["edge", "score", "picked", "truth"]].sort_values("score", ascending=False),
        n_choice=int(g.n_choice.iloc[0]), thr=float(g.thr.iloc[0]), fold=int(g.fold.iloc[0]),
    )


def render_ver() -> float:
    import viz_compare

    return max(os.path.getmtime(viz_compare.__file__), os.path.getmtime(PRED))


# ──────────────────────────────────────────────────────────────────────────
st.title("재생성 결과 뷰어")
st.caption("왼쪽이 모델이 고친 결과, 오른쪽이 사람이 고친 결과. 초록 = 원본에 없던 사각형. "
           "시점은 연동됩니다.")

d = load_pred()
if d is None:
    st.error(f"예측 파일이 없습니다: {PRED}\n\n먼저 `python src/merge_gnn.py` 를 돌리세요.")
    st.stop()

# 메쉬별 성패를 미리 계산해 목록에 쓴다
per = d.groupby("name").agg(
    n_cand=("edge", "size"), n_truth=("truth", "sum"), n_pick=("picked", "sum"),
    n_choice=("n_choice", "first"), fold=("fold", "first")).reset_index()
# groupby.apply 는 pandas 버전에 따라 인덱스가 이름이 아니라 정수로 나온다. 명시적으로 돈다.
hit = {n: set(g[g.picked == 1].edge) == set(g[g.truth == 1].edge)
       for n, g in d.groupby("name")}
per["맞음"] = per.name.map(hit)

with st.sidebar:
    st.header("메쉬 고르기")
    scope = st.radio("보기", ["틀린 것만", "맞은 것만", "전체"], index=0,
                     help="'틀린 것' 을 먼저 보시면 모델이 어디서 무너지는지 보입니다.")
    only_real = st.checkbox("실제로 골라야 하는 것만", value=True,
                            help="후보=정답이라 선택의 여지가 없는 메쉬를 뺍니다.")
    sub = per
    if only_real:
        sub = sub[sub.n_choice > 1]
    if scope == "틀린 것만":
        sub = sub[~sub["맞음"]]
    elif scope == "맞은 것만":
        sub = sub[sub["맞음"]]
    st.caption(f"{len(sub)} / {len(per)}개")
    if sub.empty:
        st.warning("해당하는 메쉬가 없습니다.")
        st.stop()
    sub = sub.sort_values(["n_choice", "name"], ascending=[False, True])
    # r.name 은 'name' 컬럼이 아니라 **인덱스**다 (pandas Series.name). r["name"] 을 쓴다.
    label = {r["name"]: f"{r['name']}  후보{r.n_cand}/정답{r.n_truth}/선택{r.n_pick}"
                        f"  {'O' if r['맞음'] else 'X'}"
             for _, r in sub.iterrows()}
    pick_lbl = st.selectbox(f"메쉬 ({len(sub)}개)", list(label.values()))
    name = [k for k, v in label.items() if v == pick_lbl][0]

    st.divider()
    ok = int(per[per.n_choice > 1]["맞음"].sum())
    tot = int((per.n_choice > 1).sum())
    st.metric("실제 선택분 정확도", f"{ok}/{tot}", f"{ok/max(1,tot):.1%}")

with st.spinner("재생성하는 중..."):
    v = build_view(name, render_ver())

st.markdown(f"**{name}** · fold {v['fold']} · 임계값 {v['thr']:.2f} · "
            f"선택지 {v['n_choice']}가지 · "
            + ("✅ 사람과 **일치**" if v["same"] else "❌ 사람과 **다름**"))

c = st.columns(4)
c[0].metric("모델 선택", f"{v['n_pick']}개", delta=v["n_pick"] - v["n_truth"],
            help="정답 개수와의 차이")
c[1].metric("삼각형 (원본)", v["before"]["삼각형"])
c[2].metric("삼각형 (모델)", v["model"]["삼각형"],
            delta=v["model"]["삼각형"] - v["before"]["삼각형"], delta_color="inverse")
c[3].metric("삼각형 (사람)", v["human"]["삼각형"],
            delta=v["human"]["삼각형"] - v["before"]["삼각형"], delta_color="inverse")

if v["notes"]:
    st.warning("적용하지 못한 선택: " + " / ".join(v["notes"]))

show_html(v["html"], height=660)

left, right = st.columns([3, 2])
with left:
    st.subheader("후보별 모델 점수")
    t = v["table"].copy()
    t["판정"] = np.where((t.picked == 1) & (t.truth == 1), "맞음",
                np.where((t.picked == 1) & (t.truth == 0), "잘못 골랐음",
                np.where((t.picked == 0) & (t.truth == 1), "놓쳤음", "-")))
    t["score"] = t.score.round(3)
    show_table(t[["edge", "score", "picked", "truth", "판정"]].rename(
        columns={"edge": "변", "score": "모델 점수", "picked": "모델 선택", "truth": "정답"}),
        hide_index=True)
with right:
    st.subheader("전/후 지표")
    show_table(pd.DataFrame([
        {"지표": k, "원본": str(v["before"][k]), "모델": str(v["model"][k]),
         "사람": str(v["human"][k])} for k in v["before"]]), hide_index=True)

with st.expander("이 화면을 읽는 법"):
    st.markdown(
        """
- **왼쪽 = 모델, 오른쪽 = 사람.** 둘 다 같은 원본에서 출발합니다.
- **초록** = 원본에 없던 사각형. 초록 위치가 서로 다르면 모델이 다른 곳을 합친 것입니다.
- 아래 표의 **판정** 열:
  - `맞음` — 사람도 합친 변
  - `잘못 골랐음` — 모델만 합친 변
  - `놓쳤음` — 사람은 합쳤는데 모델이 안 고른 변
- 점수는 **out-of-fold** 입니다. 각 메쉬는 자기를 학습에 쓰지 않은 모델이 채점했습니다.
"""
    )

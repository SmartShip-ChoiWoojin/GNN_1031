"""삼각형 병합 결과 뷰어(초기 시도) — 모델이 고친 메쉬와 사람이 고친 메쉬를 나란히 본다.

`merge_gnn.py` 가 저장한 out-of-fold 예측(`out/merge_pred.csv`)을 읽는다. 각 메쉬의
점수는 **그 메쉬를 학습에 쓰지 않은** 모델이 매긴 것이다.

    python src/merge_gnn.py              # 예측을 먼저 만든다
    streamlit run GNN_GAE_CHOI/src/app_merge.py

왼쪽이 모델, 오른쪽이 사람. 둘 다 같은 원본에서 출발하므로 초록(새로 생긴 사각형)의
위치가 다르면 모델이 다른 곳을 합친 것이다.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

import gae_paths
import viz_compare
from bdf_io import discover_pairs
from fasoo_drm import setting_fasoo
from meshedit import EditableMesh, EditError, merge_candidates, shared_edge_of
from meshrep import Edge
from pairs import discover_good_pairs, load_pair

setting_fasoo()
st.set_page_config(page_title="재생성 결과 뷰어", layout="wide")


def prediction_path():
    """예측 파일 — 이 폴더 out/ 에 없으면 원래 GNN_GAE/out 의 것."""
    own_path = os.path.join(gae_paths.OUT_DIR, "merge_pred.csv")
    if os.path.exists(own_path):
        return own_path
    return os.path.join(gae_paths.ORIGINAL_GAE_ROOT, "out", "merge_pred.csv")


PREDICTION_PATH = prediction_path()


def show_table(table, **options) -> None:
    try:
        st.dataframe(table, width="stretch", **options)
    except TypeError:
        st.dataframe(table, use_container_width=True, **options)


def show_html(page_html: str, height: int) -> None:
    iframe_function = getattr(st, "iframe", None)
    if iframe_function is not None:
        iframe_function(page_html, height=height)
    else:
        components.html(page_html, height=height, scrolling=False)


@st.cache_data
def load_predictions():
    if os.path.exists(PREDICTION_PATH):
        return pd.read_csv(PREDICTION_PATH)
    return None


@st.cache_data
def path_maps():
    return discover_good_pairs(), discover_pairs(gae_paths.FE_DIR)[0]


def parse_edge_text(text: str) -> frozenset:
    """"a-b" → Edge."""
    node_ids = []
    for node_text in text.split("-"):
        node_ids.append(int(node_text))
    return Edge(tuple(node_ids))


def apply_merges_to_copy(base_mesh: EditableMesh, edges: list):
    """주어진 변들을 차례로 합친 사본과 못 합친 사유. 합칠 때마다 쉘 번호가 밀리므로 매번 다시 찾는다."""
    mesh = base_mesh.clone()
    notes = []
    for edge in edges:
        found = None
        for candidate in merge_candidates(mesh, drop_lines=True):
            if shared_edge_of(mesh, candidate.si, candidate.sj) == edge:
                found = candidate
                break
        if found is None:
            notes.append(f"{sorted(edge)}: 후보가 아니어서 건너뜀")
            continue
        try:
            mesh.merge_tris(found.si, found.sj, drop_lines=True)
        except EditError as error:
            notes.append(f"{sorted(edge)}: {error}")
    return mesh, notes


def shell_key_set(mesh) -> set:
    keys = set()
    for shell in mesh.shells:
        keys.add(frozenset(shell[2]))
    return keys


@st.cache_data(show_spinner=False)
def build_view(name: str, version: float) -> dict:
    """모델 결과 / 사람 결과를 만들어 뷰어 HTML 과 요약을 돌려준다."""
    good_paths, original_paths = path_maps()
    pair = load_pair(name, good_paths, original_paths)
    base_mesh = EditableMesh(pair.before)

    predictions = load_predictions()
    mesh_rows = predictions[predictions.name == name].copy()
    picked_edges = []
    for edge_text in mesh_rows[mesh_rows.picked == 1].edge:
        picked_edges.append(parse_edge_text(edge_text))
    truth_edges = []
    for edge_text in mesh_rows[mesh_rows.truth == 1].edge:
        truth_edges.append(parse_edge_text(edge_text))

    model_mesh, notes = apply_merges_to_copy(base_mesh, picked_edges)
    human_mesh = EditableMesh(pair.after)

    # 양쪽 다 '원본 대비 새로 생긴 것' 을 칠한다
    base_keys = shell_key_set(base_mesh)
    model_new = shell_key_set(model_mesh) - base_keys
    human_new = shell_key_set(human_mesh) - base_keys

    ranges = viz_compare.common_ranges(model_mesh, human_mesh)
    figure_left = go.Figure(data=viz_compare.mesh_traces(model_mesh, model_new, viz_compare.ADDED,
                                                         "모델이 새로 만든 사각형"),
                            layout=viz_compare.scene_layout("모델", ranges))
    figure_right = go.Figure(data=viz_compare.mesh_traces(human_mesh, human_new, viz_compare.ADDED,
                                                          "사람이 만든 사각형"),
                             layout=viz_compare.scene_layout("사람", ranges))
    page_html = viz_compare.PAGE_TEMPLATE.format(
        title=f"모델 vs 사람 — {name}",
        plotlyjs=get_plotlyjs(),
        figL=json.dumps(figure_left, cls=PlotlyJSONEncoder),
        figR=json.dumps(figure_right, cls=PlotlyJSONEncoder),
        groups=viz_compare.group_index_json(),
        header_html="<style>.plot { height:calc(100vh - 62px); }</style>",
        footer_html="", tableL="", tableR="",
        bad=viz_compare.BAD_OUTLINE, line=viz_compare.NEUTRAL_LINE, blocker=viz_compare.BLOCKER,
        ext=viz_compare.EXT_MARKER, chnode=viz_compare.CHANGED_NODE, mvnode=viz_compare.MOVED_NODE,
    )

    same_as_human = shell_key_set(model_mesh) == shell_key_set(human_mesh)
    return dict(
        html=page_html, same=same_as_human, notes=notes,
        n_pick=len(picked_edges), n_truth=len(truth_edges),
        before=viz_compare.stats(base_mesh), model=viz_compare.stats(model_mesh),
        human=viz_compare.stats(human_mesh),
        table=mesh_rows[["edge", "score", "picked", "truth"]].sort_values("score", ascending=False),
        n_choice=int(mesh_rows.n_choice.iloc[0]), thr=float(mesh_rows.thr.iloc[0]),
        fold=int(mesh_rows.fold.iloc[0]),
    )


def render_version() -> float:
    return max(os.path.getmtime(os.path.join(gae_paths.SRC_DIR, "viz_compare.py")),
               os.path.getmtime(PREDICTION_PATH))


# ──────────────────────────────────────────────────────────────────────────
st.title("재생성 결과 뷰어")
st.caption("왼쪽이 모델이 고친 결과, 오른쪽이 사람이 고친 결과. 초록 = 원본에 없던 사각형. 시점은 연동됩니다.")

predictions = load_predictions()
if predictions is None:
    st.error(f"예측 파일이 없습니다: {PREDICTION_PATH}\n\n먼저 `python src/merge_gnn.py` 를 돌리세요.")
    st.stop()

# 메쉬별 성패를 미리 계산해 목록에 쓴다
per_mesh = predictions.groupby("name").agg(
    n_cand=("edge", "size"), n_truth=("truth", "sum"), n_pick=("picked", "sum"),
    n_choice=("n_choice", "first"), fold=("fold", "first")).reset_index()
is_correct = {}
for name, mesh_rows in predictions.groupby("name"):
    is_correct[name] = set(mesh_rows[mesh_rows.picked == 1].edge) == set(mesh_rows[mesh_rows.truth == 1].edge)
per_mesh["맞음"] = per_mesh.name.map(is_correct)

with st.sidebar:
    st.header("메쉬 고르기")
    scope = st.radio("보기", ["틀린 것만", "맞은 것만", "전체"], index=0,
                     help="'틀린 것' 을 먼저 보시면 모델이 어디서 무너지는지 보입니다.")
    only_real_choice = st.checkbox("실제로 골라야 하는 것만", value=True,
                                   help="후보=정답이라 선택의 여지가 없는 메쉬를 뺍니다.")
    shown = per_mesh
    if only_real_choice:
        shown = shown[shown.n_choice > 1]
    if scope == "틀린 것만":
        shown = shown[~shown["맞음"]]
    elif scope == "맞은 것만":
        shown = shown[shown["맞음"]]
    st.caption(f"{len(shown)} / {len(per_mesh)}개")
    if shown.empty:
        st.warning("해당하는 메쉬가 없습니다.")
        st.stop()
    shown = shown.sort_values(["n_choice", "name"], ascending=[False, True])

    labels = {}
    for _, row in shown.iterrows():
        if row["맞음"]:
            mark = "O"
        else:
            mark = "X"
        labels[row["name"]] = f"{row['name']}  후보{row.n_cand}/정답{row.n_truth}/선택{row.n_pick}  {mark}"
    picked_label = st.selectbox(f"메쉬 ({len(shown)}개)", list(labels.values()))
    picked_name = None
    for name, label in labels.items():
        if label == picked_label:
            picked_name = name
            break

    st.divider()
    real_choice = per_mesh[per_mesh.n_choice > 1]
    correct_count = int(real_choice["맞음"].sum())
    total_count = len(real_choice)
    st.metric("실제 선택분 정확도", f"{correct_count}/{total_count}", f"{correct_count / max(1, total_count):.1%}")

with st.spinner("재생성하는 중..."):
    view = build_view(picked_name, render_version())

if view["same"]:
    match_text = "✅ 사람과 **일치**"
else:
    match_text = "❌ 사람과 **다름**"
st.markdown(f"**{picked_name}** · fold {view['fold']} · 임계값 {view['thr']:.2f} · "
            f"선택지 {view['n_choice']}가지 · " + match_text)

metric_columns = st.columns(4)
metric_columns[0].metric("모델 선택", f"{view['n_pick']}개", delta=view["n_pick"] - view["n_truth"],
                         help="정답 개수와의 차이")
metric_columns[1].metric("삼각형 (원본)", view["before"]["삼각형"])
metric_columns[2].metric("삼각형 (모델)", view["model"]["삼각형"],
                         delta=view["model"]["삼각형"] - view["before"]["삼각형"], delta_color="inverse")
metric_columns[3].metric("삼각형 (사람)", view["human"]["삼각형"],
                         delta=view["human"]["삼각형"] - view["before"]["삼각형"], delta_color="inverse")

if len(view["notes"]) > 0:
    st.warning("적용하지 못한 선택: " + " / ".join(view["notes"]))

show_html(view["html"], height=660)

left_column, right_column = st.columns([3, 2])
with left_column:
    st.subheader("후보별 모델 점수")
    score_table = view["table"].copy()
    verdicts = []
    for _, row in score_table.iterrows():
        if row.picked == 1 and row.truth == 1:
            verdicts.append("맞음")
        elif row.picked == 1 and row.truth == 0:
            verdicts.append("잘못 골랐음")
        elif row.picked == 0 and row.truth == 1:
            verdicts.append("놓쳤음")
        else:
            verdicts.append("-")
    score_table["판정"] = verdicts
    score_table["score"] = score_table.score.round(3)
    show_table(score_table[["edge", "score", "picked", "truth", "판정"]].rename(
        columns={"edge": "변", "score": "모델 점수", "picked": "모델 선택", "truth": "정답"}), hide_index=True)
with right_column:
    st.subheader("전/후 지표")
    stats_rows = []
    for key in view["before"]:
        stats_rows.append({"지표": key, "원본": str(view["before"][key]), "모델": str(view["model"][key]),
                           "사람": str(view["human"][key])})
    show_table(pd.DataFrame(stats_rows), hide_index=True)

with st.expander("이 화면을 읽는 법"):
    st.markdown(
        """
- **왼쪽 = 모델, 오른쪽 = 사람.** 둘 다 같은 원본에서 출발합니다.
- **초록** = 원본에 없던 사각형. 초록 위치가 서로 다르면 모델이 다른 곳을 합친 것입니다.
- 아래 표의 **판정** 열: `맞음` 사람도 합친 변 / `잘못 골랐음` 모델만 합친 변 / `놓쳤음` 사람은 합쳤는데 모델이 안 고른 변
- 점수는 **out-of-fold** 입니다. 각 메쉬는 자기를 학습에 쓰지 않은 모델이 채점했습니다.
"""
    )

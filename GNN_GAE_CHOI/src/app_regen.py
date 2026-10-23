"""메쉬 재생성 결과 화면 — 원본 / 모델 재생성 결과 / 사람 수정 결과를 나란히, 세 화면 시점 연동.

    streamlit run GNN_GAE_CHOI/src/app_regen.py

    왼쪽 (세로 전체)  원본(비정상 메쉬) — 지워진 element 를 누가 지웠는지 색으로
    오른쪽 위         모델 재생성 결과 — 지우기·만들기가 반영된 최종 메쉬, 새로 만든 element 초록
    오른쪽 아래       사람 수정 결과   — 같은 방식
    (색 규칙과 '원본에 표시할 삭제' 보기 전환은 viz_regen.py 머리말 참고)

모델을 여기서 돌리지 않는다. 채점 때 기록한 연산(어느 절점을 어디로 합쳤고 어느 삼각형
쌍을 합쳤는지)을 원본에 **다시 적용**해 결과를 만든다 — 그래서 화면은 채점된 결과와
정확히 같고, torch 없이도 열린다.
    테스트 셋   runs/regen_<tag>/eval_test.csv  (regen_eval.py 가 만든다)
    학습 셋     runs/regen_<tag>/eval_cv.csv 또는 cv_pred.csv (각 메쉬를 자기를 학습에 안 쓴 모델이 만든 것)
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import gae_paths
from bdf_io import discover_pairs
from fasoo_drm import setting_fasoo
from meshedit import EditableMesh, check, replay_ops
from pairs import discover_good_pairs, load_pair
from regen_oracle import parse_collapses, parse_edges

setting_fasoo()
st.set_page_config(page_title="메쉬 재생성 결과", layout="wide")


def show_table(table, **options) -> None:
    """Streamlit 1.49 에서 표 폭 인자가 바뀌어서 양쪽 버전을 다 받아준다."""
    try:
        st.dataframe(table, width="stretch", **options)
    except TypeError:
        st.dataframe(table, use_container_width=True, **options)


def show_html(page_html: str, height: int) -> None:
    """HTML 을 iframe 으로 띄운다 (새 버전은 st.iframe, 옛 버전은 components.html)."""
    iframe_function = getattr(st, "iframe", None)
    if iframe_function is not None:
        iframe_function(page_html, height=height)
    else:
        components.html(page_html, height=height, scrolling=False)


# ──────────────────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────────────────
def list_runs() -> list:
    """학습된 모델 폴더 목록 (runs/regen_*/model.json 이 있는 것)."""
    run_dirs = []
    for model_json in glob.glob(os.path.join(gae_paths.RUNS_DIR, "regen_*", "model.json")):
        run_dirs.append(os.path.dirname(model_json))
    return sorted(run_dirs)


@st.cache_data
def run_meta(run_dir: str, version: float) -> dict:
    """model.json (+ 있으면 eval_test.json 을 "test" 로)."""
    with open(os.path.join(run_dir, "model.json"), encoding="utf-8") as file_handle:
        meta = json.load(file_handle)
    test_summary_path = os.path.join(run_dir, "eval_test.json")
    if os.path.exists(test_summary_path):
        with open(test_summary_path, encoding="utf-8") as file_handle:
            meta["test"] = json.load(file_handle)
    return meta


@st.cache_data
def table_path(run_dir: str, split_name: str) -> str:
    """테스트는 eval_test.csv. 학습 셋은 eval_cv.csv 가 있으면 그것을, 없으면 cv_pred.csv."""
    if split_name == "test":
        return os.path.join(run_dir, "eval_test.csv")
    cv_eval_path = os.path.join(run_dir, "eval_cv.csv")
    if os.path.exists(cv_eval_path):
        return cv_eval_path
    return os.path.join(run_dir, "cv_pred.csv")


def load_table(run_dir: str, split_name: str, version: float):
    """결과 표 + 분할 파일의 묶음/fold. 파일이 없으면 None."""
    path = table_path(run_dir, split_name)
    if not os.path.exists(path):
        return None
    table = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    for column in ("exact", "exact_shells", "did_nothing"):
        if column in table:
            table[column] = table[column].astype(str).str.lower().eq("true")

    # 분할 파일의 묶음/fold 를 붙인다 — 같은 결함을 고친 다른 쌍이 몇 개인지 보이려고
    from regen_data import load_split

    split_table = load_split()
    group_sizes = split_table.groupby("group").name.size()
    columns_to_drop = []
    for column in ("cv_fold", "group"):
        if column in table:
            columns_to_drop.append(column)
    table = table.drop(columns=columns_to_drop).merge(split_table[["name", "cv_fold", "group"]], on="name",
                                                     how="left")
    table["group_size"] = table.group.map(group_sizes).fillna(1).astype(int)
    return table


@st.cache_data
def path_maps():
    """(수정본 경로표, 원본 경로표)."""
    return discover_good_pairs(), discover_pairs(gae_paths.FE_DIR)[0]


def replay(bad_mesh: EditableMesh, collapses_text: str, merges_text: str) -> EditableMesh:
    """기록된 연산을 원본에 다시 적용한다 (모델을 돌리지 않는다)."""
    return replay_ops(bad_mesh, parse_collapses(collapses_text), parse_edges(merges_text))


def code_version() -> float:
    """화면 결과를 만드는 모듈이 바뀌면 캐시를 버린다 — st.cache_data 는 데코레이터가
    붙은 함수의 코드만 해시하므로 다른 모듈을 고쳐도 캐시가 남는다."""
    newest = 0.0
    for file_name in ("meshdiff.py", "meshedit.py", "viz_regen.py"):
        newest = max(newest, os.path.getmtime(os.path.join(gae_paths.SRC_DIR, file_name)))
    return newest


def steps_version() -> float:
    """'편집 과정' 탭은 소스 코드 자체를 보여주므로, 보여주는 파일이 바뀌어도 다시 만든다."""
    newest = 0.0
    for file_name in ("viz_steps.py", "meshedit.py", "meshdiff.py", "regen_model.py", "regen_feats.py"):
        newest = max(newest, os.path.getmtime(os.path.join(gae_paths.SRC_DIR, file_name)))
    return newest


@st.cache_data(show_spinner=False)
def build_steps_html(name: str, collapses_text: str, merges_text: str, margins_text: str,
                     stage1_threshold: float, stage2_threshold: float, version: float):
    """기록된 연산을 원본에 한 단계씩 다시 적용하는 단계별 화면 (viz_steps). (HTML, 단계 수)."""
    from viz_steps import build_steps, parse_record, render_steps_html

    good_paths, original_paths = path_maps()
    pair = load_pair(name, good_paths, original_paths)
    collapses, merges, margins = parse_record({"collapses": collapses_text, "merges": merges_text,
                                               "margins": margins_text})
    steps = build_steps(EditableMesh(pair.before), collapses, merges, margins, stage1_threshold, stage2_threshold,
                        EditableMesh(pair.after))
    return render_steps_html(steps, name), len(steps)


@st.cache_data(show_spinner=False)
def build(name: str, collapses_text: str, merges_text: str, version: float) -> dict:
    """세 화면 HTML + 변경 목록 + 요약 + 검사 + 내려받을 파일 내용."""
    from viz_regen import build_view, render_html

    good_paths, original_paths = path_maps()
    pair = load_pair(name, good_paths, original_paths)
    bad_mesh = EditableMesh(pair.before)
    model_mesh = replay(bad_mesh, collapses_text, merges_text)
    human_mesh = EditableMesh(pair.after)
    model_absorbed = {}
    for remove_node, keep_node in parse_collapses(collapses_text):
        model_absorbed[remove_node] = keep_node
    panes, changes, summary = build_view(bad_mesh, model_mesh, human_mesh, model_absorbed,
                                         dict(pair.nodes.absorbed))
    median_edge = bad_mesh.rep.median_edge_length
    check_report = check(model_mesh, bad_mesh, regen=True)

    change_rows = []
    for change in changes:
        if change.shared:
            shared_mark = "✓"
        else:
            shared_mark = "✗"
        change_rows.append(dict(누가=change.who, 변경=change.what, 대상=change.target, 상대쪽도=shared_mark,
                                설명=change.note))
    return dict(
        html=render_html(panes, changes, focus_radius=1.2 * median_edge),
        changes=pd.DataFrame(change_rows),
        summary=summary, viol=list(check_report.violations), warn=list(check_report.warnings),
        bdf=model_mesh.bdf_text(), json=model_mesh.json_text(),
    )


# ──────────────────────────────────────────────────────────────────────────
# 화면
# ──────────────────────────────────────────────────────────────────────────
st.title("메쉬 재생성 결과")
st.caption("왼쪽은 원본(비정상 메쉬), 오른쪽 위는 모델 재생성 결과, 오른쪽 아래는 사람 수정 결과입니다. "
           "세 화면의 시점이 연동됩니다. 왼쪽: 빨강 = 둘 다 지운 element · 주황 = 모델만 지운 element · "
           "보라 = 사람만 지운 element / 오른쪽: 초록 = 새로 만든 element")

runs = list_runs()
if len(runs) == 0:
    st.error("학습된 재생성 모델이 없습니다. 먼저 "
             "`python src/regen_train.py --tag final --loss list --scorer2 gbm --seeds 0,1,2` 를 돌리세요.")
    st.stop()


def run_label(run_dir: str) -> str:
    meta_of_run = run_meta(run_dir, os.path.getmtime(os.path.join(run_dir, "model.json")))
    test_text = ""
    if "test" in meta_of_run:
        test_text = f" · 테스트 {meta_of_run['test']['exact']:.0%}"
    return f"{os.path.basename(run_dir)}  (CV {meta_of_run.get('cv_exact', 0):.0%}{test_text})"


def split_label(split_name: str) -> str:
    if split_name == "test":
        return "테스트 (10%)"
    return "학습 (90%, CV 예측)"


with st.sidebar:
    st.header("모델과 메쉬 고르기")
    default_index = 0
    for run_index in range(len(runs)):
        if os.path.basename(runs[run_index]) == "regen_final":
            default_index = run_index
            break

    run_dir = st.selectbox("재생성 모델", runs, index=default_index, format_func=run_label)
    meta = run_meta(run_dir, os.path.getmtime(os.path.join(run_dir, "model.json")))
    has_test = os.path.exists(os.path.join(run_dir, "eval_test.csv"))
    if has_test:
        default_split_index = 0
    else:
        default_split_index = 1
    split_name = st.radio("데이터", ["test", "train"], horizontal=True, index=default_split_index,
                          format_func=split_label,
                          help="테스트 셋은 학습·임계값 조정 어디에도 쓰지 않은 10%이고, 학습 셋과 같은 결함을 "
                               "공유하지 않습니다. 학습 셋은 5-fold 로 나눠 각 메쉬를 자기를 학습에 안 쓴 모델이 "
                               "만든 결과입니다.")
    result_path = table_path(run_dir, split_name)
    if os.path.exists(result_path):
        table_version = os.path.getmtime(result_path)
    else:
        table_version = 0.0
    table = load_table(run_dir, split_name, table_version)
    if table is None:
        st.warning("이 데이터의 결과 파일이 없습니다.\n\n"
                   f"`python src/regen_eval.py --run runs/{os.path.basename(run_dir)}` 를 먼저 돌리세요.")
        st.stop()

    result_filter = st.radio("결과", ["전체", "사람과 일치", "사람과 다름"], horizontal=True)
    kinds = st.multiselect("수정 종류", sorted(table.kind.unique()))
    only_reproducible = st.checkbox("우리 연산으로 재현 가능한 쌍만", value=False,
                                    help="절점 추가·좌표 이동 등 재생성 연산 밖의 수정이 섞인 쌍을 뺍니다 "
                                         "(regen_oracle.py 의 '재현됨').")
    query = st.text_input("검색 (예: 0006)")

    shown_table = table
    if result_filter == "사람과 일치":
        shown_table = shown_table[shown_table.exact]
    elif result_filter == "사람과 다름":
        shown_table = shown_table[~shown_table.exact]
    if kinds:
        shown_table = shown_table[shown_table.kind.isin(kinds)]
    if only_reproducible:
        shown_table = shown_table[shown_table.oracle == "재현됨"]
    if query:
        shown_table = shown_table[shown_table.name.str.contains(query, case=False)]
    st.caption(f"{len(shown_table)} / {len(table)}쌍")
    if shown_table.empty:
        st.warning("해당하는 메쉬가 없습니다.")
        st.stop()

    shown_table = shown_table.sort_values(["exact", "change_iou", "name"], ascending=[True, True, True])
    labels = {}
    for _, row in shown_table.iterrows():
        if row["exact"]:
            mark = "✓"
        else:
            mark = "✗"
        short_name = row["name"].replace("SN2592_FWD_Clean_", "_")
        labels[row["name"]] = f"{mark} {short_name}  IoU {float(row['change_iou']):.2f}  {row['kind']}"

    def mesh_label(mesh_name: str) -> str:
        return labels[mesh_name]

    name = st.selectbox(f"메쉬 ({len(shown_table)}개)", list(labels), format_func=mesh_label)

    st.divider()
    st.metric("정확 일치 (사람과 똑같이 재생성)", f"{table.exact.mean():.1%}",
              help="요소·1D·좌표까지 사람 결과와 같은 비율. 절점 번호는 보지 않습니다.")
    left_column, right_column = st.columns(2)
    left_column.metric("바뀐 요소 IoU", f"{table.change_iou.astype(float).mean():.3f}")
    right_column.metric("상한 (정답 대본)", f"{(table.oracle == '재현됨').mean():.1%}",
                        help="사람의 수정을 우리 연산으로 그대로 옮겼을 때 재현되는 비율 — 이 연산 체계의 천장")
    st.caption(f"임계값 1단계 {meta['thr1']:.2f} / 2단계 {meta['thr2']:.2f} · 채점기 {meta['scorer']}"
               f" · 학습 {meta['n_fit']}쌍")

row = table[table.name == name].iloc[0]
with st.spinner("재생성 결과를 그리는 중..."):
    view = build(name, row["collapses"], row["merges"], code_version())

if row["exact"]:
    match_text = "✅ 사람과 **정확히 일치**"
else:
    match_text = "❌ 사람과 **다름**"
head = f"**{name}** · {row['kind']} · {match_text} · 바뀐 요소 IoU **{float(row['change_iou']):.2f}**"
if row["oracle"] != "재현됨":
    head = head + f" · ⚠️ 이 쌍은 우리 연산으로 재현 불가 ({row['oracle']})"
st.markdown(head)

notes = []
if split_name == "train" and int(row["cv_fold"]) > 0:
    notes.append(f"CV fold {int(row['cv_fold'])} 모델이 만든 결과 (이 메쉬를 학습에 쓰지 않은 모델)")
if int(row["group_size"]) > 1:
    notes.append(f"같은 결함을 고친 쌍이 이 밖에 {int(row['group_size']) - 1}개 더 있음 — "
                 "분할할 때 한 묶음으로 같은 쪽에 두었습니다")
if len(notes) > 0:
    st.caption(" · ".join(notes))
if "p_bad" in row and str(row["p_bad"]) != "":
    st.caption(f"심판 분류기의 비정상 확률: 원본 {float(row['p_bad']):.3f} → "
               f"모델 {float(row['p_regen']):.3f} · 사람 {float(row['p_good']):.3f}")

model_summary = view["summary"]["model"]
human_summary = view["summary"]["human"]
metric_columns = st.columns(len(model_summary))
column_index = 0
for key in model_summary:
    metric_columns[column_index].metric(key, f"모델 {model_summary[key]} · 사람 {human_summary[key]}")
    column_index = column_index + 1

show_html(view["html"], height=960)

steps_tab, changes_tab, check_tab, stats_tab, download_tab = st.tabs(
    ["편집 과정 (단계별)", "변경 목록", "검사", "전/후 지표", "내려받기"])

with steps_tab:
    st.caption("모델이 적용한 연산을 원본에 한 단계씩 다시 적용합니다. 단계마다 그 상태의 메쉬, "
               "실행되는 실제 소스 코드(강조된 줄), 그 메쉬에서 일어난 일(element 별 결과, 검사 수치)을 "
               "보여줍니다. ◀ ▶ 버튼이나 키보드 ← → 로 넘깁니다. 시점은 단계를 넘겨도 유지됩니다.")
    if st.toggle("편집 과정 보기", value=False, key=f"steps_{name}"):
        margins_text = str(row.get("margins", "") or "")
        with st.spinner("단계를 만드는 중..."):
            steps_html, step_count = build_steps_html(name, row["collapses"], row["merges"], margins_text,
                                                      float(meta["thr1"]), float(meta["thr2"]), steps_version())
        st.caption(f"단계 {step_count}개 · 1단계 edge collapse {len(parse_collapses(row['collapses']))}회 · "
                   f"2단계 triangle-pair merge {len(parse_edges(row['merges']))}회")
        show_html(steps_html, height=860)

with changes_tab:
    only_one_side = st.checkbox("상대 쪽에 없는 변경만", value=False)
    change_table = view["changes"]
    if only_one_side:
        change_table = change_table[change_table["상대쪽도"] == "✗"]
    if change_table.empty:
        st.write("변경이 없습니다.")
    else:
        show_table(change_table, hide_index=True)
    st.caption("화면 위 '변경 위치로 확대…' 에서 같은 목록을 골라 그 자리로 갈 수 있습니다.")

with check_tab:
    if len(view["viol"]) > 0:
        violation_lines = []
        for violation in view["viol"]:
            violation_lines.append(f"- {violation}")
        st.error("불변식 위반:\n\n" + "\n".join(violation_lines))
    else:
        st.success("불변식 위반 없음 (1D 는 쉘 변 위, 비매니폴드 변 증가 없음, 법선 반전 없음, "
                   "ext 절점 이동 없음, 총 면적 변화 5% 이내)")
    if len(view["warn"]) > 0:
        warning_lines = []
        for warning in view["warn"]:
            warning_lines.append(f"- {warning}")
        st.warning("경고 (사람의 수정에서도 일어나는 일):\n\n" + "\n".join(warning_lines))
    applied_collapses = parse_collapses(row["collapses"])
    st.markdown(f"**모델이 적용한 연산** — edge collapse(절점 병합) {len(applied_collapses)}개, "
                f"triangle-pair merge(삼각형 쌍 병합) {len(parse_edges(row['merges']))}개")
    if len(applied_collapses) > 0:
        collapse_texts = []
        for remove_node, keep_node in applied_collapses:
            collapse_texts.append(f"{remove_node} → {keep_node}")
        st.write("edge collapse (remove → keep): " + ", ".join(collapse_texts))
    if row["merges"]:
        st.write("triangle-pair merge (공유 변): " + ", ".join(row["merges"].split(";")))

with stats_tab:
    stats_by_mesh = view["summary"]["stats"]
    stats_rows = []
    for key in stats_by_mesh["원본"]:
        stats_row = {"지표": key}
        for mesh_label_text in stats_by_mesh:
            stats_row[mesh_label_text] = str(stats_by_mesh[mesh_label_text][key])
        stats_rows.append(stats_row)
    show_table(pd.DataFrame(stats_rows), hide_index=True)

with download_tab:
    st.download_button("재생성 메쉬 .bdf", view["bdf"], file_name=f"{name}_regen.bdf")
    st.download_button("ext 정보 .json", view["json"], file_name=f"{name}_regen.json")
    st.caption("원본과 같은 쉼표 구분 .bdf 형식입니다. 절점 번호는 새로 만들지 않고, 합쳐진 "
               "절점은 자리가 남는 쪽 번호를 씁니다.")

with st.expander("이 화면을 읽는 법"):
    st.markdown(
        """
**배치** — 왼쪽은 원본(비정상 메쉬), 오른쪽 위는 모델 재생성 결과, 오른쪽 아래는 사람 수정 결과입니다.
오른쪽 두 화면은 지우기와 만들기가 모두 반영된 **최종 메쉬**이고, 지워진 것은 결과에 없으므로 왼쪽
원본 위에 표시합니다. 세 화면의 시점은 연동됩니다.

| 표시 | 의미 |
|---|---|
| 🔴 빨강 면 (왼쪽) | 모델과 사람이 **둘 다** 지운 element |
| 🟠 주황 면 (왼쪽) | **모델만** 지운 element (사람 결과에는 원래 모양 그대로 남음) |
| 🟣 보라 면 (왼쪽) | **사람만** 지운 element (모델 결과에는 원래 모양 그대로 남음) |
| 🟢 초록 면 (오른쪽) | 새로 만든 element (원본에 없던 element) |
| 굵은 선 | 지운 1D element (왼쪽, 위 색 규칙) / 새로 생긴 1D element (오른쪽, 진한 초록) |

- **원본에 표시할 삭제**: `비교` 는 누가 지웠는지 색으로 나눕니다. `모델` / `사람` 을 고르면 그쪽이
  지운 element 만 빨강으로 칠합니다.
- 도구막대의 체크박스로 강조를 하나씩 끄고 켭니다. 끄면 그 element 는 회색으로 돌아갑니다.
- 비교는 node **번호가 아니라 좌표**로 합니다. 그래서 변경 목록의 'node 이동' 은 사람 결과에만 나올 수
  있습니다 — 모델은 좌표를 옮기지 않습니다.
- node 삭제·이동은 3D 화면에 따로 표시하지 않습니다. 아래 **변경 목록** 표와 '변경 위치로 확대…' 에 있습니다.
"""
    )

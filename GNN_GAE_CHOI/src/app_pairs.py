"""Bad ↔ Good 쌍 뷰어 — 사람이 무엇을 고쳤는지 직접 보는 화면.

왼쪽이 원본(Bad), 오른쪽이 사람이 고친 것(Good). 두 화면은 시점이 연동되므로 한쪽을
돌리면 반대쪽도 같은 각도로 따라온다.

    streamlit run GNN_GAE_CHOI/src/app_pairs.py

색:
    빨강 (왼쪽)   이 수정으로 **없어진** 요소
    초록 (오른쪽) **새로 생긴** 요소
    자주 X        사라지거나 새로 생긴 **절점**
    노란 마름모   자리를 옮긴 절점
    회색          그대로, 검정 테두리 = 불량각 요소, 파랑 = 1D 요소, 하늘색 = ext 절점
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import gae_paths
from bdf_io import discover_pairs
from fasoo_drm import setting_fasoo
from meshedit import EditableMesh
from pairs import all_pairs, discover_good_pairs, load_pair
from regen_data import kind_of
from viz_compare import render_html, stats

setting_fasoo()
st.set_page_config(page_title="Bad ↔ Good 쌍 뷰어", layout="wide")


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
@st.cache_data
def pair_names() -> list:
    return all_pairs()


@st.cache_data
def path_maps():
    return discover_good_pairs(), discover_pairs(gae_paths.FE_DIR)[0]


@st.cache_data
def judge_probabilities():
    """judge_pairs.py 가 저장한 수정 전/후 비정상 확률. 없으면 원래 GNN_GAE 의 것, 그것도 없으면 None."""
    for folder in (gae_paths.OUT_DIR, os.path.join(gae_paths.ORIGINAL_GAE_ROOT, "out")):
        path = os.path.join(folder, "judge_pairs.csv")
        if os.path.exists(path):
            return pd.read_csv(path).set_index("name")
    return None


def classify(pair) -> str:
    """수정의 종류 (regen_data.kind_of 와 같은 규칙).

    네 축(절점삭제/절점추가/좌표이동/변변화)으로 판정하되 이름에서는 '변변화' 를 뺀다 — 508쌍 중
    503쌍에 붙어 있어 구분이 안 된다. 대신 **연결이 안 바뀐 5건**만 따로 적는다.
    """
    return kind_of(pair)


def join_numbers(numbers) -> str:
    texts = []
    for number in numbers:
        texts.append(str(number))
    return ", ".join(texts)


@st.cache_data(show_spinner=False)
def pair_view(name: str, version: float) -> dict:
    """쌍 하나의 뷰어 HTML 과 요약. version 은 뷰어 코드가 바뀌면 캐시를 버리게 하는 값."""
    good_paths, original_paths = path_maps()
    pair = load_pair(name, good_paths, original_paths)
    before_mesh = EditableMesh(pair.before)
    after_mesh = EditableMesh(pair.after)
    _, removed, created = pair.shell_diff()
    kind = classify(pair)
    node_match = pair.nodes

    # 자주색 X = 구조가 바뀐 절점 (사라짐 / 새로 생김)
    node_marks_before = {}
    for node_id in node_match.deleted:
        node_marks_before[node_id] = f"절점 {node_id} — 이 수정으로 삭제됨 (이어받은 절점 없음)"
    for absorbed_node, survivor_node in node_match.absorbed.items():
        node_marks_before[absorbed_node] = f"절점 {absorbed_node} — {survivor_node} 에 흡수되어 사라짐 (병합)"
    for gone_node, new_node in node_match.replaced_by_new.items():
        node_marks_before[gone_node] = f"절점 {gone_node} — 새로 만든 {new_node} 가 대신함"
    node_marks_after = {}
    for node_id in node_match.added:
        node_marks_after[node_id] = f"절점 {node_id} — 새로 생김"

    # 노란 마름모 = 자리를 옮긴 절점. 흡수한 쪽이 움직였다면 **출발 위치**도 찍는다.
    # 이게 없으면 "어디서 어디로 갔는지" 가 안 보이고, 도착 위치가 흡수된 절점의
    # 자리와 정확히 겹쳐서 좌우 화면에 같은 마커가 찍힌 것처럼 보인다 (실측 `_0014`).
    survivors = node_match.survivors
    move_marks_before = {}
    for node_id in node_match.moved:
        move_marks_before[node_id] = f"절점 {node_id} — 여기 있다가 옮겨감"
    for survivor_node in node_match.survivor_landed:
        absorbed_list = survivors.get(survivor_node, [])
        move_marks_before[survivor_node] = (f"절점 {survivor_node} — 여기 있다가 "
                                            f"{join_numbers(absorbed_list)} 자리로 옮겨가며 흡수")
    for node_id, other_node in node_match.coincident.items():
        move_marks_before[node_id] = f"절점 {node_id} — {other_node} 와 좌표가 겹침 (확인 필요)"

    move_marks_after = {}
    for node_id in node_match.moved:
        move_marks_after[node_id] = f"절점 {node_id} — 여기로 옮겨옴"
    for survivor_node, absorbed_list in survivors.items():
        where = node_match.survivor_landed.get(survivor_node, "제자리")
        move_marks_after[survivor_node] = f"절점 {survivor_node} — {join_numbers(absorbed_list)} 를 흡수 ({where})"
    for node_id, other_node in node_match.coincident.items():
        move_marks_after[node_id] = f"절점 {node_id} — {other_node} 와 좌표가 겹침 (확인 필요)"

    page_html = render_html(before_mesh, after_mesh, [], "", embed=True,
                            node_marks_before=node_marks_before, node_marks_after=node_marks_after,
                            move_marks_before=move_marks_before, move_marks_after=move_marks_after)
    return {
        "html": page_html, "kind": kind,
        "before": stats(before_mesh), "after": stats(after_mesh),
        "n_removed": len(removed), "n_created": len(created),
        "deleted": sorted(node_match.deleted), "added": sorted(node_match.added),
        "moved": sorted(node_match.moved), "survivors": survivors,
        "replaced": dict(node_match.replaced_by_new),
        "landed": dict(node_match.survivor_landed),
        "basis": dict(node_match.basis), "explained": node_match.explained,
        "coincident": dict(node_match.coincident),
        "n_line_before": len(pair.before.lines), "n_line_after": len(pair.after.lines),
    }


def render_version() -> float:
    return os.path.getmtime(os.path.join(gae_paths.SRC_DIR, "viz_compare.py"))


# ──────────────────────────────────────────────────────────────────────────
st.title("Bad ↔ Good 쌍 뷰어")
st.caption("왼쪽이 원본, 오른쪽이 사람이 고친 것. 한쪽을 돌리면 반대쪽도 같은 각도로 따라옵니다. "
           "빨강 = 없어진 요소, 초록 = 새로 생긴 요소, 자주색 X = 바뀐 절점.")

names = pair_names()
probabilities = judge_probabilities()

with st.sidebar:
    st.header("메쉬 고르기")
    kinds = st.multiselect(
        "수정 종류로 거르기",
        ["절점삭제", "연결만 변경", "절점삭제 + 절점추가", "절점추가",
         "좌표이동", "절점삭제 + 좌표이동", "변화없음", "좌표이동 (연결 변화 없음)"],
        default=[],
        help="비워 두면 전부 봅니다. 종류를 판별하려면 쌍을 열어봐야 해서 처음 고를 때 시간이 조금 걸립니다.",
    )

    shown_names = names
    if kinds:
        good_paths, original_paths = path_maps()
        with st.spinner("종류를 판별하는 중..."):
            shown_names = []
            for name in names:
                if classify(load_pair(name, good_paths, original_paths)) in kinds:
                    shown_names.append(name)
        st.caption(f"{len(shown_names)} / {len(names)}개")

    order = st.radio("정렬", ["많이 좋아진 순", "이름순", "오히려 나빠진 순"],
                     help="'오히려 나빠진 순' 은 사람이 고쳤는데 심판 확률이 올라간 쌍입니다 — "
                          "508개 중 13개뿐이고, 라벨이나 수정 작업을 되짚어볼 후보입니다.")
    if probabilities is not None and order != "이름순":
        change_in_probability = probabilities["p_after"] - probabilities["p_before"]
        ranked_names = change_in_probability.sort_values(ascending=(order == "많이 좋아진 순")).index
        shown_set = set(shown_names)
        shown_names = []
        for name in ranked_names:
            if name in shown_set:
                shown_names.append(name)

    query = st.text_input("검색 (부분 문자열, 예: 0006)")
    candidates = []
    for name in shown_names:
        if not query or query.lower() in name.lower():
            candidates.append(name)
    if len(candidates) == 0:
        st.warning("해당하는 메쉬가 없습니다.")
        st.stop()

    def name_label(name: str) -> str:
        if probabilities is not None and name in probabilities.index:
            probability_row = probabilities.loc[name]
            return f"{name}   p {probability_row.p_before:.2f} → {probability_row.p_after:.2f}"
        return name

    candidate_labels = []
    for name in candidates:
        candidate_labels.append(name_label(name))
    picked_label = st.selectbox(f"메쉬 ({len(candidates)}개)", candidate_labels, index=0)
    picked_name = candidates[candidate_labels.index(picked_label)]

    st.divider()
    st.caption("p 는 심판 분류기(`deploy_10_14`)가 본 비정상 확률입니다. "
               "이 모델은 이 메쉬들을 학습에 쓰지 않았습니다.")

with st.spinner("쌍을 불러오는 중..."):
    view = pair_view(picked_name, render_version())

before_stats = view["before"]
after_stats = view["after"]

head = f"**{picked_name}** · 수정 종류: **{view['kind']}**"
if probabilities is not None and picked_name in probabilities.index:
    probability_row = probabilities.loc[picked_name]
    head = head + f" · 심판 p {probability_row.p_before:.3f} → {probability_row.p_after:.3f}"
st.markdown(head)

metric_columns = st.columns(5)
metric_columns[0].metric("없어진 요소", view["n_removed"])
metric_columns[1].metric("새로 생긴 요소", view["n_created"])
metric_columns[2].metric("쉘 수", after_stats["쉘 요소"], delta=after_stats["쉘 요소"] - before_stats["쉘 요소"])
metric_columns[3].metric("삼각형", after_stats["삼각형"], delta=after_stats["삼각형"] - before_stats["삼각형"],
                         delta_color="inverse")
metric_columns[4].metric("불량각 요소", after_stats["불량각 요소"],
                         delta=after_stats["불량각 요소"] - before_stats["불량각 요소"], delta_color="inverse")

show_html(view["html"], height=660)

left_column, right_column = st.columns(2)
with left_column:
    st.subheader("바뀐 절점")
    rows = []
    for node_id in view["deleted"]:
        rows.append({"절점": node_id, "무엇": "삭제됨"})
    for survivor_node, absorbed_list in view["survivors"].items():
        where = view["landed"].get(survivor_node, "제자리")
        rows.append({"절점": survivor_node, "무엇": f"{join_numbers(absorbed_list)} 를 흡수 (병합, {where})"})
    for gone_node, new_node in view["replaced"].items():
        rows.append({"절점": gone_node, "무엇": f"새로 만든 {new_node} 가 대신함"})
    for node_id in view["added"]:
        rows.append({"절점": node_id, "무엇": "새로 생김"})
    for node_id in view["moved"]:
        rows.append({"절점": node_id, "무엇": "좌표 이동"})
    for node_id, other_node in view["coincident"].items():
        rows.append({"절점": node_id, "무엇": f"{other_node} 와 좌표 겹침 — 확인 필요"})
    if len(rows) > 0:
        show_table(pd.DataFrame(rows), hide_index=True)
    else:
        st.write("없음 — 절점은 그대로 두고 요소 연결만 바뀌었습니다.")

with right_column:
    st.subheader("전/후 지표")
    # 값을 문자열로 통일한다 — '삼각형 비율' 만 문자열이라 한 열에 숫자와 섞이면 표 변환이 실패한다.
    stats_rows = []
    for key in before_stats:
        stats_rows.append({"지표": key, "원본(Bad)": str(before_stats[key]), "수정(Good)": str(after_stats[key])})
    stats_rows.append({"지표": "1D 요소", "원본(Bad)": str(view["n_line_before"]),
                       "수정(Good)": str(view["n_line_after"])})
    show_table(pd.DataFrame(stats_rows), hide_index=True)

with st.expander("색이 뜻하는 것"):
    st.markdown(
        """
| 색 | 의미 |
|---|---|
| 🔴 빨강 (왼쪽) | 이 수정으로 **없어진** 요소 |
| 🟢 초록 (오른쪽) | **새로 생긴** 요소 |
| 🟣 자주색 X | 사라지거나 새로 생긴 절점 |
| 🟡 노란 마름모 | 자리를 옮긴 절점 (출발 위치·도착 위치) |
| ⬛ 검은 테두리 | 불량각 요소 (내각 < 30° 또는 > 150°) |
| 🔵 파랑 | 1D 요소 (CBAR/CBEAM) |
| 🟠 주황 | 양옆 삼각형을 합치려면 **같이 지워야 하는** 1D 요소 |
| 하늘색 원 | ext 절점 — 이웃 패치와 공유하는 절단면 |

요소를 클릭하면 내각·skew·aspect·절점 번호가 뜹니다. 도구막대 체크박스로 각각 켜고 끌 수 있습니다.
"""
    )

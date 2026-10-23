"""규칙 기반 수리기 뷰어 — .bdf 를 골라 수리 전/후를 나란히 본다 (repair_rule 용, 지금은 참고용).

    streamlit run GNN_GAE_CHOI/src/app.py

두 3D 화면은 **시점이 연동**된다. 연동 스크립트가 iframe 안에서 돌아야 해서 plotly 차트를
``st.plotly_chart`` 로 두 개 그리지 않고, 뷰어 HTML 한 덩어리를 ``components.html`` 로 띄운다
(Streamlit 은 차트 사이 JavaScript 통신을 막는다).
"""

from __future__ import annotations

import os
import random

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import gae_paths
import features as F
from bdf_io import discover_pairs
from fasoo_drm import setting_fasoo
from labels import load_label_table
from meshedit import EditableMesh, merge_candidates
from repair_rule import RepairConfig, repair
from viz_compare import corrupt, render_html, stats

# 라벨 엑셀이 fasoo 로 암호화돼 있을 수 있으므로 파일을 열기 전에 한 번 해제한다.
setting_fasoo()
st.set_page_config(page_title="메쉬 수리 뷰어", layout="wide")

LABEL_TAGS = {0: "정상", 1: "비정상", 2: "분류어려움"}


def show_table(table, **options) -> None:
    """Streamlit 1.49 에서 표 폭 인자가 바뀌어서 양쪽 버전을 다 받아준다."""
    try:
        st.dataframe(table, width="stretch", **options)
    except TypeError:
        st.dataframe(table, use_container_width=True, **options)


# ──────────────────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def available_samples() -> list:
    mesh_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    return sorted(mesh_paths)


@st.cache_data
def mesh_paths_table() -> dict:
    mesh_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    return mesh_paths


@st.cache_data
def label_lookup() -> dict:
    """{이름: (라벨, 애매 여부)}."""
    label_table = load_label_table(gae_paths.LABEL_XLSX)
    lookup = {}
    for row in label_table.itertuples():
        lookup[row.name] = (int(row.cls), int(row.ambig))
    return lookup


def render_version() -> float:
    """viz_compare.py 가 바뀌면 아래 캐시를 버리게 하는 값 (st.cache_data 는 다른 모듈 변경을 모른다)."""
    return os.path.getmtime(os.path.join(gae_paths.SRC_DIR, "viz_compare.py"))


@st.cache_data
def repair_index():
    """scan_repairs.py 가 남긴 전수 조사 결과. 없으면 원래 GNN_GAE 의 것, 그것도 없으면 None."""
    for folder in (gae_paths.OUT_DIR, os.path.join(gae_paths.ORIGINAL_GAE_ROOT, "out")):
        path = os.path.join(folder, "repairs_cls1.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def label_tag(name: str) -> str:
    label = label_lookup().get(name)
    if label is None:
        return "(라벨 없음)"
    label_class, is_ambiguous = label
    if label_class == 2:
        if is_ambiguous:
            return "(분류어려움·애매)"
        return "(분류어려움)"
    if is_ambiguous:
        return f"(애매·{LABEL_TAGS[label_class]})"
    return f"({LABEL_TAGS[label_class]})"


def pick_by_label(names: list, key: str, title: str):
    """이름 옆에 라벨을 붙여 고르게 한다."""
    query = st.text_input("검색 (부분 문자열, 예: 0694)", key=key,
                          help="입력한 글자를 이름 어디에든 포함하는 항목만 드롭다운에 남습니다.")
    shown_pairs = []
    for name in names:
        display = f"{name}  {label_tag(name)}"
        if not query or query.lower() in display.lower():
            shown_pairs.append((display, name))
    if len(shown_pairs) == 0:
        st.warning(f"'{query}' 를 포함하는 항목이 없습니다.")
        return None
    displays = []
    for display, name in shown_pairs:
        displays.append(display)
    picked_display = st.selectbox(f"{title} ({len(shown_pairs)}개)", displays, index=0)
    for display, name in shown_pairs:
        if display == picked_display:
            return name
    return None


# ──────────────────────────────────────────────────────────────────────────
# 수리 — 무거우니 캐시한다. 캐시 키는 아래 인자 전부.
# ──────────────────────────────────────────────────────────────────────────
def load_base_mesh(name: str, corrupt_count: int, seed: int):
    """(메쉬, 실제로 쪼갠 사각형 수)."""
    bdf_path, json_path = mesh_paths_table()[name]
    base_mesh = EditableMesh.load(bdf_path, json_path)
    done_count = 0
    if corrupt_count:
        base_mesh, done_count = corrupt(base_mesh, corrupt_count, random.Random(seed))
    return base_mesh, done_count


def blocked_reason(candidate, config: RepairConfig) -> str:
    """수리 뒤에도 남은 인접 삼각형 쌍이 왜 안 합쳐졌나."""
    if candidate.blocked:
        if "1D" in candidate.blocked:
            return "1D 요소가 변 위에 있음"
        if "오목" in candidate.blocked:
            return "합치면 오목 사각형"
        if "패널" in candidate.blocked:
            return "다른 패널 (이면각 큼)"
        if "면적" in candidate.blocked:
            return "표면 면적이 바뀜"
        return candidate.blocked
    if candidate.dihedral > config.max_dihedral:
        return "다른 패널 (이면각 큼)"
    if candidate.quality is not None and F.is_bad_shell(candidate.quality):
        return "결과가 불량각 요소"
    if candidate.quality is not None and candidate.quality.skew > config.max_skew:
        return f"결과 skew > {config.max_skew:.0f}°"
    if candidate.quality is not None and candidate.quality.aspect > config.max_aspect:
        return f"결과 aspect > {config.max_aspect:.1f}"
    return "남은 후보 (적용 가능)"


@st.cache_data(show_spinner=False)
def run_repair(name: str, max_dihedral: float, max_skew: float, max_aspect: float, use_swap: bool,
               corrupt_count: int, seed: int, version: float) -> dict:
    """선택한 메쉬를 수리하고 뷰어 HTML 과 요약을 돌려준다."""
    base_mesh, done_count = load_base_mesh(name, corrupt_count, seed)
    config = RepairConfig(max_dihedral=max_dihedral, max_skew=max_skew, max_aspect=max_aspect, use_swap=use_swap)
    fixed_mesh, operations = repair(base_mesh, config)

    before_stats = stats(base_mesh)
    after_stats = stats(fixed_mesh)
    subtitle = f"연산 {len(operations)}회 · 삼각형 {before_stats['삼각형']} → {after_stats['삼각형']}"
    page_html = render_html(base_mesh, fixed_mesh, operations, subtitle, embed=True)

    # 왜 더 못 고쳤는지 — 막힌 후보를 사유별로 센다
    blocked_counts = {}
    for candidate in merge_candidates(fixed_mesh):
        reason = blocked_reason(candidate, config)
        if reason not in blocked_counts:
            blocked_counts[reason] = 0
        blocked_counts[reason] = blocked_counts[reason] + 1

    # 수리 뒤에도 남은 삼각형 중 짝이 아예 없는 것
    paired_shells = set()
    for candidate in merge_candidates(fixed_mesh):
        paired_shells.add(candidate.si)
        paired_shells.add(candidate.sj)
    lonely_count = 0
    for shell_index in fixed_mesh.tri_indices():
        if shell_index not in paired_shells:
            lonely_count = lonely_count + 1

    operation_list = []
    for operation in operations:
        operation_list.append((operation.kind, operation.key))
    return {"html": page_html, "ops": operation_list, "before": before_stats, "after": after_stats,
            "n_corrupt": done_count, "blocked": blocked_counts, "lonely_tri": lonely_count,
            "bdf": fixed_mesh.to_raw()}


@st.cache_data(show_spinner=False)
def repaired_bdf_text(name: str, max_dihedral: float, max_skew: float, max_aspect: float, use_swap: bool,
                      corrupt_count: int, seed: int, version: float) -> str:
    """수리된 메쉬를 .bdf 문자열로. 내려받기 버튼용."""
    base_mesh, _ = load_base_mesh(name, corrupt_count, seed)
    config = RepairConfig(max_dihedral=max_dihedral, max_skew=max_skew, max_aspect=max_aspect, use_swap=use_swap)
    fixed_mesh, _ = repair(base_mesh, config)
    return fixed_mesh.bdf_text()


# ──────────────────────────────────────────────────────────────────────────
# 화면
# ──────────────────────────────────────────────────────────────────────────
st.title("메쉬 수리 전/후 뷰어")
st.caption("비정상 메쉬의 삼각형을 사각형으로 되돌린다. 두 3D 화면은 시점이 연동되므로 "
           "한쪽을 돌리면 반대쪽도 같은 각도로 따라온다.")

names = available_samples()
index_table = repair_index()

with st.sidebar:
    st.header("메쉬 고르기")
    if index_table is not None:
        scope_options = ["수리가 일어나는 것만", "비정상 전체", "전체"]
    else:
        scope_options = ["비정상 전체", "전체"]
    scope = st.radio("목록", scope_options,
                     help="'수리가 일어나는 것만' 은 scan_repairs.py 가 미리 조사한 결과를 씁니다 — "
                          "규칙 기반 수리기가 실제로 손댈 수 있는 메쉬입니다.")
    if scope == "수리가 일어나는 것만" and index_table is not None:
        name_set = set(names)
        repaired_rows = index_table[index_table.n_op > 0].sort_values("n_op", ascending=False)
        shown_names = []
        for name in repaired_rows["name"]:
            if name in name_set:
                shown_names.append(name)
        st.caption(f"비정상 {len(index_table)}개 중 {len(shown_names)}개 ({len(shown_names) / len(index_table):.0%}) — "
                   "연산 많은 순")
    elif scope == "비정상 전체":
        lookup = label_lookup()
        shown_names = []
        for name in names:
            if lookup.get(name, (9, 0))[0] == 1:
                shown_names.append(name)
    else:
        shown_names = names

    picked_name = pick_by_label(shown_names, "pick", "모델 이름")

    st.divider()
    st.header("수리 설정")
    use_swap = st.checkbox("swap 보조 탐색", value=True,
                           help="바로는 못 합치는 쌍을 한 번 뒤집어 합칠 수 있는지 본다. 실측으로는 이득이 0이었다.")
    max_dihedral = st.slider("이면각 상한 (°)", 0.0, 45.0, 10.0, 1.0,
                             help="두 삼각형이 이보다 꺾여 있으면 서로 다른 패널로 보고 합치지 않는다.")
    max_skew = st.slider("결과 skew 상한 (°)", 10.0, 90.0, 45.0, 5.0,
                         help="합친 사각형이 이보다 찌그러지면 합치지 않는다.")
    max_aspect = st.slider("결과 aspect 상한", 1.5, 10.0, 4.0, 0.5)

    st.divider()
    st.header("손상 주입 (선택)")
    st.caption("정상 메쉬의 사각형을 일부러 쪼갠 뒤 수리한다. 수리기가 제대로 도는지 보는 검증이다.")
    corrupt_count = st.number_input("쪼갤 사각형 수", 0, 30, 0, 1)
    seed = st.number_input("seed", 0, 9999, 0, 1)

if not picked_name:
    st.stop()

with st.spinner("수리하는 중..."):
    result = run_repair(picked_name, max_dihedral, max_skew, max_aspect, use_swap, int(corrupt_count), int(seed),
                        render_version())

before_stats = result["before"]
after_stats = result["after"]
head = f"**{picked_name}** {label_tag(picked_name)}"
if result["n_corrupt"]:
    head = head + f" · 사각형 {result['n_corrupt']}개를 일부러 쪼갠 뒤 수리"
st.markdown(head)

metric_columns = st.columns(4)
metric_columns[0].metric("적용한 연산", f"{len(result['ops'])}회")
metric_columns[1].metric("삼각형", after_stats["삼각형"], delta=after_stats["삼각형"] - before_stats["삼각형"],
                         delta_color="inverse")
ratio_change = float(after_stats["삼각형 비율"]) - float(before_stats["삼각형 비율"])
metric_columns[2].metric("삼각형 비율", after_stats["삼각형 비율"], delta=f"{ratio_change:+.3f}",
                         delta_color="inverse")
metric_columns[3].metric("불량각 요소", after_stats["불량각 요소"],
                         delta=after_stats["불량각 요소"] - before_stats["불량각 요소"], delta_color="inverse")

if len(result["ops"]) == 0:
    st.info("이 메쉬에서는 적용할 수 있는 연산이 없어 좌우가 동일합니다. "
            "아래 **왜 더 못 고쳤나** 를 보면 후보가 무엇에 막혔는지 알 수 있습니다.")

components.html(result["html"], height=640, scrolling=False)

left_column, right_column = st.columns([3, 2])
operation_names = {"merge": "삼각형 2개 → 사각형", "swap": "대각선 뒤집기", "split": "사각형 → 삼각형 2개"}

with left_column:
    st.subheader("적용한 연산")
    if len(result["ops"]) > 0:
        operation_rows = []
        for kind, key in result["ops"]:
            node_texts = []
            for node_id in key:
                node_texts.append(str(node_id))
            operation_rows.append({"연산": operation_names.get(kind, kind), "절점": ", ".join(node_texts)})
        show_table(pd.DataFrame(operation_rows, index=range(1, len(result["ops"]) + 1)))
    else:
        st.write("없음")

with right_column:
    st.subheader("왜 더 못 고쳤나")
    st.caption("수리 후에도 남아 있는 인접 삼각형 쌍이 무엇에 막혀 있는지")
    if len(result["blocked"]) > 0:
        def count_of(item):
            return -item[1]

        sorted_reasons = sorted(result["blocked"].items(), key=count_of)
        show_table(pd.DataFrame(sorted_reasons, columns=["사유", "쌍 수"]), hide_index=True)
    else:
        st.write("남은 인접 삼각형 쌍이 없습니다.")
    if result["lonely_tri"]:
        st.warning(f"수리 후 남은 삼각형 중 **{result['lonely_tri']}개는 인접 삼각형이 아예 없습니다** — "
                   "사각형에만 둘러싸여 있어 '삼각형 쌍 병합' 으로는 손댈 수 없습니다.")

st.divider()
st.download_button("수리된 .bdf 내려받기",
                   repaired_bdf_text(picked_name, max_dihedral, max_skew, max_aspect, use_swap, int(corrupt_count),
                                     int(seed), render_version()),
                   file_name=f"{picked_name}_repaired.bdf", mime="text/plain")

with st.expander("색이 뜻하는 것"):
    st.markdown(
        """
| 색 | 의미 |
|---|---|
| 🔴 빨강 (왼쪽) | 이 연산으로 없어질 요소 |
| 🟢 초록 (오른쪽) | 새로 생긴 요소 |
| ⬛ 검은 테두리 | 불량각 요소 (내각 < 30° 또는 > 150°) |
| 🔵 파랑 | 1D 요소 (CBAR/CBEAM) |
| 🟠 주황 | 양옆 삼각형을 합치려면 **같이 지워야 하는** 1D 요소 |
| 하늘색 원 | ext 절점 — 이웃 패치와 공유하는 절단면이라 움직일 수 없다 |
"""
    )

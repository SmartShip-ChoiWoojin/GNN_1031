"""메쉬 수리 뷰어 — .bdf 를 골라 수리 전/후를 나란히 본다.

분류 프로젝트의 ``../src/app.py`` 와 같은 조작감을 쓴다: 이름 옆에 라벨이 붙은
드롭다운, 부분 문자열 검색, 좌우 분할.

    streamlit run GNN_GAE/src/app.py

두 3D 화면은 **시점이 연동**된다. 한쪽을 돌리면 반대쪽이 같은 각도로 따라오므로
무엇이 바뀌었는지 눈으로 바로 비교된다. 연동 스크립트가 iframe 안에서 돌아야 해서
plotly 차트를 ``st.plotly_chart`` 로 두 개 그리지 않고, 뷰어 HTML 한 덩어리를
``components.html`` 로 띄운다 (Streamlit 은 차트 사이 JS 통신을 막는다).
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import gae_paths as gp
import features as F
from bdf_io import discover_pairs
from fasoo_drm import setting_fasoo
from labels import load_label_table
from meshedit import EditableMesh, EditError, merge_candidates
from repair_rule import RepairConfig, repair
from viz_compare import GROUPS, corrupt, render_html, stats  # noqa: F401

# 라벨 엑셀이 fasoo 로 암호화돼 있을 수 있으므로 파일을 열기 전에 한 번 해제한다.
# (중복 호출 가드가 있어서 Streamlit 이 매 조작마다 스크립트를 다시 실행해도 안전)
setting_fasoo()

st.set_page_config(page_title="메쉬 수리 뷰어", layout="wide")

LABEL_TAGS = {0: "정상", 1: "비정상", 2: "분류어려움"}


def show_table(df, **kw) -> None:
    """표를 가로 폭에 맞춰 그린다.

    Streamlit 1.49 를 기점으로 폭 지정 인자가 ``use_container_width=True`` 에서
    ``width="stretch"`` 로 바뀌었다. 구버전에 문자열을 주면 ``proto.width = width``
    에서 ``TypeError: 'str' object cannot be interpreted as an integer`` 가 난다.
    이 프로젝트는 SHI_GNN(1.60)에서 쓰지만 base(1.45)로 실행하는 경우가 있어
    양쪽을 다 받아준다.
    """
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


# ──────────────────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def available_samples() -> list[str]:
    pairs, _, _ = discover_pairs(gp.FE_DIR)
    return sorted(pairs)


@st.cache_data
def pair_paths() -> dict[str, tuple[str, str]]:
    pairs, _, _ = discover_pairs(gp.FE_DIR)
    return pairs


@st.cache_data
def label_lookup() -> dict[str, tuple[int, int]]:
    df = load_label_table(gp.LABEL_XLSX)
    return {r.name: (int(r.cls), int(r.ambig)) for r in df.itertuples()}


def render_ver() -> float:
    """``viz_compare.py`` 가 바뀌면 아래 캐시를 버리게 하는 값.

    ``st.cache_data`` 는 데코레이터가 붙은 함수의 코드만 해시한다. 그래서 뷰어 쪽
    (``viz_compare``)만 고치면 캐시가 그대로 남아 "코드를 고쳤는데 화면이 안 바뀐다"
    가 된다 — 실제로 한 번 겪었다. 파일 수정시각을 캐시 키에 끼워 자동으로 비운다.
    """
    import viz_compare

    return os.path.getmtime(viz_compare.__file__)


@st.cache_data
def repair_index() -> pd.DataFrame | None:
    """``scan_repairs.py`` 가 남긴 전수 조사 결과. 없으면 None (앱은 그대로 동작한다)."""
    p = os.path.join(gp.OUT_DIR, "repairs_cls1.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def label_tag(name: str) -> str:
    lab = label_lookup().get(name)
    if lab is None:
        return "(라벨 없음)"
    cls, ambig = lab
    if cls == 2:
        return "(분류어려움·애매)" if ambig else "(분류어려움)"
    return f"(애매·{LABEL_TAGS[cls]})" if ambig else f"({LABEL_TAGS[cls]})"


def pick_by_label(names: list[str], key: str, title: str) -> str | None:
    """이름 옆에 라벨을 붙여 고르게 한다. ``../src/app.py`` 와 같은 동작."""
    query = st.text_input(
        "검색 (부분 문자열, 예: 0694)", key=key,
        help="입력한 글자를 이름 어디에든 포함하는 항목만 드롭다운에 남습니다.",
    )
    disp = [f"{n}  {label_tag(n)}" for n in names]
    pairs = [(d, n) for d, n in zip(disp, names) if not query or query.lower() in d.lower()]
    if not pairs:
        st.warning(f"'{query}' 를 포함하는 항목이 없습니다.")
        return None
    pick = st.selectbox(f"{title} ({len(pairs)}개)", [d for d, _ in pairs], index=0)
    return dict(pairs)[pick]


# ──────────────────────────────────────────────────────────────────────────
# 수리 — 무거우니 캐시한다. 캐시 키는 아래 인자 전부.
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_repair(name: str, max_dih: float, max_skew: float, max_aspect: float,
               use_swap: bool, n_corrupt: int, seed: int, ver: float) -> dict:
    """선택한 메쉬를 수리하고 뷰어 HTML 과 요약을 돌려준다."""
    import random

    bdf, js = pair_paths()[name]
    base = EditableMesh.load(bdf, js)
    n_done = 0
    if n_corrupt:
        base, n_done = corrupt(base, n_corrupt, random.Random(seed))

    cfg = RepairConfig(max_dihedral=max_dih, max_skew=max_skew,
                       max_aspect=max_aspect, use_swap=use_swap)
    fixed, ops = repair(base, cfg)

    sb, sa = stats(base), stats(fixed)
    sub = f"연산 {len(ops)}회 · 삼각형 {sb['삼각형']} → {sa['삼각형']}"
    html = render_html(base, fixed, ops, sub, embed=True)

    # 왜 더 못 고쳤는지 — 막힌 후보를 사유별로 센다
    blocked: dict[str, int] = {}
    for c in merge_candidates(fixed):
        if c.blocked:
            key = ("1D 요소가 변 위에 있음" if "1D" in c.blocked
                   else "합치면 오목 사각형" if "오목" in c.blocked
                   else "다른 패널 (이면각 큼)" if "패널" in c.blocked
                   else "표면 면적이 바뀜" if "면적" in c.blocked else c.blocked)
        elif c.dihedral > cfg.max_dihedral:
            key = "다른 패널 (이면각 큼)"
        elif c.quality is not None and F.is_bad_shell(c.quality):
            key = "결과가 불량각 요소"
        elif c.quality is not None and c.quality.skew > cfg.max_skew:
            key = f"결과 skew > {cfg.max_skew:.0f}°"
        elif c.quality is not None and c.quality.aspect > cfg.max_aspect:
            key = f"결과 aspect > {cfg.max_aspect:.1f}"
        else:
            key = "남은 후보 (적용 가능)"
        blocked[key] = blocked.get(key, 0) + 1

    # 수리 뒤에도 남은 삼각형 중 짝이 아예 없는 것
    paired = set()
    for c in merge_candidates(fixed):
        paired |= {c.si, c.sj}
    lonely = sum(1 for i in fixed.tri_indices() if i not in paired)

    return {
        "html": html,
        "ops": [(o.kind, o.key) for o in ops],
        "before": sb,
        "after": sa,
        "n_corrupt": n_done,
        "blocked": blocked,
        "lonely_tri": lonely,
        "bdf": fixed.to_raw(),
    }


@st.cache_data(show_spinner=False)
def repaired_bdf_text(name: str, max_dih: float, max_skew: float, max_aspect: float,
                      use_swap: bool, n_corrupt: int, seed: int, ver: float) -> str:
    """수리된 메쉬를 .bdf 문자열로. 내려받기 버튼용."""
    import random
    import tempfile

    bdf, js = pair_paths()[name]
    base = EditableMesh.load(bdf, js)
    if n_corrupt:
        base, _ = corrupt(base, n_corrupt, random.Random(seed))
    fixed, _ = repair(base, RepairConfig(max_dihedral=max_dih, max_skew=max_skew,
                                         max_aspect=max_aspect, use_swap=use_swap))
    p = os.path.join(tempfile.mkdtemp(), f"{name}_repaired.bdf")
    fixed.write_bdf(p)
    with open(p, encoding="utf-8") as fh:
        return fh.read()


# ──────────────────────────────────────────────────────────────────────────
# 화면
# ──────────────────────────────────────────────────────────────────────────
st.title("메쉬 수리 전/후 뷰어")
st.caption(
    "비정상 메쉬의 삼각형을 사각형으로 되돌린다. 두 3D 화면은 시점이 연동되므로 "
    "한쪽을 돌리면 반대쪽도 같은 각도로 따라온다."
)

names = available_samples()
idx = repair_index()

with st.sidebar:
    st.header("메쉬 고르기")
    scope = st.radio(
        "목록",
        ["수리가 일어나는 것만", "비정상 전체", "전체"] if idx is not None
        else ["비정상 전체", "전체"],
        help="'수리가 일어나는 것만' 은 scan_repairs.py 가 미리 조사한 결과를 씁니다 — "
             "규칙 기반 수리기가 실제로 손댈 수 있는 메쉬입니다.",
    )
    if scope == "수리가 일어나는 것만" and idx is not None:
        shown = [n for n in idx[idx.n_op > 0].sort_values("n_op", ascending=False)["name"]
                 if n in set(names)]
        st.caption(f"비정상 {len(idx)}개 중 {len(shown)}개 ({len(shown)/len(idx):.0%}) — "
                   "연산 많은 순")
    elif scope == "비정상 전체":
        lab = label_lookup()
        shown = [n for n in names if lab.get(n, (9, 0))[0] == 1]
    else:
        shown = names

    name = pick_by_label(shown, "pick", "모델 이름")

    st.divider()
    st.header("수리 설정")
    use_swap = st.checkbox("swap 보조 탐색", value=True,
                           help="바로는 못 합치는 쌍을 한 번 뒤집어 합칠 수 있는지 본다. "
                                "실측으로는 이득이 0이었다.")
    max_dih = st.slider("이면각 상한 (°)", 0.0, 45.0, 10.0, 1.0,
                        help="두 삼각형이 이보다 꺾여 있으면 서로 다른 패널로 보고 "
                             "합치지 않는다. 높이면 선체 형상이 바뀔 수 있다.")
    max_skew = st.slider("결과 skew 상한 (°)", 10.0, 90.0, 45.0, 5.0,
                         help="합친 사각형이 이보다 찌그러지면 합치지 않는다.")
    max_aspect = st.slider("결과 aspect 상한", 1.5, 10.0, 4.0, 0.5)

    st.divider()
    st.header("손상 주입 (선택)")
    st.caption("정상 메쉬의 사각형을 일부러 쪼갠 뒤 수리한다. 자기지도 학습 데이터의 "
               "맛보기이자, 수리기가 제대로 도는지 보는 검증이다.")
    n_corrupt = st.number_input("쪼갤 사각형 수", 0, 30, 0, 1)
    seed = st.number_input("seed", 0, 9999, 0, 1)

if not name:
    st.stop()

with st.spinner("수리하는 중..."):
    r = run_repair(name, max_dih, max_skew, max_aspect, use_swap,
                   int(n_corrupt), int(seed), render_ver())

sb, sa = r["before"], r["after"]
head = f"**{name}** {label_tag(name)}"
if r["n_corrupt"]:
    head += f" · 사각형 {r['n_corrupt']}개를 일부러 쪼갠 뒤 수리"
st.markdown(head)

c1, c2, c3, c4 = st.columns(4)
c1.metric("적용한 연산", f"{len(r['ops'])}회")
c2.metric("삼각형", sa["삼각형"], delta=sa["삼각형"] - sb["삼각형"], delta_color="inverse")
c3.metric("삼각형 비율", sa["삼각형 비율"],
          delta=f"{float(sa['삼각형 비율']) - float(sb['삼각형 비율']):+.3f}",
          delta_color="inverse")
c4.metric("불량각 요소", sa["불량각 요소"],
          delta=sa["불량각 요소"] - sb["불량각 요소"], delta_color="inverse")

if not r["ops"]:
    st.info(
        "이 메쉬에서는 적용할 수 있는 연산이 없어 좌우가 동일합니다. "
        "아래 **왜 더 못 고쳤나** 를 보면 후보가 무엇에 막혔는지 알 수 있습니다."
    )

components.html(r["html"], height=640, scrolling=False)

left, right = st.columns([3, 2])

with left:
    st.subheader("적용한 연산")
    if r["ops"]:
        show_table(
            pd.DataFrame(
                [{"연산": {"merge": "삼각형 2개 → 사각형", "swap": "대각선 뒤집기",
                           "split": "사각형 → 삼각형 2개"}.get(k, k),
                  "절점": ", ".join(str(x) for x in key)} for k, key in r["ops"]],
                index=range(1, len(r["ops"]) + 1),
            )
        )
    else:
        st.write("없음")

with right:
    st.subheader("왜 더 못 고쳤나")
    st.caption("수리 후에도 남아 있는 인접 삼각형 쌍이 무엇에 막혀 있는지")
    if r["blocked"]:
        show_table(
            pd.DataFrame(sorted(r["blocked"].items(), key=lambda kv: -kv[1]),
                         columns=["사유", "쌍 수"]),
            hide_index=True,
        )
    else:
        st.write("남은 인접 삼각형 쌍이 없습니다.")
    if r["lonely_tri"]:
        st.warning(
            f"수리 후 남은 삼각형 중 **{r['lonely_tri']}개는 인접 삼각형이 아예 없습니다** — "
            "사각형에만 둘러싸여 있어 '삼각형 쌍 병합' 으로는 손댈 수 없습니다. "
            "국소 재사각형화 연산이 필요한 지점입니다."
        )

st.divider()
st.download_button(
    "수리된 .bdf 내려받기",
    repaired_bdf_text(name, max_dih, max_skew, max_aspect, use_swap,
                      int(n_corrupt), int(seed), render_ver()),
    file_name=f"{name}_repaired.bdf",
    mime="text/plain",
)

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

요소를 클릭하면 내각·skew·aspect·절점 번호가 뜬다. 도구막대 체크박스로 각 요소를
켜고 끌 수 있다.
"""
    )

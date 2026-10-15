"""Bad ↔ Good 쌍 뷰어 — 사람이 무엇을 고쳤는지 직접 보는 화면.

왼쪽이 원본(Bad), 오른쪽이 사람이 고친 것(Good). 두 화면은 시점이 연동되므로 한쪽을
돌리면 반대쪽도 같은 각도로 따라온다.

    streamlit run GNN_GAE/src/app_pairs.py

색:
    빨강 (왼쪽)   이 수정으로 **없어진** 요소
    초록 (오른쪽) **새로 생긴** 요소
    자주 X        사라지거나 새로 생긴 **절점**
    회색          그대로, 검정 테두리 = 불량각 요소, 파랑 = 1D 요소, 하늘색 = ext 절점
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import gae_paths as gp
import paths as parent_paths
from bdf_io import discover_pairs
from fasoo_drm import setting_fasoo
from meshedit import EditableMesh
from meshrep import Edge
from pairs import all_pairs, discover_good_pairs, load_pair
from viz_compare import render_html, stats

setting_fasoo()
st.set_page_config(page_title="Bad ↔ Good 쌍 뷰어", layout="wide")


def show_table(df, **kw) -> None:
    """Streamlit 1.49 에서 표 폭 인자가 바뀌어서 양쪽 버전을 다 받아준다."""
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def show_html(html: str, height: int) -> None:
    """HTML 을 iframe 으로 띄운다.

    ``st.components.v1.html`` 은 2026-06-01 이후 제거 예정이고 ``st.iframe`` 으로
    바뀌었다. 구버전(1.45)에는 ``st.iframe`` 이 없으므로 있는 쪽을 쓴다.
    """
    fn = getattr(st, "iframe", None)
    if fn is not None:
        # st.iframe 은 scrolling 인자를 받지 않는다 (components.html 과 다른 점)
        fn(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)


# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def pair_names() -> list[str]:
    return all_pairs()


@st.cache_data
def path_maps():
    return discover_good_pairs(), discover_pairs(gp.FE_DIR)[0]


@st.cache_data
def judge_probs() -> pd.DataFrame | None:
    """judge_pairs.py 가 저장한 수정 전/후 비정상 확률. 없으면 None."""
    p = os.path.join(gp.OUT_DIR, "judge_pairs.csv")
    return pd.read_csv(p).set_index("name") if os.path.exists(p) else None


def classify(p) -> str:
    """수정의 종류 (export_pair_list.kind_of 와 같은 규칙).

    네 축으로 판정하되 이름에서는 '변변화' 를 뺀다 — 508쌍 중 503쌍에 붙어 있어
    이름에 넣으면 길기만 하고 구분이 안 된다. 대신 **연결이 안 바뀐 5건**만 따로 적는다:
    절점도 연결도 그대로면 "변화없음"(4건), 절점만 움직이고 연결은 그대로면
    "(연결 변화 없음)" 을 덧붙인다(`_0267` 1건).

    '절점병합' 과 '신규교체' 를 축에서 내린 이유: 파일 수준에서는 둘 다 **절점이
    사라진 것**이고, 그 참조를 누가 이어받았는지는 세부 속성이다 (451 대 12 대 18 로
    쏠려 분류 값어치가 없다). 승계 정보는 열/표에 그대로 남겨 둔다.
    """
    n = p.nodes
    _, rm, cr = p.shell_diff()
    eb = {Edge(e) for _, _, nd in p.before.shells for e in zip(nd, nd[1:] + nd[:1])}
    ea = {Edge(e) for _, _, nd in p.after.shells for e in zip(nd, nd[1:] + nd[:1])}
    edge_chg = eb != ea
    gone = set(p.before.coords) - set(p.after.coords)
    parts = []
    if gone:
        parts.append("절점삭제")
    if n.added:
        parts.append("절점추가")
    if n.moved:
        parts.append("좌표이동")
    if not parts:
        return "연결만 변경" if edge_chg else "변화없음"
    return " + ".join(parts) + ("" if edge_chg else " (연결 변화 없음)")


@st.cache_data(show_spinner=False)
def pair_view(name: str, ver: float) -> dict:
    """쌍 하나의 뷰어 HTML 과 요약. ver 는 뷰어 코드가 바뀌면 캐시를 버리게 하는 값."""
    good, orig = path_maps()
    p = load_pair(name, good, orig)
    before, after = EditableMesh(p.before), EditableMesh(p.after)
    _, removed, created = p.shell_diff()
    kind = classify(p)

    html = render_html(
        before, after, [], "", embed=True,
        # 자주색 X = 구조가 바뀐 절점 (사라짐 / 새로 생김)
        node_marks_before={n: f"절점 {n} — 이 수정으로 삭제됨 (이어받은 절점 없음)"
                           for n in p.nodes.deleted}
        | {b: f"절점 {b} — {a} 에 흡수되어 사라짐 (병합)" for b, a in p.nodes.absorbed.items()}
        | {b: f"절점 {b} — 새로 만든 {a} 가 대신함" for b, a in p.nodes.replaced_by_new.items()},
        node_marks_after={n: f"절점 {n} — 새로 생김" for n in p.nodes.added},
        # 노란 마름모 = 자리를 옮긴 절점. 흡수한 쪽이 움직였다면 **출발 위치**도 찍는다.
        # 이게 없으면 "어디서 어디로 갔는지" 가 안 보이고, 도착 위치가 흡수된 절점의
        # 자리와 정확히 겹쳐서 좌우 화면에 같은 마커가 찍힌 것처럼 보인다 (실측 `_0014`).
        move_marks_before={n: f"절점 {n} — 여기 있다가 옮겨감" for n in p.nodes.moved}
        | {s: f"절점 {s} — 여기 있다가 "
              f"{', '.join(map(str, p.nodes.survivors.get(s, [])))} 자리로 옮겨가며 흡수"
           for s in p.nodes.survivor_landed}
        | {a: f"절점 {a} — {b} 와 좌표가 겹침 (확인 필요)" for a, b in p.nodes.coincident.items()},
        move_marks_after={n: f"절점 {n} — 여기로 옮겨옴" for n in p.nodes.moved}
        | {a: f"절점 {a} — {', '.join(map(str, bs))} 를 흡수 "
              f"({p.nodes.survivor_landed.get(a, '제자리')})"
           for a, bs in p.nodes.survivors.items()}
        | {a: f"절점 {a} — {b} 와 좌표가 겹침 (확인 필요)" for a, b in p.nodes.coincident.items()},
    )
    return {
        "html": html, "kind": kind,
        "before": stats(before), "after": stats(after),
        "n_removed": len(removed), "n_created": len(created),
        "deleted": sorted(p.nodes.deleted), "added": sorted(p.nodes.added),
        "moved": sorted(p.nodes.moved), "survivors": p.nodes.survivors,
        "replaced": dict(p.nodes.replaced_by_new),
        "landed": dict(p.nodes.survivor_landed),
        "basis": dict(p.nodes.basis), "explained": p.nodes.explained,
        "coincident": dict(p.nodes.coincident),
        "n_line_before": len(p.before.lines), "n_line_after": len(p.after.lines),
    }


def render_ver() -> float:
    import viz_compare

    return os.path.getmtime(viz_compare.__file__)


# ──────────────────────────────────────────────────────────────────────────
st.title("Bad ↔ Good 쌍 뷰어")
st.caption(
    "왼쪽이 원본, 오른쪽이 사람이 고친 것. 한쪽을 돌리면 반대쪽도 같은 각도로 따라옵니다. "
    "빨강 = 없어진 요소, 초록 = 새로 생긴 요소, 자주색 X = 바뀐 절점."
)

names = pair_names()
jp = judge_probs()

with st.sidebar:
    st.header("메쉬 고르기")

    kinds = st.multiselect(
        "수정 종류로 거르기",
        ["절점삭제", "연결만 변경", "절점삭제 + 절점추가", "절점추가",
         "좌표이동", "절점삭제 + 좌표이동", "변화없음",
         "좌표이동 (연결 변화 없음)"],
        default=[],
        help="비워 두면 전부 봅니다. 종류를 판별하려면 쌍을 열어봐야 해서 "
             "처음 고를 때 시간이 조금 걸립니다.",
    )

    shown = names
    if kinds:
        good, orig = path_maps()
        with st.spinner("종류를 판별하는 중..."):
            shown = [n for n in names if classify(load_pair(n, good, orig)) in kinds]
        st.caption(f"{len(shown)} / {len(names)}개")

    order = st.radio(
        "정렬",
        ["많이 좋아진 순", "이름순", "오히려 나빠진 순"],
        help="'오히려 나빠진 순' 은 사람이 고쳤는데 심판 확률이 올라간 쌍입니다 — "
             "508개 중 13개뿐이고, 라벨이나 수정 작업을 되짚어볼 후보입니다.",
    )
    if jp is not None and order != "이름순":
        delta = (jp["p_after"] - jp["p_before"])
        rank = delta.sort_values(ascending=(order == "많이 좋아진 순")).index
        shown = [n for n in rank if n in set(shown)]

    query = st.text_input("검색 (부분 문자열, 예: 0006)")
    cand = [n for n in shown if not query or query.lower() in n.lower()]
    if not cand:
        st.warning("해당하는 메쉬가 없습니다.")
        st.stop()

    def label(n: str) -> str:
        if jp is not None and n in jp.index:
            r = jp.loc[n]
            return f"{n}   p {r.p_before:.2f} → {r.p_after:.2f}"
        return n

    pick = st.selectbox(f"메쉬 ({len(cand)}개)", [label(n) for n in cand], index=0)
    name = cand[[label(n) for n in cand].index(pick)]

    st.divider()
    st.caption(
        "p 는 심판 분류기(`deploy_10_14`)가 본 비정상 확률입니다. "
        "이 모델은 이 메쉬들을 학습에 쓰지 않았습니다."
    )

with st.spinner("쌍을 불러오는 중..."):
    v = pair_view(name, render_ver())

sb, sa = v["before"], v["after"]

head = f"**{name}** · 수정 종류: **{v['kind']}**"
if jp is not None and name in jp.index:
    r = jp.loc[name]
    head += f" · 심판 p {r.p_before:.3f} → {r.p_after:.3f}"
st.markdown(head)

c = st.columns(5)
c[0].metric("없어진 요소", v["n_removed"])
c[1].metric("새로 생긴 요소", v["n_created"])
c[2].metric("쉘 수", sa["쉘 요소"], delta=sa["쉘 요소"] - sb["쉘 요소"])
c[3].metric("삼각형", sa["삼각형"], delta=sa["삼각형"] - sb["삼각형"], delta_color="inverse")
c[4].metric("불량각 요소", sa["불량각 요소"],
            delta=sa["불량각 요소"] - sb["불량각 요소"], delta_color="inverse")

show_html(v["html"], height=660)

left, right = st.columns(2)
with left:
    st.subheader("바뀐 절점")
    rows = ([{"절점": n, "무엇": "삭제됨"} for n in v["deleted"]]
            + [{"절점": a,
                "무엇": f"{', '.join(map(str, bs))} 를 흡수 (병합, {v['landed'].get(a, '제자리')})"}
               for a, bs in v["survivors"].items()]
            + [{"절점": b, "무엇": f"새로 만든 {a} 가 대신함"} for b, a in v["replaced"].items()]
            + [{"절점": n, "무엇": "새로 생김"} for n in v["added"]]
            + [{"절점": n, "무엇": "좌표 이동"} for n in v["moved"]]
            + [{"절점": a, "무엇": f"{b} 와 좌표 겹침 — 확인 필요"}
               for a, b in v["coincident"].items()])
    if rows:
        show_table(pd.DataFrame(rows), hide_index=True)
    else:
        st.write("없음 — 절점은 그대로 두고 요소 연결만 바뀌었습니다.")

with right:
    st.subheader("전/후 지표")
    show_table(
        pd.DataFrame(
            # 값을 문자열로 통일한다 — '삼각형 비율' 만 포맷된 문자열이라 한 열에
            # 숫자와 섞이면 Arrow 변환이 실패한다.
            [{"지표": k, "원본(Bad)": str(sb[k]), "수정(Good)": str(sa[k])} for k in sb]
            + [{"지표": "1D 요소", "원본(Bad)": str(v["n_line_before"]),
                "수정(Good)": str(v["n_line_after"])}]
        ),
        hide_index=True,
    )

with st.expander("색이 뜻하는 것"):
    st.markdown(
        """
| 색 | 의미 |
|---|---|
| 🔴 빨강 (왼쪽) | 이 수정으로 **없어진** 요소 |
| 🟢 초록 (오른쪽) | **새로 생긴** 요소 |
| 🟣 자주색 X | 사라지거나 새로 생긴 절점 (좌표만 옮긴 절점은 양쪽에 모두 표시) |
| ⬛ 검은 테두리 | 불량각 요소 (내각 < 30° 또는 > 150°) |
| 🔵 파랑 | 1D 요소 (CBAR/CBEAM) |
| 🟠 주황 | 양옆 삼각형을 합치려면 **같이 지워야 하는** 1D 요소 (사람도 손댄 구역 1D 의 19%를 지웁니다) |
| 하늘색 원 | ext 절점 — 이웃 패치와 공유하는 절단면 |

요소를 클릭하면 내각·skew·aspect·절점 번호가 뜹니다. 도구막대 체크박스로 각각 켜고 끌 수 있습니다.
"""
    )

"""메쉬 재생성 결과 화면 — 원본 / 모델 재생성 결과 / 사람 수정 결과를 나란히, 세 화면 시점 연동.

    streamlit run GNN_GAE/src/app_regen.py

    왼쪽 (세로 전체)  원본(비정상 메쉬) — 지워진 element·node 를 누가 지웠는지 색으로
    오른쪽 위         모델 재생성 결과 — 지우기·만들기가 반영된 최종 메쉬, 새로 만든 element 초록
    오른쪽 아래       사람 수정 결과   — 같은 방식
    색: 왼쪽 빨강 = 둘 다 삭제·변형, 주황 = 모델만, 보라 = 사람만 / 오른쪽 초록 = 새로 만든 element
    (색 규칙과 '원본에 표시할 삭제' 보기 전환은 viz_regen.py 머리말 참고)

모델을 여기서 돌리지 않는다. 채점 때 기록한 연산(어느 절점을 어디로 합쳤고 어느 삼각형
쌍을 합쳤는지)을 원본에 **다시 적용**해 결과를 만든다 — 그래서 화면은 채점된 결과와
정확히 같고, torch 없이도 열린다.
    테스트 셋   runs/regen_<tag>/eval_test.csv  (regen_eval.py 가 만든다)
    학습 셋     runs/regen_<tag>/cv_pred.csv   (regen_train.py 의 out-of-fold 예측 —
                                               각 메쉬를 자기를 학습에 안 쓴 모델이 만든 것)
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import gae_paths as gp
from bdf_io import discover_pairs
from fasoo_drm import setting_fasoo
from meshedit import EditableMesh, check, replay_ops
from pairs import discover_good_pairs, load_pair

setting_fasoo()
st.set_page_config(page_title="메쉬 재생성 결과", layout="wide")


def show_table(df, **kw) -> None:
    """Streamlit 1.49 에서 표 폭 인자가 바뀌어서 양쪽 버전을 다 받아준다."""
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


# ──────────────────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────────────────
def list_runs() -> list[str]:
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(gp.RUNS_DIR, "regen_*", "model.json")))


@st.cache_data
def run_meta(run: str, ver: float) -> dict:
    with open(os.path.join(run, "model.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    ev = os.path.join(run, "eval_test.json")
    if os.path.exists(ev):
        with open(ev, encoding="utf-8") as fh:
            meta["test"] = json.load(fh)
    return meta


@st.cache_data
def table_path(run: str, split: str) -> str:
    """테스트는 eval_test.csv. 학습 셋은 out-of-fold 를 같은 지표로 다시 잰 eval_cv.csv 가 있으면
    그것을(심판 확률·검사 포함), 없으면 학습 때 남긴 cv_pred.csv 를 쓴다."""
    if split == "test":
        return os.path.join(run, "eval_test.csv")
    p = os.path.join(run, "eval_cv.csv")
    return p if os.path.exists(p) else os.path.join(run, "cv_pred.csv")


def load_table(run: str, split: str, ver: float) -> pd.DataFrame | None:
    path = table_path(run, split)
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    for c in ("exact", "exact_shells", "did_nothing"):
        if c in d:
            d[c] = d[c].astype(str).str.lower().eq("true")
    # 분할 파일의 묶음/fold 를 붙인다 — 같은 결함을 고친 다른 쌍이 몇 개인지 보이려고
    from regen_data import load_split

    sp = load_split()
    sizes = sp.groupby("group").name.size()
    d = d.drop(columns=[c for c in ("cv_fold", "group") if c in d]).merge(
        sp[["name", "cv_fold", "group"]], on="name", how="left")
    d["group_size"] = d.group.map(sizes).fillna(1).astype(int)
    return d


@st.cache_data
def path_maps():
    return discover_good_pairs(), discover_pairs(gp.FE_DIR)[0]


def parse_collapses(s: str) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in t.split(">")) for t in s.split(";") if t]


def parse_edges(s: str) -> list[frozenset]:
    return [frozenset(int(x) for x in t.split("-")) for t in s.split(";") if t]


def replay(bad: EditableMesh, collapses: str, merges: str) -> EditableMesh:
    """기록된 연산을 원본에 다시 적용한다 (모델을 돌리지 않는다)."""
    return replay_ops(bad, parse_collapses(collapses), parse_edges(merges))


def code_ver() -> float:
    """화면 결과를 만드는 모듈이 바뀌면 캐시를 버린다 — st.cache_data 는 데코레이터가
    붙은 함수의 코드만 해시하므로 다른 모듈을 고쳐도 캐시가 남는다 (README 8절)."""
    import meshdiff
    import meshedit
    import viz_regen

    return max(os.path.getmtime(m.__file__) for m in (meshdiff, meshedit, viz_regen))


@st.cache_data(show_spinner=False)
def build(name: str, collapses: str, merges: str, ver: float) -> dict:
    from viz_regen import build_view, render_html

    good, orig = path_maps()
    p = load_pair(name, good, orig)
    bad = EditableMesh(p.before)
    model = replay(bad, collapses, merges)
    human = EditableMesh(p.after)
    absorbed = {r: k for r, k in parse_collapses(collapses)}
    panes, changes, summary = build_view(bad, model, human, absorbed, dict(p.nodes.absorbed))
    med = bad.rep.median_edge_length
    ck = check(model, bad, regen=True)
    return dict(
        html=render_html(panes, changes, focus_radius=1.2 * med),
        changes=pd.DataFrame([dict(누가=c.who, 변경=c.what, 대상=c.target,
                                   상대쪽도=("✓" if c.shared else "✗"), 설명=c.note) for c in changes]),
        summary=summary, viol=list(ck.violations), warn=list(ck.warnings),
        bdf=model.bdf_text(), json=model.json_text(),
    )


# ──────────────────────────────────────────────────────────────────────────
st.title("메쉬 재생성 결과")
st.caption("왼쪽은 원본(비정상 메쉬), 오른쪽 위는 모델 재생성 결과, 오른쪽 아래는 사람 수정 결과입니다. "
           "세 화면의 시점이 연동됩니다. 왼쪽: 빨강 = 둘 다 지운 element · 주황 = 모델만 지운 element · "
           "보라 = 사람만 지운 element / 오른쪽: 초록 = 새로 만든 element")

runs = list_runs()
if not runs:
    st.error("학습된 재생성 모델이 없습니다. 먼저 `python src/regen_train.py --tag final --loss list --scorer2 gbm --seeds 0,1,2` 를 돌리세요.")
    st.stop()

with st.sidebar:
    st.header("모델과 메쉬 고르기")
    default = next((i for i, r in enumerate(runs) if os.path.basename(r) == "regen_final"), 0)

    def run_label(r: str) -> str:
        m = run_meta(r, os.path.getmtime(os.path.join(r, "model.json")))
        t = f" · 테스트 {m['test']['exact']:.0%}" if "test" in m else ""
        return f"{os.path.basename(r)}  (CV {m.get('cv_exact', 0):.0%}{t})"

    run = st.selectbox("재생성 모델", runs, index=default, format_func=run_label)
    meta = run_meta(run, os.path.getmtime(os.path.join(run, "model.json")))
    has_test = os.path.exists(os.path.join(run, "eval_test.csv"))
    split = st.radio("데이터", ["test", "train"], horizontal=True, index=0 if has_test else 1,
                     format_func=lambda s: "테스트 (10%)" if s == "test" else "학습 (90%, CV 예측)",
                     help="테스트 셋은 학습·임계값 조정 어디에도 쓰지 않은 10%이고, 학습 셋과 같은 결함을 공유하지 않습니다. "
                          "학습 셋은 5-fold 로 나눠 각 메쉬를 자기를 학습에 안 쓴 모델이 만든 결과입니다.")
    tab_path = table_path(run, split)
    d = load_table(run, split, os.path.getmtime(tab_path) if os.path.exists(tab_path) else 0.0)
    if d is None:
        st.warning("이 데이터의 결과 파일이 없습니다.\n\n"
                   f"`python src/regen_eval.py --run runs/{os.path.basename(run)}` 를 먼저 돌리세요.")
        st.stop()

    res = st.radio("결과", ["전체", "사람과 일치", "사람과 다름"], horizontal=True)
    kinds = st.multiselect("수정 종류", sorted(d.kind.unique()))
    only_rep = st.checkbox("우리 연산으로 재현 가능한 쌍만", value=False,
                           help="절점 추가·좌표 이동 등 재생성 연산 밖의 수정이 섞인 쌍을 뺍니다 "
                                "(regen_oracle.py 의 '재현됨').")
    query = st.text_input("검색 (예: 0006)")
    sub = d
    if res == "사람과 일치":
        sub = sub[sub.exact]
    elif res == "사람과 다름":
        sub = sub[~sub.exact]
    if kinds:
        sub = sub[sub.kind.isin(kinds)]
    if only_rep:
        sub = sub[sub.oracle == "재현됨"]
    if query:
        sub = sub[sub.name.str.contains(query, case=False)]
    st.caption(f"{len(sub)} / {len(d)}쌍")
    if sub.empty:
        st.warning("해당하는 메쉬가 없습니다.")
        st.stop()
    sub = sub.sort_values(["exact", "change_iou", "name"], ascending=[True, True, True])
    labels = {r["name"]: f"{'✓' if r['exact'] else '✗'} {r['name'].replace('SN2592_FWD_Clean_', '_')}"
                         f"  IoU {float(r['change_iou']):.2f}  {r['kind']}"
              for _, r in sub.iterrows()}
    name = st.selectbox(f"메쉬 ({len(sub)}개)", list(labels), format_func=lambda n: labels[n])

    st.divider()
    st.metric("정확 일치 (사람과 똑같이 재생성)", f"{d.exact.mean():.1%}",
              help="요소·1D·좌표까지 사람 결과와 같은 비율. 절점 번호는 보지 않습니다.")
    c1, c2 = st.columns(2)
    c1.metric("바뀐 요소 IoU", f"{d.change_iou.astype(float).mean():.3f}")
    c2.metric("상한 (정답 대본)", f"{(d.oracle == '재현됨').mean():.1%}",
              help="사람의 수정을 우리 연산으로 그대로 옮겼을 때 재현되는 비율 — 이 연산 체계의 천장")
    st.caption(f"임계값 1단계 {meta['thr1']:.2f} / 2단계 {meta['thr2']:.2f} · 채점기 {meta['scorer']}"
               f" · 학습 {meta['n_fit']}쌍")

row = d[d.name == name].iloc[0]
with st.spinner("재생성 결과를 그리는 중..."):
    v = build(name, row["collapses"], row["merges"], code_ver())

head = (f"**{name}** · {row['kind']} · "
        + ("✅ 사람과 **정확히 일치**" if row["exact"] else "❌ 사람과 **다름**")
        + f" · 바뀐 요소 IoU **{float(row['change_iou']):.2f}**")
if row["oracle"] != "재현됨":
    head += f" · ⚠️ 이 쌍은 우리 연산으로 재현 불가 ({row['oracle']})"
st.markdown(head)
note = []
if split == "train" and int(row["cv_fold"]) > 0:
    note.append(f"CV fold {int(row['cv_fold'])} 모델이 만든 결과 (이 메쉬를 학습에 쓰지 않은 모델)")
if int(row["group_size"]) > 1:
    note.append(f"같은 결함을 고친 쌍이 이 밖에 {int(row['group_size']) - 1}개 더 있음 — "
                "분할할 때 한 묶음으로 같은 쪽에 두었습니다")
if note:
    st.caption(" · ".join(note))
if "p_bad" in row and str(row["p_bad"]) != "":
    st.caption(f"심판 분류기의 비정상 확률: 원본 {float(row['p_bad']):.3f} → "
               f"모델 {float(row['p_regen']):.3f} · 사람 {float(row['p_good']):.3f}")

sm, sh = v["summary"]["model"], v["summary"]["human"]
cols = st.columns(len(sm))
for c, k in zip(cols, sm):
    c.metric(k, f"모델 {sm[k]} · 사람 {sh[k]}")

show_html(v["html"], height=960)

t1, t2, t3, t4 = st.tabs(["변경 목록", "검사", "전/후 지표", "내려받기"])
with t1:
    only_diff = st.checkbox("상대 쪽에 없는 변경만", value=False)
    ch = v["changes"]
    if only_diff:
        ch = ch[ch["상대쪽도"] == "✗"]
    if ch.empty:
        st.write("변경이 없습니다.")
    else:
        show_table(ch, hide_index=True)
    st.caption("화면 위 '변경 위치로 확대…' 에서 같은 목록을 골라 그 자리로 갈 수 있습니다.")
with t2:
    if v["viol"]:
        st.error("불변식 위반:\n\n" + "\n".join(f"- {x}" for x in v["viol"]))
    else:
        st.success("불변식 위반 없음 (1D 는 쉘 변 위, 비매니폴드 변 증가 없음, 법선 반전 없음, "
                   "ext 절점 이동 없음, 총 면적 변화 5% 이내)")
    if v["warn"]:
        st.warning("경고 (사람의 수정에서도 일어나는 일):\n\n" + "\n".join(f"- {x}" for x in v["warn"]))
    ops = parse_collapses(row["collapses"])
    st.markdown(f"**모델이 적용한 연산** — edge collapse(절점 병합) {len(ops)}개, "
                f"triangle-pair merge(삼각형 쌍 병합) {len(parse_edges(row['merges']))}개")
    if ops:
        st.write("edge collapse (remove → keep): " + ", ".join(f"{r} → {k}" for r, k in ops))
    if row["merges"]:
        st.write("triangle-pair merge (공유 변): " + ", ".join(row["merges"].split(";")))
with t3:
    s3 = v["summary"]["stats"]
    show_table(pd.DataFrame([{"지표": k, **{who: str(s3[who][k]) for who in s3}}
                             for k in s3["원본"]]), hide_index=True)
with t4:
    st.download_button("재생성 메쉬 .bdf", v["bdf"], file_name=f"{name}_regen.bdf")
    st.download_button("ext 정보 .json", v["json"], file_name=f"{name}_regen.json")
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
- element 에 마우스를 올리면 내각·skew·node 번호와 누가 지웠는지 / 상대 쪽도 같은 변경을 했는지가 뜹니다.
- 비교는 node **번호가 아니라 좌표**로 합니다. 사람은 두 node 를 합칠 때 남길 번호를 일정하게 고르지
  않아서(흡수한 node 의 번호를 흡수된 자리로 옮기기도 함), 번호로 비교하면 같은 결과도 다르게 나옵니다.
  그래서 변경 목록의 'node 이동' 은 사람 결과에만 나올 수 있습니다 — 모델은 좌표를 옮기지 않습니다.
- node 삭제·이동은 3D 화면에 따로 표시하지 않습니다. 아래 **변경 목록** 표와 '변경 위치로 확대…' 에 있습니다.
"""
    )

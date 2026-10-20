"""재생성 학습/평가 데이터 — 고정된 90/10 분할과 쌍 하나의 학습 표본.

분할은 **한 번 만들어 파일로 남기고, 다시 계산하지 않는다** (`runs/regen_split.csv`).
부모 프로젝트에서 fold 를 화면에서 다시 계산했다가 학습대상 목록이 하나 바뀌자 대부분의
메쉬가 다른 fold 로 가서, '그 메쉬를 학습에 쓴 모델' 이 채점한 사고가 있었다
(커밋 9be64d9). 테스트 10%는 학습·임계값 조정·모델 선택 어디에도 쓰지 않는다.

층화는 수정 종류로 한다 — '절점삭제' 52.6%, '연결만 변경' 38.6%, 나머지 8.9%. 나머지
7종은 각각 몇 건뿐이라 하나로 묶어 층화한다(안 묶으면 1건짜리 종류에서 분할이 안 된다).

**같은 결함은 한쪽에만** 둔다 (``defect_groups``). 패치들이 서로 겹쳐서, 같은 결함을 여러
패치에서 똑같이 고친 경우가 많다 — 508쌍 중 334쌍이 '지운 요소(좌표 기준)를 다른 쌍과
공유' 한다(한 결함이 최대 10개 패치에). 처음 만든 무작위 분할에서는 테스트 51쌍 중
32쌍이 학습 셋에 같은 결함의 쌍둥이를 두고 있었다 — 테스트 점수가 낙관적이 된다.
그래서 (a) 지운 요소를 공유하거나 (b) 원본 요소의 50% 넘게 겹치는 쌍을 한 묶음으로 보고,
묶음 단위로 나눈다. 학습 셋 안의 5-fold 도 같은 묶음 단위로 미리 정해 파일에 적는다.
(이 문제는 테스트 셋을 한 번도 열기 전에 발견해 분할을 다시 만들었다.)

    python src/regen_data.py            # 분할 파일이 없으면 만들고, 있으면 요약만
    python src/regen_data.py --force    # 다시 만든다 (테스트 셋이 바뀐다 — 신중히)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import gae_paths as gp

SPLIT_PATH = os.path.join(gp.RUNS_DIR, "regen_split.csv")
TEST_FRAC = 0.10
SPLIT_SEED = 0
#: 학습 셋 안의 CV fold 수 (묶음 단위)
CV_FOLDS = 5
#: 원본 요소가 이 비율 넘게 겹치는 두 패치는 같은 묶음 (거의 같은 패치)
OVERLAP_FRAC = 0.5


def kind_of(p) -> str:
    """수정 종류 (`app_pairs.classify` 와 같은 규칙). streamlit 을 import 하지 않으려고 따로 둔다."""
    from meshrep import Edge

    n = p.nodes
    eb = {Edge(e) for _, _, nd in p.before.shells for e in zip(nd, nd[1:] + nd[:1])}
    ea = {Edge(e) for _, _, nd in p.after.shells for e in zip(nd, nd[1:] + nd[:1])}
    parts = []
    if set(p.before.coords) - set(p.after.coords):
        parts.append("절점삭제")
    if n.added:
        parts.append("절점추가")
    if n.moved:
        parts.append("좌표이동")
    if not parts:
        return "연결만 변경" if eb != ea else "변화없음"
    return " + ".join(parts) + ("" if eb != ea else " (연결 변화 없음)")


def defect_groups(pairs: dict) -> dict[str, int]:
    """{이름: 묶음 번호}. 같은 결함을 고친 쌍 / 거의 같은 패치는 같은 묶음.

    (a) 사람이 지운 요소(좌표 집합)를 하나라도 공유하거나
    (b) 원본 요소가 ``OVERLAP_FRAC`` 넘게 겹치면 묶는다 (union-find).
    실측 508쌍 → 259묶음, 가장 큰 묶음 10쌍.
    """
    from meshdiff import signature

    names = sorted(pairs)
    sig = {n: (signature(pairs[n].before), signature(pairs[n].after)) for n in names}
    par = {n: n for n in names}

    def find(x: str) -> str:
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    owner: dict = {}
    for n in names:
        for k in sig[n][0].shells - sig[n][1].shells:
            if k in owner:
                par[find(n)] = find(owner[k])
            else:
                owner[k] = n
    for i, a in enumerate(names):
        sa = sig[a][0].shells
        for b in names[i + 1:]:
            sb = sig[b][0].shells
            s = len(sa & sb)
            if s and s / min(len(sa), len(sb)) > OVERLAP_FRAC:
                par[find(a)] = find(b)
    roots = {r: i for i, r in enumerate(sorted({find(n) for n in names}))}
    return {n: roots[find(n)] for n in names}


def make_split(force: bool = False) -> pd.DataFrame:
    """분할 파일을 만든다. 이미 있으면 ``force`` 없이는 건드리지 않는다."""
    if os.path.exists(SPLIT_PATH) and not force:
        return load_split()
    from sklearn.model_selection import StratifiedGroupKFold

    from bdf_io import discover_pairs
    from pairs import all_pairs, discover_good_pairs, load_pair
    from regen_oracle import build_script

    good = discover_good_pairs()
    orig, _, _ = discover_pairs(gp.FE_DIR)
    rows, pairs = [], {}
    for nm in all_pairs():
        p = load_pair(nm, good, orig)
        pairs[nm] = p
        s, _, _ = build_script(p)
        rows.append(dict(name=nm, kind=kind_of(p), oracle=s.reason))
    d = pd.DataFrame(rows)
    grp = defect_groups(pairs)
    d["group"] = d.name.map(grp)
    strata = d.kind.where(d.kind.isin(["절점삭제", "연결만 변경"]), "기타")

    # 테스트 = 묶음 단위 10-fold 중 하나 (≈10%)
    outer = StratifiedGroupKFold(n_splits=round(1 / TEST_FRAC), shuffle=True, random_state=SPLIT_SEED)
    _, te = next(outer.split(d.index, strata, d.group))
    d["split"] = "train"
    d.loc[d.index[te], "split"] = "test"

    # 학습 셋 안의 CV fold — 이것도 묶음 단위로 미리 정해 둔다
    tr = d[d.split == "train"]
    inner = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    d["cv_fold"] = 0
    for k, (_, idx) in enumerate(inner.split(tr.index, strata[tr.index], tr.group), 1):
        d.loc[tr.index[idx], "cv_fold"] = k

    assert not (set(d[d.split == "test"].group) & set(d[d.split == "train"].group)), "묶음이 갈라졌다"
    gp.ensure_dirs()
    d[["name", "split", "cv_fold", "group", "kind", "oracle"]].to_csv(
        SPLIT_PATH, index=False, encoding="utf-8-sig")
    return d


def load_split() -> pd.DataFrame:
    if not os.path.exists(SPLIT_PATH):
        raise SystemExit(f"분할 파일이 없습니다: {SPLIT_PATH}\n먼저 `python src/regen_data.py` 를 돌리세요.")
    return pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")


def names_of(split: str) -> list[str]:
    d = load_split()
    return sorted(d[d.split == split].name) if split != "all" else sorted(d.name)


# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PairSample:
    """쌍 하나에서 학습·평가에 쓰는 것 전부."""

    name: str
    pair: object            # pairs.Pair (before / after / nodes)
    #: 정답 절점 병합 (remove, keep) — 원본 번호, 자리가 남는 쪽이 keep
    collapses: list
    #: 정답 병합을 적용한 뒤의 메쉬 (2단계 학습 입력)
    m1: object
    #: 2단계 정답 — 합칠 삼각형 쌍의 공유 변 (m1 의 절점 번호)
    merges: list
    oracle_exact: bool
    oracle_reason: str


def load_samples(names: list[str], verbose: bool = True) -> list[PairSample]:
    from bdf_io import discover_pairs
    from pairs import discover_good_pairs, load_pair
    from regen_oracle import build_script, collapse_groups

    good = discover_good_pairs()
    orig, _, _ = discover_pairs(gp.FE_DIR)
    out = []
    for i, nm in enumerate(names, 1):
        p = load_pair(nm, good, orig)
        s, m1, _ = build_script(p)
        # 정답 병합은 '의도한 것' 전부를 쓴다 (원본에서 이웃이 아니라 적용이 안 된 것 포함 —
        # 그런 쌍은 후보에 없으니 학습에는 자연히 안 들어간다)
        want = [(r, k) for k, rs in collapse_groups(p) for r in rs]
        out.append(PairSample(nm, p, want, m1, list(s.merges), s.exact, s.reason))
        if verbose and i % 100 == 0:
            print(f"  ... {i}/{len(names)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="재생성 90/10 분할")
    ap.add_argument("--force", action="store_true", help="분할을 다시 만든다 (테스트 셋이 바뀐다)")
    a = ap.parse_args()

    existed = os.path.exists(SPLIT_PATH)
    if existed and a.force:
        print("[주의] 분할을 다시 만듭니다 — 이전 테스트 셋으로 잰 숫자와 비교할 수 없게 됩니다.")
    d = make_split(force=a.force)
    print(("기존 분할: " if existed and not a.force else "새 분할: ") + SPLIT_PATH)
    print(pd.crosstab(d.kind, d.split, margins=True).to_string())
    print()
    print(pd.crosstab(d.oracle, d.split, margins=True).to_string())
    g = d.groupby("group").name.size()
    print(f"\n묶음 {len(g)}개 (가장 큰 묶음 {g.max()}쌍) · 테스트와 학습이 공유하는 묶음 "
          f"{len(set(d[d.split == 'test'].group) & set(d[d.split == 'train'].group))}개")
    print("학습 셋 CV fold 크기:", d[d.split == "train"].cv_fold.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

"""GAE 실험 데이터 — 정상 메쉬(학습)와 사람이 고친 쌍의 원본(채점).

    학습  정상 메쉬 1,932개 (확실·정상). 라벨은 '정상' 이라는 사실만 쓴다.
          공간 격자(6 m) 묶음 단위로 90/10 → 학습 / 검증(조기종료 + 메쉬 단위 채점용 정상)
    채점  수정 쌍 508개의 **원본**. 사람이 지운 요소 = 1, 나머지 = 0.
          GAE 는 이 쌍을 학습에 한 번도 보지 않는다.

그래프는 부모 프로젝트의 요소기반 그래프(노드 = 요소, 47차원)를 그대로 쓴다.
정상 메쉬는 부모 캐시에서 읽고, 쌍 원본은 여기서 새로 변환해 ``cache/`` 에 둔다
(부모 캐시는 '확실' 라벨만 담고 있어서 애매 118개가 빠져 있다).

    python src/ae_data.py          # 캐시 생성 + 요약
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import ae_paths as ap

PAIRS_CACHE = os.path.join(ap.CACHE_DIR, "pairs_elem.pkl")
NORMAL_SPLIT = os.path.join(ap.CACHE_DIR, "normal_split.csv")
VAL_FRAC = 0.10
SEED = 0


@dataclass
class PairGraph:
    """쌍 원본 하나 — 그래프 + 요소별 정답."""

    name: str
    ga: object                       # graph_base.GraphArrays
    #: 그래프 노드(요소)별 꼭짓점 절점 id. 쉘 다음에 1D 순서 — graph_elem 과 같다.
    elem_nodes: list = field(default_factory=list)
    #: 요소별 정답: 사람이 지운 쉘/1D = 1 (좌표 기준, meshdiff.diff)
    y: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    #: 절점 단위 정답 — node_baseline.py 와 같은 정의 (지운 쉘에 걸친 절점)
    touched_nodes: set = field(default_factory=set)
    split: str = ""                  # regen 분할: train / test
    kind: str = ""                   # 수정 종류 (연결만 변경 / 절점삭제 ...)


# ──────────────────────────────────────────────────────────────────────────
def normal_split(force: bool = False) -> pd.DataFrame:
    """정상 메쉬 목록과 학습/검증 분할. 공간 격자 묶음 단위로 나눈다.

    패치들은 한 선박 메쉬에서 잘라낸 조각이라 서로 겹친다. 무작위로 나누면 검증 정상이
    학습 정상과 거의 같은 패치가 되어 '정상다움' 이 부풀려진다.
    """
    if os.path.exists(NORMAL_SPLIT) and not force:
        return pd.read_csv(NORMAL_SPLIT, encoding="utf-8-sig")
    from data_index import list_samples, spatial_groups
    from pairs import all_pairs

    pair_names = set(all_pairs())
    s = [x for x in list_samples(trainable_only=True, verbose=False)
         if x.cls == 0 and x.name not in pair_names]
    groups = spatial_groups(s)
    ug = np.array(sorted(set(groups)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(ug)
    # 묶음을 섞어 앞에서부터 검증에 담되, 메쉬 수로 VAL_FRAC 을 채운다
    cnt = pd.Series(groups).value_counts()
    val_g, n = set(), 0
    for g in ug:
        if n >= VAL_FRAC * len(s):
            break
        val_g.add(g)
        n += int(cnt[g])
    df = pd.DataFrame({"name": [x.name for x in s], "group": groups,
                       "split": ["val" if g in val_g else "train" for g in groups]})
    ap.ensure_dirs()
    df.to_csv(NORMAL_SPLIT, index=False, encoding="utf-8-sig")
    return df


def load_normal_graphs(names: list[str]) -> list:
    """부모 캐시에서 정상 메쉬 그래프를 꺼낸다."""
    with open(ap.PARENT_ELEM_CACHE, "rb") as fh:
        blob = pickle.load(fh)
    store = blob["graphs"] if blob.get("version") == 2 else dict(zip(blob["names"], blob["graphs"]))
    missing = [n for n in names if n not in store]
    if missing:
        raise RuntimeError(f"부모 캐시에 없는 정상 메쉬 {len(missing)}개 (예: {missing[:3]}) — "
                           "부모 프로젝트에서 python src/dataset.py 로 캐시를 다시 만든다")
    return [store[n] for n in names]


# ──────────────────────────────────────────────────────────────────────────
def build_pairs(force: bool = False, verbose: bool = True) -> list[PairGraph]:
    if os.path.exists(PAIRS_CACHE) and not force:
        with open(PAIRS_CACHE, "rb") as fh:
            return pickle.load(fh)

    import graph_elem
    from bdf_io import discover_pairs
    from meshdiff import diff
    from meshrep import MeshRep
    from pairs import all_pairs, discover_good_pairs, load_pair

    split = pd.read_csv(ap.REGEN_SPLIT, encoding="utf-8-sig").set_index("name")
    good = discover_good_pairs()
    orig, _, _ = discover_pairs(ap.gp.FE_DIR)
    out: list[PairGraph] = []
    names = all_pairs()
    for i, nm in enumerate(names, 1):
        p = load_pair(nm, good, orig)
        rep = MeshRep(p.before)
        ga = graph_elem.build(rep, corner_edges=True, include_lines=True)
        d = diff(p.before, p.after)
        n_shell = len(rep.shells)
        y = np.zeros(ga.n_node, dtype=np.int8)
        y[d.removed_shells] = 1
        y[[n_shell + li for li in d.removed_lines]] = 1
        _, removed, _ = p.shell_diff()
        touched = set().union(*removed) if removed else set()
        elem_nodes = [tuple(s.nodes) for s in rep.shells] + [tuple(l.nodes) for l in rep.lines]
        row = split.loc[nm] if nm in split.index else None
        out.append(PairGraph(
            name=nm, ga=ga, elem_nodes=elem_nodes, y=y, touched_nodes=touched,
            split="" if row is None else str(row["split"]),
            kind="" if row is None else str(row["kind"]),
        ))
        if verbose and i % 100 == 0:
            print(f"  ... {i}/{len(names)}")
    ap.ensure_dirs()
    with open(PAIRS_CACHE, "wb") as fh:
        pickle.dump(out, fh)
    return out


def main() -> None:
    import argparse

    ar = argparse.ArgumentParser(description="GAE 실험 데이터 캐시")
    ar.add_argument("--force", action="store_true")
    a = ar.parse_args()

    df = normal_split(a.force)
    print(f"정상 메쉬 {len(df)}개 — 학습 {int((df.split == 'train').sum())} / "
          f"검증 {int((df.split == 'val').sum())}  (공간 묶음 {df.group.nunique()}개)")
    gs = load_normal_graphs(df.name.tolist())
    nn = np.array([g.n_node for g in gs])
    print(f"  요소 수 평균 {nn.mean():.1f} / 최대 {nn.max()}")

    # 모듈 이름으로 불러야 pickle 에 'ae_data.PairGraph' 로 저장된다 (__main__ 이 아니라)
    import ae_data

    pg = ae_data.build_pairs(a.force)
    ys = np.concatenate([p.y for p in pg])
    has = sum(1 for p in pg if p.y.any())
    print(f"\n수정 쌍 원본 {len(pg)}개 — 지운 요소가 있는 것 {has}개")
    print(f"  요소 {len(ys)}개 중 지운 요소 {int(ys.sum())}개 ({ys.mean():.1%})")
    print(f"  메쉬당 지운 요소 중앙값 {np.median([p.y.sum() for p in pg if p.y.any()]):.0f}")
    print(f"  regen 분할: " + ", ".join(f"{k} {v}" for k, v in
                                       pd.Series([p.split for p in pg]).value_counts().items()))


if __name__ == "__main__":
    main()

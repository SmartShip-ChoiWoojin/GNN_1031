"""재생성 학습/평가 데이터 — 고정된 90/10 분할과 쌍 하나의 학습 표본.

분할은 **한 번 만들어 파일로 남기고, 다시 계산하지 않는다** (`runs/regen_split.csv`).
부모 프로젝트에서 fold 를 화면에서 다시 계산했다가 학습대상 목록이 하나 바뀌자 대부분의
메쉬가 다른 fold 로 가서, '그 메쉬를 학습에 쓴 모델' 이 채점한 사고가 있었다.
테스트 10%는 학습·임계값 조정·모델 선택 어디에도 쓰지 않는다.

층화는 수정 종류로 한다 — '절점삭제' 52.6%, '연결만 변경' 38.6%, 나머지 8.9%. 나머지
7종은 각각 몇 건뿐이라 하나로 묶어 층화한다(안 묶으면 1건짜리 종류에서 분할이 안 된다).

**같은 결함은 한쪽에만** 둔다 (``defect_groups``). 패치들이 서로 겹쳐서, 같은 결함을 여러
패치에서 똑같이 고친 경우가 많다 — 508쌍 중 334쌍이 '지운 요소(좌표 기준)를 다른 쌍과
공유' 한다(한 결함이 최대 10개 패치에). 그래서 (a) 지운 요소를 공유하거나 (b) 원본 요소의
50% 넘게 겹치는 쌍을 한 묶음으로 보고, 묶음 단위로 나눈다. 학습 셋 안의 5-fold 도 같은
묶음 단위로 미리 정해 파일에 적는다.

    python src/regen_data.py            # 분할 파일이 없으면 만들고, 있으면 요약만
    python src/regen_data.py --force    # 다시 만든다 (테스트 셋이 바뀐다 — 신중히)
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

import gae_paths
from meshrep import Edge

#: 분할 파일 위치
SPLIT_PATH = os.path.join(gae_paths.RUNS_DIR, "regen_split.csv")
#: 테스트 비율
TEST_FRAC = 0.10
#: 분할 무작위 씨앗
SPLIT_SEED = 0
#: 학습 셋 안의 CV fold 수 (묶음 단위)
CV_FOLDS = 5
#: 원본 요소가 이 비율 넘게 겹치는 두 패치는 같은 묶음 (거의 같은 패치)
OVERLAP_FRAC = 0.5


def shell_edge_set(mesh) -> set:
    """메쉬의 쉘 변 집합."""
    edges = set()
    for element_id, element_type, node_ids in mesh.shells:
        corner_count = len(node_ids)
        for corner_index in range(corner_count):
            edges.add(Edge((node_ids[corner_index], node_ids[(corner_index + 1) % corner_count])))
    return edges


def kind_of(pair) -> str:
    """수정 종류 (app_pairs.classify 와 같은 규칙).

    절점삭제 / 절점추가 / 좌표이동 을 " + " 로 잇는다. 셋 다 없으면 변 연결이 바뀌었는지로
    "연결만 변경" 또는 "변화없음". 절점 변화가 있는데 연결이 그대로면 "(연결 변화 없음)" 을 붙인다.
    """
    node_match = pair.nodes
    edges_before = shell_edge_set(pair.before)
    edges_after = shell_edge_set(pair.after)
    edges_changed = edges_before != edges_after

    parts = []
    if len(set(pair.before.coords) - set(pair.after.coords)) > 0:
        parts.append("절점삭제")
    if len(node_match.added) > 0:
        parts.append("절점추가")
    if len(node_match.moved) > 0:
        parts.append("좌표이동")

    if len(parts) == 0:
        if edges_changed:
            return "연결만 변경"
        return "변화없음"

    kind = " + ".join(parts)
    if not edges_changed:
        kind = kind + " (연결 변화 없음)"
    return kind


def defect_groups(pairs: dict) -> dict:
    """{이름: 묶음 번호}. 같은 결함을 고친 쌍 / 거의 같은 패치는 같은 묶음 (union-find).

    (a) 사람이 지운 요소(좌표 집합)를 하나라도 공유하거나
    (b) 원본 요소가 ``OVERLAP_FRAC`` 넘게 겹치면 묶는다.
    실측 508쌍 → 259묶음, 가장 큰 묶음 10쌍.
    """
    from meshdiff import signature

    names = sorted(pairs)
    before_signature = {}
    after_signature = {}
    for name in names:
        before_signature[name] = signature(pairs[name].before)
        after_signature[name] = signature(pairs[name].after)

    # union-find 의 부모 표. 처음에는 자기 자신이 부모다.
    parent = {}
    for name in names:
        parent[name] = name

    def find_root(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    # (a) 지운 요소를 공유하는 쌍
    first_owner_of_removed = {}
    for name in names:
        removed_keys = before_signature[name].shells - after_signature[name].shells
        for removed_key in removed_keys:
            if removed_key in first_owner_of_removed:
                parent[find_root(name)] = find_root(first_owner_of_removed[removed_key])
            else:
                first_owner_of_removed[removed_key] = name

    # (b) 원본 요소가 많이 겹치는 쌍
    for index_a in range(len(names)):
        name_a = names[index_a]
        shells_a = before_signature[name_a].shells
        for name_b in names[index_a + 1:]:
            shells_b = before_signature[name_b].shells
            shared_count = len(shells_a & shells_b)
            if shared_count == 0:
                continue
            if shared_count / min(len(shells_a), len(shells_b)) > OVERLAP_FRAC:
                parent[find_root(name_a)] = find_root(name_b)

    root_set = set()
    for name in names:
        root_set.add(find_root(name))
    group_number_of_root = {}
    group_number = 0
    for root in sorted(root_set):
        group_number_of_root[root] = group_number
        group_number = group_number + 1

    group_of = {}
    for name in names:
        group_of[name] = group_number_of_root[find_root(name)]
    return group_of


def make_split(force: bool = False) -> pd.DataFrame:
    """분할 파일을 만든다. 이미 있으면 ``force`` 없이는 건드리지 않는다."""
    if os.path.exists(SPLIT_PATH) and not force:
        return load_split()

    from sklearn.model_selection import StratifiedGroupKFold

    from bdf_io import discover_pairs
    from pairs import all_pairs, discover_good_pairs, load_pair
    from regen_oracle import build_script

    good_paths = discover_good_pairs()
    original_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    rows = []
    pairs = {}
    for name in all_pairs():
        pair = load_pair(name, good_paths, original_paths)
        pairs[name] = pair
        script, _, _ = build_script(pair)
        rows.append(dict(name=name, kind=kind_of(pair), oracle=script.reason))
    table = pd.DataFrame(rows)

    group_of = defect_groups(pairs)
    table["group"] = table.name.map(group_of)
    # 층화 기준: 큰 두 종류는 그대로, 나머지는 "기타" 하나로
    strata = table.kind.where(table.kind.isin(["절점삭제", "연결만 변경"]), "기타")

    # 테스트 = 묶음 단위 10-fold 중 하나 (≈10%)
    outer_splitter = StratifiedGroupKFold(n_splits=round(1 / TEST_FRAC), shuffle=True, random_state=SPLIT_SEED)
    _, test_positions = next(outer_splitter.split(table.index, strata, table.group))
    table["split"] = "train"
    table.loc[table.index[test_positions], "split"] = "test"

    # 학습 셋 안의 CV fold — 이것도 묶음 단위로 미리 정해 둔다
    train_table = table[table.split == "train"]
    inner_splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    table["cv_fold"] = 0
    fold_number = 0
    for _, fold_positions in inner_splitter.split(train_table.index, strata[train_table.index], train_table.group):
        fold_number = fold_number + 1
        table.loc[train_table.index[fold_positions], "cv_fold"] = fold_number

    test_groups = set(table[table.split == "test"].group)
    train_groups = set(table[table.split == "train"].group)
    if len(test_groups & train_groups) > 0:
        raise AssertionError("묶음이 갈라졌다")

    gae_paths.ensure_dirs()
    table[["name", "split", "cv_fold", "group", "kind", "oracle"]].to_csv(
        SPLIT_PATH, index=False, encoding="utf-8-sig")
    return table


def load_split() -> pd.DataFrame:
    """분할 파일을 읽는다. 열: name, split(train/test), cv_fold, group, kind, oracle."""
    if not os.path.exists(SPLIT_PATH):
        raise SystemExit(f"분할 파일이 없습니다: {SPLIT_PATH}\n먼저 `python src/regen_data.py` 를 돌리세요.")
    return pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")


def names_of(split_name: str) -> list:
    """"train" / "test" / "all" 에 속한 쌍 이름 (정렬)."""
    table = load_split()
    if split_name == "all":
        return sorted(table.name)
    return sorted(table[table.split == split_name].name)


# ──────────────────────────────────────────────────────────────────────────
class PairSample:
    """쌍 하나에서 학습·평가에 쓰는 것 전부."""

    def __init__(self, name, pair, collapses, mesh_after_stage1, merges, oracle_exact, oracle_reason):
        self.name = name
        #: pairs.Pair (before / after / nodes)
        self.pair = pair
        #: 정답 절점 병합 [(remove, keep)] — 원본 번호, 자리가 남는 쪽이 keep
        self.collapses = collapses
        #: 정답 병합을 적용한 뒤의 메쉬 (2단계 학습 입력)
        self.mesh_after_stage1 = mesh_after_stage1
        #: 2단계 정답 — 합칠 삼각형 쌍의 공유 변 (mesh_after_stage1 의 절점 번호)
        self.merges = merges
        #: 정답 대본이 사람 결과를 정확히 재현하는가
        self.oracle_exact = oracle_exact
        #: 재현 결과 한 마디 (regen_oracle.Script.reason)
        self.oracle_reason = oracle_reason


def load_samples(names: list, verbose: bool = True) -> list:
    """쌍 이름 목록 → [PairSample]."""
    from bdf_io import discover_pairs
    from pairs import discover_good_pairs, load_pair
    from regen_oracle import build_script, collapse_groups, pairs_from_groups

    good_paths = discover_good_pairs()
    original_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    samples = []
    count = 0
    for name in names:
        count = count + 1
        pair = load_pair(name, good_paths, original_paths)
        script, mesh_after_stage1, _ = build_script(pair)
        # 정답 병합은 '의도한 것' 전부를 쓴다 (원본에서 이웃이 아니라 적용이 안 된 것 포함 —
        # 그런 쌍은 후보에 없으니 학습에는 자연히 안 들어간다)
        wanted_collapses = pairs_from_groups(collapse_groups(pair))
        samples.append(PairSample(name, pair, wanted_collapses, mesh_after_stage1, list(script.merges),
                                  script.exact, script.reason))
        if verbose and count % 100 == 0:
            print(f"  ... {count}/{len(names)}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="재생성 90/10 분할")
    parser.add_argument("--force", action="store_true", help="분할을 다시 만든다 (테스트 셋이 바뀐다)")
    args = parser.parse_args()

    existed = os.path.exists(SPLIT_PATH)
    if existed and args.force:
        print("[주의] 분할을 다시 만듭니다 — 이전 테스트 셋으로 잰 숫자와 비교할 수 없게 됩니다.")
    table = make_split(force=args.force)
    if existed and not args.force:
        print("기존 분할: " + SPLIT_PATH)
    else:
        print("새 분할: " + SPLIT_PATH)
    print(pd.crosstab(table.kind, table.split, margins=True).to_string())
    print()
    print(pd.crosstab(table.oracle, table.split, margins=True).to_string())

    group_sizes = table.groupby("group").name.size()
    test_groups = set(table[table.split == "test"].group)
    train_groups = set(table[table.split == "train"].group)
    print(f"\n묶음 {len(group_sizes)}개 (가장 큰 묶음 {group_sizes.max()}쌍) · 테스트와 학습이 공유하는 묶음 "
          f"{len(test_groups & train_groups)}개")
    fold_sizes = table[table.split == "train"].cv_fold.value_counts().sort_index().to_dict()
    print("학습 셋 CV fold 크기:", fold_sizes)


if __name__ == "__main__":
    main()

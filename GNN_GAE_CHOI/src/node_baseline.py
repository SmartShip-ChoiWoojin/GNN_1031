"""절점 단위 표(tabular) 기준선 — "고친 자리" 가 국소 형상만으로 얼마나 설명되나.

GNN 을 만들기 전에 던진 질문: **절점 주변의 생김새만 보고 사람이 거기를 고칠지 맞힐 수
있는가?** 맞힐 수 있다면 그 피처가 곧 "왜 고쳤나" 의 후보다.

라벨: 사람이 없앤 쉘에 걸쳐 있던 절점 = 1, 나머지 = 0.
평가는 **메쉬 단위**로 한다 — "이 메쉬에서 어디를 고칠까" 가 실제 질문이라서.

    python src/node_baseline.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import gae_paths
import features as F
from bdf_io import discover_pairs
from meshrep import MeshRep
from pairs import all_pairs, discover_good_pairs, load_pair

FEATURES = [
    "deg_shell", "deg_line", "n_tri_adj", "tri_frac_adj",
    "ang_min_adj", "ang_max_adj", "skew_max_adj", "aspect_max_adj",
    "n_bad_adj", "is_ext", "angle_sum_dev", "n_free_edge", "n_nonman_edge",
    "dihedral_max", "area_ratio_adj", "len_rel_min", "len_rel_max",
]


def node_rows(rep: MeshRep):
    """메쉬 하나의 절점별 피처. (절점 번호 목록, (절점 수, 피처 수) 배열)."""
    median_edge = rep.median_edge_length
    node_ids = []
    rows = []
    for node_id in rep.node_ids:
        shell_indices = rep.node_shells.get(node_id, [])
        qualities = []
        for shell_index in shell_indices:
            qualities.append(rep.shells[shell_index].quality)
        edges = rep.node_edges(node_id)
        lengths = []
        for edge in edges:
            lengths.append(rep.edge_length(edge))
        if len(lengths) == 0:
            lengths = [median_edge]
        areas = []
        for quality in qualities:
            areas.append(quality.area)
        if len(areas) == 0:
            areas = [1.0]

        triangle_count = 0
        bad_count = 0
        for quality in qualities:
            if not quality.is_quad:
                triangle_count = triangle_count + 1
            if F.is_bad_shell(quality):
                bad_count = bad_count + 1
        if len(qualities) > 0:
            triangle_fraction = triangle_count / len(qualities)
        else:
            triangle_fraction = 0.0

        # 주변 요소의 최소/최대 값 (주변 요소가 없으면 정해 둔 기본값)
        smallest_angle = 90.0
        largest_angle = 90.0
        largest_skew = 0.0
        largest_aspect = 1.0
        if len(qualities) > 0:
            smallest_angle = qualities[0].angle_min
            largest_angle = qualities[0].angle_max
            largest_skew = qualities[0].skew
            largest_aspect = qualities[0].aspect
            for quality in qualities:
                if quality.angle_min < smallest_angle:
                    smallest_angle = quality.angle_min
                if quality.angle_max > largest_angle:
                    largest_angle = quality.angle_max
                if quality.skew > largest_skew:
                    largest_skew = quality.skew
                if quality.aspect > largest_aspect:
                    largest_aspect = quality.aspect

        free_edge_count = 0
        nonmanifold_edge_count = 0
        for edge in edges:
            if rep.is_free_edge(edge):
                free_edge_count = free_edge_count + 1
            if rep.is_nonmanifold_edge(edge):
                nonmanifold_edge_count = nonmanifold_edge_count + 1

        node_ids.append(node_id)
        rows.append([
            len(shell_indices),
            len(rep.node_lines.get(node_id, [])),
            triangle_count,
            triangle_fraction,
            smallest_angle,
            largest_angle,
            largest_skew,
            largest_aspect,
            bad_count,
            float(rep.ext.get(node_id, False)),
            rep.node_angle_sum(node_id) - 360.0,
            free_edge_count,
            nonmanifold_edge_count,
            rep.node_max_dihedral(node_id),
            max(areas) / (min(areas) + 1e-9),
            min(lengths) / (median_edge + 1e-9),
            max(lengths) / (median_edge + 1e-9),
        ])
    return node_ids, np.array(rows, dtype=float)


def build(limit: int | None = None) -> pd.DataFrame:
    """절점 표. 열: name, node, FEATURES..., y."""
    good_paths = discover_good_pairs()
    original_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    names = all_pairs()
    if limit:
        names = names[:limit]
    tables = []
    count = 0
    for name in names:
        count = count + 1
        pair = load_pair(name, good_paths, original_paths)
        _, removed, _ = pair.shell_diff()
        if len(removed) == 0:
            # 변화 없는 4건
            continue
        touched_nodes = set()
        for removed_shell in removed:
            touched_nodes = touched_nodes | set(removed_shell)
        rep = MeshRep(pair.before)
        node_ids, feature_matrix = node_rows(rep)
        table = pd.DataFrame(feature_matrix, columns=FEATURES)
        table.insert(0, "node", node_ids)
        table.insert(0, "name", name)
        labels = []
        for node_id in node_ids:
            labels.append(int(node_id in touched_nodes))
        table["y"] = labels
        tables.append(table)
        if count % 100 == 0:
            print(f"  ... {count}/{len(names)}")
    return pd.concat(tables, ignore_index=True)


def top_k_by_mesh(table: pd.DataFrame, scores: np.ndarray, k_values=(1, 3, 5)) -> dict:
    """메쉬마다 점수 상위 k개 안에 '고친 자리' 절점이 들어 있는 비율."""
    scored = table.copy()
    scored["s"] = scores
    hits = {}
    for k in k_values:
        hits[k] = []
    for _, mesh_table in scored.groupby("name"):
        mesh_table = mesh_table.sort_values("s", ascending=False)
        for k in k_values:
            hits[k].append(int(mesh_table.head(k)["y"].sum() > 0))
    rates = {}
    for k, values in hits.items():
        rates[k] = float(np.mean(values))
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description="절점 단위 tabular 기준선")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("절점 피처 추출 중...")
    table = build(args.limit)
    meshes = sorted(table["name"].unique())
    random_generator = np.random.default_rng(args.seed)
    random_generator.shuffle(meshes)
    test_count = int(len(meshes) * args.test_frac)
    test_meshes = set(meshes[:test_count])
    train_table = table[~table["name"].isin(test_meshes)]
    test_table = table[table["name"].isin(test_meshes)]
    print(f"\n메쉬 {len(meshes)}개 (학습 {len(meshes) - test_count} / 채점 {test_count})")
    print(f"절점 {len(table)}개 중 '고친 자리' {int(table.y.sum())}개 ({table.y.mean():.1%})")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    classifier = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=args.seed)
    classifier.fit(train_table[FEATURES], train_table["y"])
    scores = classifier.predict_proba(test_table[FEATURES])[:, 1]

    print(f"\n── 채점 ({test_count}개 메쉬) ──")
    print(f"  절점 단위 AUC {roc_auc_score(test_table['y'], scores):.4f}")
    for k, rate in top_k_by_mesh(test_table, scores).items():
        print(f"  상위 {k}개 안에 고친 자리 포함: {rate:.1%}")
    random_rates = top_k_by_mesh(test_table, random_generator.random(len(test_table)))
    random_parts = []
    for k, rate in random_rates.items():
        random_parts.append(f"{k}위 {rate:.1%}")
    print("  (무작위 기준: " + " / ".join(random_parts) + ")")

    print("\n── 어떤 피처가 쓰였나 (permutation importance) ──")
    from sklearn.inspection import permutation_importance

    importance = permutation_importance(classifier, test_table[FEATURES], test_table["y"], n_repeats=5,
                                        random_state=args.seed, scoring="roc_auc")
    order = np.argsort(-importance.importances_mean)
    for feature_index in order[:10]:
        print(f"  {FEATURES[feature_index]:16s} {importance.importances_mean[feature_index]:+.4f}")

    print("\n── 고친 자리 절점 vs 나머지 (중앙값) ──")
    for column in ("n_tri_adj", "n_bad_adj", "ang_min_adj", "skew_max_adj", "deg_shell", "angle_sum_dev",
                   "aspect_max_adj"):
        touched_median = table[table.y == 1][column].median()
        other_median = table[table.y == 0][column].median()
        print(f"  {column:16s} {touched_median:8.2f}  vs {other_median:8.2f}")

    gae_paths.ensure_dirs()
    out_path = os.path.join(gae_paths.OUT_DIR, "node_table.csv")
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()

"""절점 단위 tabular 베이스라인 — "고친 자리"가 국소 형상만으로 얼마나 설명되나.

GNN 을 만들기 전에 던져야 할 질문: **절점 주변의 생김새만 보고 사람이 거기를 고칠지
맞힐 수 있는가?** 맞힐 수 있다면 그 피처가 곧 "왜 고쳤나"의 후보다. 못 맞힌다면 이유가
더 넓은 문맥에 있다는 뜻이고, 그때 비로소 GNN 이 필요하다.

라벨: 사람이 없앤 쉘에 걸쳐 있던 절점 = 1, 나머지 = 0.
중심 절점 하나를 임의로 고르지 않는 이유는 동점이 절반이라 그 선택이 잡음이 되기 때문.

평가는 **메쉬 단위**로 한다. 절점 단위 AUC 는 메쉬 크기에 휘둘려서, 실제로 쓸 때의
질문("이 메쉬에서 어디를 고칠까")과 맞지 않는다.

    python src/node_baseline.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import gae_paths as gp
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


def node_rows(rep: MeshRep) -> tuple[list[int], np.ndarray]:
    """메쉬 하나의 절점별 피처. (절점 id 목록, (n, d) 배열)"""
    med = rep.median_edge_length
    ids, rows = [], []
    for nid in rep.node_ids:
        sis = rep.node_shells.get(nid, [])
        Q = [rep.shells[s].quality for s in sis]
        edges = rep.node_edges(nid)
        lens = [rep.edge_length(e) for e in edges] or [med]
        areas = [q.area for q in Q] or [1.0]
        ids.append(nid)
        rows.append([
            len(sis),
            len(rep.node_lines.get(nid, [])),
            sum(1 for q in Q if not q.is_quad),
            (sum(1 for q in Q if not q.is_quad) / len(Q)) if Q else 0.0,
            min((q.angle_min for q in Q), default=90.0),
            max((q.angle_max for q in Q), default=90.0),
            max((q.skew for q in Q), default=0.0),
            max((q.aspect for q in Q), default=1.0),
            sum(1 for q in Q if F.is_bad_shell(q)),
            float(rep.ext.get(nid, False)),
            rep.node_angle_sum(nid) - 360.0,
            sum(1 for e in edges if rep.is_free_edge(e)),
            sum(1 for e in edges if rep.is_nonmanifold_edge(e)),
            rep.node_max_dihedral(nid),
            max(areas) / (min(areas) + 1e-9),
            min(lens) / (med + 1e-9),
            max(lens) / (med + 1e-9),
        ])
    return ids, np.array(rows, dtype=float)


def build(limit: int | None = None) -> pd.DataFrame:
    good = discover_good_pairs()
    orig, _, _ = discover_pairs(gp.FE_DIR)
    names = all_pairs()[:limit] if limit else all_pairs()
    out = []
    for i, nm in enumerate(names, 1):
        p = load_pair(nm, good, orig)
        _, removed, _ = p.shell_diff()
        if not removed:
            continue  # 변화 없는 4건
        touched = set().union(*removed)
        rep = MeshRep(p.before)
        ids, X = node_rows(rep)
        df = pd.DataFrame(X, columns=FEATURES)
        df.insert(0, "node", ids)
        df.insert(0, "name", nm)
        df["y"] = [int(n in touched) for n in ids]
        out.append(df)
        if i % 100 == 0:
            print(f"  ... {i}/{len(names)}")
    return pd.concat(out, ignore_index=True)


def topk_by_mesh(df: pd.DataFrame, score: np.ndarray, ks=(1, 3, 5)) -> dict:
    """메쉬마다 점수 상위 k개 안에 '고친 자리' 절점이 들어 있나."""
    d = df.copy()
    d["s"] = score
    res = {k: [] for k in ks}
    for _, g in d.groupby("name"):
        g = g.sort_values("s", ascending=False)
        for k in ks:
            res[k].append(int(g.head(k)["y"].sum() > 0))
    return {k: float(np.mean(v)) for k, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="절점 단위 tabular 베이스라인")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("절점 피처 추출 중...")
    df = build(a.limit)
    meshes = sorted(df["name"].unique())
    rng = np.random.default_rng(a.seed)
    rng.shuffle(meshes)
    n_test = int(len(meshes) * a.test_frac)
    test = set(meshes[:n_test])
    tr, te = df[~df["name"].isin(test)], df[df["name"].isin(test)]
    print(f"\n메쉬 {len(meshes)}개 (학습 {len(meshes)-n_test} / 채점 {n_test})")
    print(f"절점 {len(df)}개 중 '고친 자리' {int(df.y.sum())}개 ({df.y.mean():.1%})")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                         random_state=a.seed)
    clf.fit(tr[FEATURES], tr["y"])
    s = clf.predict_proba(te[FEATURES])[:, 1]

    print(f"\n── 채점 ({n_test}개 메쉬) ──")
    print(f"  절점 단위 AUC {roc_auc_score(te['y'], s):.4f}")
    hit = topk_by_mesh(te, s)
    for k, v in hit.items():
        print(f"  상위 {k}개 안에 고친 자리 포함: {v:.1%}")
    rnd = topk_by_mesh(te, rng.random(len(te)))
    print(f"  (무작위 기준: " + " / ".join(f"{k}위 {v:.1%}" for k, v in rnd.items()) + ")")

    print("\n── 어떤 피처가 쓰였나 (permutation importance) ──")
    from sklearn.inspection import permutation_importance

    r = permutation_importance(clf, te[FEATURES], te["y"], n_repeats=5,
                               random_state=a.seed, scoring="roc_auc")
    order = np.argsort(-r.importances_mean)
    for i in order[:10]:
        print(f"  {FEATURES[i]:16s} {r.importances_mean[i]:+.4f}")

    print("\n── 고친 자리 절점 vs 나머지 (중앙값) ──")
    for c in ("n_tri_adj", "n_bad_adj", "ang_min_adj", "skew_max_adj",
              "deg_shell", "angle_sum_dev", "aspect_max_adj"):
        a1 = df[df.y == 1][c].median()
        a0 = df[df.y == 0][c].median()
        print(f"  {c:16s} {a1:8.2f}  vs {a0:8.2f}")

    gp.ensure_dirs()
    import os
    df.to_csv(os.path.join(gp.OUT_DIR, "node_table.csv"), index=False, encoding="utf-8-sig")
    print(f"\n저장: {os.path.join(gp.OUT_DIR, 'node_table.csv')}")


if __name__ == "__main__":
    main()

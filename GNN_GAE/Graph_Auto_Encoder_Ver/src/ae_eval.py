"""GAE 이상 점수가 '사람이 고친 자리' 를 가리키는지 채점한다.

질문은 하나다: **정상 메쉬만 보고 배운 모델이, 사람이 지운 요소를 높게 찍는가?**
수정 쌍은 GAE 학습에 한 번도 쓰지 않았으므로 508쌍 전부가 채점 표본이다.

비교 대상 (학습 없이 바로 나오는 것들 — GAE 가 이것보다 나아야 의미가 있다)
    random      무작위
    tri         삼각형이면 1 (사람 수정의 대부분이 삼각형 주변이다)
    bad_angle   최소내각 < 30° 또는 최대내각 > 150°
    min_angle   최소내각이 작을수록 높게
    maha        정상 요소 피처 분포로부터의 마할라노비스 거리 (그래프 없음, 신경망 없음)
그리고 절점 단위로는 지도학습 기준선 node_baseline.py (GBM, 수정 쌍 403개로 학습) 와
같은 채점 메쉬 100개에서 비교한다.

지표 (메쉬마다 계산해 평균 — "이 메쉬에서 어디를 고칠까" 라는 실제 질문에 맞춘다)
    top-k   점수 상위 k 요소 안에 사람이 지운 요소가 하나라도 있나
    AUC/AP  메쉬 안에서 지운 요소 vs 나머지
동점은 무작위로 깬다(20번 평균) — 0/1 규칙이 순서 운으로 이득을 보지 않게.

메쉬 단위로는 "이 메쉬가 고쳐야 하는 메쉬인가" 도 본다: 검증 정상 205개 vs 쌍 원본 508개.

    python src/ae_eval.py --runs mae ae ae_nograph
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

import ae_paths as ap
from ae_data import build_pairs, load_normal_graphs, normal_split
from ae_train import load_run, to_data
from graph_elem import NODE_SPEC

KS = (1, 3, 5)
N_TIE = 20


# ──────────────────────────────────────────────────────────────────────────
# 점수 만들기
# ──────────────────────────────────────────────────────────────────────────
def _maha_fit(X: np.ndarray, ridge: float = 1e-2):
    mu = X.mean(0)
    sd = np.maximum(X.std(0), 1e-2)  # 거의 상수인 칸이 거리를 폭발시키지 않게
    Z = (X - mu) / sd
    P = np.linalg.inv(np.cov(Z, rowvar=False) + ridge * np.eye(X.shape[1]))
    return lambda A: np.einsum("ij,jk,ik->i", (A - mu) / sd, P, (A - mu) / sd)


def gae_scores(run: str, pairs, normals, train_normals, device) -> dict[str, tuple[list, list]]:
    """{점수이름: (쌍별 요소 점수 리스트, 정상별 요소 점수 리스트)}

    feat   잔차 제곱 평균 (그대로)
    rz     잔차 벡터의 마할라노비스 거리 — 학습 정상 메쉬의 잔차 분포로 적합한다.
           "정상일 때도 원래 잘 틀리는 피처" 의 오차를 깎아 준다.
    xr     [피처, 잔차] 를 이어 붙인 마할라노비스 — "요소 자체가 드문가" + "주변과 안 맞나"
    """
    model, norm, _ = load_run(run, device)

    def score_all(gas):
        ds = [to_data(g, norm) for g in gas]
        out: dict[str, list] = {"x": []}
        for d in DataLoader(ds, batch_size=64):
            d = d.to(device)
            sc = model.node_scores(d)
            cnt = torch.bincount(d.batch).tolist()
            out["x"].extend(torch.split(d.x.cpu(), cnt))
            for k, v in sc.items():
                out.setdefault(k, []).extend(torch.split(v.cpu(), cnt))
        return {k: [t.numpy() for t in v] for k, v in out.items()}

    sp, sn = score_all([p.ga for p in pairs]), score_all(normals)
    st = score_all(train_normals)
    res = {k: (sp[k], sn[k]) for k in sp if k not in ("x", "resid")}
    f_rz = _maha_fit(np.concatenate(st["resid"]))
    res["rz"] = ([f_rz(r) for r in sp["resid"]], [f_rz(r) for r in sn["resid"]])
    cat = lambda s: [np.hstack([x, r]) for x, r in zip(s["x"], s["resid"])]
    f_xr = _maha_fit(np.concatenate(cat(st)))
    res["xr"] = ([f_xr(a) for a in cat(sp)], [f_xr(a) for a in cat(sn)])
    if "struct" in sp:
        # 두 오차를 검증 정상 요소의 분포로 표준화해서 더한다 (DOMINANT 의 가중합 대신)
        def z(k):
            allv = np.concatenate(sn[k])
            mu, sd = allv.mean(), allv.std() + 1e-9
            return [(v - mu) / sd for v in sp[k]], [(v - mu) / sd for v in sn[k]]
        (fp, fn), (stp, stn) = z("feat"), z("struct")
        res["sum"] = ([a + b for a, b in zip(fp, stp)], [a + b for a, b in zip(fn, stn)])
    return res


def heuristic_scores(pairs, normals, train_normals) -> dict[str, tuple[list, list]]:
    ti, ba, am = NODE_SPEC.index("is_tria3"), NODE_SPEC.index("s_bad_angle"), \
        NODE_SPEC.index("s_angle_min")
    shell = lambda g: g.node_kind == 0

    def f(fn):
        return [fn(p.ga) for p in pairs], [fn(g) for g in normals]

    out = {
        "random": ([np.zeros(p.ga.n_node) for p in pairs], [np.zeros(g.n_node) for g in normals]),
        "tri": f(lambda g: g.x[:, ti].astype(float)),
        "bad_angle": f(lambda g: g.x[:, ba].astype(float)),
        "min_angle": f(lambda g: np.where(shell(g), -g.x[:, am], -1e3)),
    }
    # 마할라노비스 — 학습 정상 요소로 적합 (정규화도 학습 정상으로)
    from dataset import NormalizerSet, _valid_of

    ns = NormalizerSet("elem").fit(train_normals, np.arange(len(train_normals)))
    X = np.concatenate([ns.node.transform(g.x, _valid_of(g, "x")) for g in train_normals])
    mu = X.mean(0)
    cov = np.cov(X - mu, rowvar=False) + 1e-2 * np.eye(X.shape[1])
    P = np.linalg.inv(cov)

    def maha(g):
        Z = ns.node.transform(g.x, _valid_of(g, "x")) - mu
        return np.einsum("ij,jk,ik->i", Z, P, Z)

    out["maha"] = f(maha)
    return out


# ──────────────────────────────────────────────────────────────────────────
# 지표
# ──────────────────────────────────────────────────────────────────────────
def topk_hit(score: np.ndarray, y: np.ndarray, rng) -> dict[int, float]:
    """동점을 무작위로 깨서 N_TIE 번 평균한 top-k 적중."""
    hits = {k: 0.0 for k in KS}
    for _ in range(N_TIE):
        order = np.lexsort((rng.random(len(score)), -score))
        for k in KS:
            hits[k] += float(y[order[:k]].any())
    return {k: v / N_TIE for k, v in hits.items()}


def elem_metrics(scores: list, pairs, idx: list[int], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for i in idx:
        y, s = pairs[i].y, scores[i]
        r = {f"top{k}": v for k, v in topk_hit(s, y, rng).items()}
        # 1순위가 지운 요소이거나 그 바로 옆(엣지 1개)인가 — "동네는 맞혔나"
        ei = pairs[i].ga.edge_index
        near = y.astype(bool).copy()
        near[ei[1][y[ei[0]] > 0]] = True
        r["near1"] = topk_hit(s, near, rng)[1]
        if 0 < y.sum() < len(y):
            r["auc"] = roc_auc_score(y, s)
            r["ap"] = average_precision_score(y, s)
        rows.append(r)
    return pd.DataFrame(rows).mean().to_dict()


def node_level(scores: list, pairs) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """요소 점수 → 절점 점수 (붙은 요소 중 최댓값). {이름: (절점 id, 점수)}"""
    out = {}
    for p, s in zip(pairs, scores):
        best: dict[int, float] = {}
        for e, nodes in enumerate(p.elem_nodes):
            for n in nodes:
                if s[e] > best.get(n, -np.inf):
                    best[n] = float(s[e])
        out[p.name] = best
    return out


def node_topk(node_scores: dict, table: pd.DataFrame, names: list[str], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in KS}
    for nm in names:
        g = table[table.name == nm]
        s = np.array([node_scores[nm].get(int(n), -1e9) for n in g.node])
        for k, v in topk_hit(s, g.y.to_numpy().astype(bool), rng).items():
            acc[k].append(v)
    return {f"node_top{k}": float(np.mean(v)) for k, v in acc.items()}


def gbm_node_baseline(table: pd.DataFrame) -> tuple[list[str], dict]:
    """node_baseline.py 를 같은 분할로 다시 돌려 채점 메쉬 100개의 절점 점수를 얻는다."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    from node_baseline import FEATURES

    meshes = sorted(table.name.unique())
    rng = np.random.default_rng(0)
    rng.shuffle(meshes)
    test = list(meshes[:int(len(meshes) * 0.2)])
    tr, te = table[~table.name.isin(test)], table[table.name.isin(test)]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0)
    clf.fit(tr[FEATURES], tr["y"])
    te = te.assign(s=clf.predict_proba(te[FEATURES])[:, 1])
    sc = {nm: dict(zip(g.node.astype(int), g.s)) for nm, g in te.groupby("name")}
    return test, sc


def mesh_auc(sp: list, sn: list) -> dict:
    """메쉬 점수(상위 3 요소 평균 / 최댓값)로 정상(검증) vs 쌍 원본을 가른다."""
    def top3(v):
        return float(np.sort(v)[-3:].mean())
    y = np.r_[np.zeros(len(sn)), np.ones(len(sp))]
    return {"mesh_auc_top3": roc_auc_score(y, [top3(v) for v in sn] + [top3(v) for v in sp]),
            "mesh_auc_max": roc_auc_score(y, [float(v.max()) for v in sn] + [float(v.max()) for v in sp])}


# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    ar = argparse.ArgumentParser(description="GAE 이상 점수 채점")
    ar.add_argument("--runs", nargs="*", default=["mae", "ae", "ae_nograph"])
    a = ar.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = build_pairs(verbose=False)
    sp = normal_split()
    normals = load_normal_graphs(sp[sp.split == "val"].name.tolist())
    train_normals = load_normal_graphs(sp[sp.split == "train"].name.tolist())

    scorers = heuristic_scores(pairs, normals, train_normals)
    ap.ensure_dirs()
    with open(os.path.join(ap.RUNS_DIR, "baseline_scores.pkl"), "wb") as fh:
        pickle.dump({k: {p.name: s for p, s in zip(pairs, v[0])} for k, v in scorers.items()}, fh)
    for r in a.runs:
        for k, v in gae_scores(os.path.join(ap.RUNS_DIR, r), pairs, normals, train_normals,
                               dev).items():
            scorers[f"{r}:{k}"] = v
            with open(os.path.join(ap.RUNS_DIR, r, f"pair_scores_{k}.pkl"), "wb") as fh:
                pickle.dump({p.name: s for p, s in zip(pairs, v[0])}, fh)

    has = [i for i, p in enumerate(pairs) if p.y.any()]
    test = [i for i in has if pairs[i].split == "test"]
    table = pd.read_csv(ap.NODE_TABLE, encoding="utf-8-sig")
    nb_test, gbm = gbm_node_baseline(table)
    nb_idx = [i for i in has if pairs[i].name in set(nb_test)]

    rows = []
    for name, (s_pair, s_norm) in scorers.items():
        r = {"scorer": name}
        r.update({f"all_{k}": v for k, v in elem_metrics(s_pair, pairs, has).items()})
        r.update({f"test_{k}": v for k, v in elem_metrics(s_pair, pairs, test).items()})
        ns = node_level(s_pair, pairs)
        r.update(node_topk(ns, table, nb_test))
        if name != "random":
            r.update(mesh_auc(s_pair, s_norm))
        rows.append(r)
    r = {"scorer": "GBM 지도학습 (node_baseline)"}
    r.update(node_topk(gbm, table, nb_test))
    rows.append(r)
    df = pd.DataFrame(rows)
    out = os.path.join(ap.RUNS_DIR, "eval_summary.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"채점 메쉬: 전체 {len(has)} / regen 테스트 {len(test)} / node_baseline 채점 {len(nb_idx)}")
    print("\n── 요소 단위 (전체 504 메쉬) ──")
    print(df[["scorer", "all_top1", "all_near1", "all_top3", "all_top5", "all_auc", "all_ap"]]
          .dropna(subset=["all_top1"]).round(3).to_string(index=False))
    print("\n── 요소 단위 (regen 테스트 셋) ──")
    print(df[["scorer", "test_top1", "test_top3", "test_auc", "test_ap"]]
          .dropna(subset=["test_top1"]).round(3).to_string(index=False))
    print("\n── 절점 단위 (node_baseline 채점 메쉬 100개 — GBM 은 지도학습) ──")
    print(df[["scorer", "node_top1", "node_top3", "node_top5"]].round(3).to_string(index=False))
    print("\n── 메쉬 단위: 검증 정상 205 vs 쌍 원본 508 (AUC) ──")
    print(df[["scorer", "mesh_auc_top3", "mesh_auc_max"]].dropna().round(3).to_string(index=False))

    # 수정 종류별 — 가장 나은 GAE 점수와 규칙 기준선
    kinds = pd.Series([pairs[i].kind for i in has])
    print("\n── 수정 종류별 top-3 (전체) ──")
    pick = [c for c in scorers if c in ("tri", "min_angle", "maha") or ":" in c]
    rows = []
    for kd, cnt in kinds.value_counts().items():
        if cnt < 15:
            continue
        idx = [i for i in has if pairs[i].kind == kd]
        rows.append({"kind": kd, "n": cnt,
                     **{c: elem_metrics(scorers[c][0], pairs, idx)["top3"] for c in pick}})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()

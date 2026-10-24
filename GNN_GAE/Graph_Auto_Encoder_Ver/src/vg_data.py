"""학습 데이터 — Bad 메쉬의 절점 그래프와, Good 메쉬에서 뽑은 정답.

쌍 하나에서 만드는 것:

  입력 (Bad 메쉬)
    노드      절점. 피처 = 주변 요소 품질 17개 + 메쉬 평면 기준 위치 3개
    엣지      쉘 요소의 변. 피처 = 길이, 공유 요소 수, 1D 여부, 양끝 ext 여부, 붙은 삼각형 수

  정답 (Good 메쉬를 Bad 절점 번호로 옮긴 것)
    keep      Bad 절점이 Good 에도 남았나 (좌표로 비교)
    adj_good  Good 메쉬의 변을 Bad 절점 쌍으로 옮긴 것

절점 대응은 번호가 아니라 **좌표**로 한다 (meshdiff 와 같은 규칙). 사람은 절점을 합칠 때
남길 번호를 일정하게 고르지 않아서, 번호로 비교하면 같은 결과를 다르다고 본다.
Good 에만 있는 자리(새로 만든 절점, 옮긴 절점)는 Bad 절점으로 옮길 수 없어서 정답에서 빠진다.

    python src/vg_data.py          # cache/pairs_node.pkl 을 만든다 (508쌍, 몇 분)
"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

import vg_paths as vp
from meshdiff import pkey

EDGE_FEATS = ["len_rel", "n_shell", "has_line", "both_ext", "n_tri"]


def sides(shells) -> set[tuple[int, int]]:
    """쉘 요소들의 변 집합 (작은 번호, 큰 번호)."""
    out = set()
    for s in shells:
        ns = s[2]
        for a, b in zip(ns, list(ns[1:]) + [ns[0]]):
            out.add((a, b) if a < b else (b, a))
    return out


def local_xyz(P: np.ndarray, scale: float) -> np.ndarray:
    """메쉬 평면(PCA) 기준 좌표를 중앙 변 길이로 나눈 것. 부호는 3차 모멘트로 고정한다."""
    c = P - P.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    q = c @ vt.T
    sgn = np.sign((q ** 3).sum(0))
    sgn[sgn == 0] = 1.0
    return q * sgn / (scale + 1e-9)


def graph_of(bad) -> dict:
    """Bad 메쉬 하나 → 입력 그래프만 (정답 없음). 테스트 메쉬처럼 Good 이 없을 때 쓴다."""
    from meshrep import MeshRep
    from node_baseline import node_rows

    rep = MeshRep(bad)
    ids, Xq = node_rows(rep)
    idx = {n: i for i, n in enumerate(ids)}
    P = np.array([bad.coords[n] for n in ids], dtype=float)
    X = np.hstack([Xq, local_xyz(P, rep.median_edge_length)]).astype(np.float32)

    # 입력 엣지 = Bad 의 변
    tri_of = {}
    for s in bad.shells:
        ns = s[2]
        for a, b in zip(ns, list(ns[1:]) + [ns[0]]):
            e = (a, b) if a < b else (b, a)
            tri_of.setdefault(e, []).append(len(ns) == 3)
    line_e = {tuple(sorted(l[2])) for l in bad.lines}
    E, EA = [], []
    med = rep.median_edge_length
    for (a, b), tris in sorted(tri_of.items()):
        L = float(np.linalg.norm(P[idx[a]] - P[idx[b]]))
        EA.append([L / (med + 1e-9), len(tris), float((a, b) in line_e),
                   float(bad.ext.get(a, False) and bad.ext.get(b, False)), sum(tris)])
        E.append((idx[a], idx[b]))
    return dict(
        name=bad.name, ids=ids, x=X,
        edge_index=np.array(E, dtype=np.int64).T, edge_attr=np.array(EA, dtype=np.float32),
        adj_bad=sorted(E), before=bad,
    )


def build_one(pair) -> dict:
    """쌍 하나 → 입력 그래프 + 정답 (keep, adj_good)."""
    bad, good = pair.before, pair.after
    g = graph_of(bad)
    ids = g["ids"]

    # 정답 — Good 을 좌표로 Bad 절점에 옮긴다
    pos = {pkey(bad.coords[n]): i for i, n in enumerate(ids)}
    gmap = {g: pos.get(pkey(c)) for g, c in good.coords.items()}
    used_good = {n for s in good.shells for n in s[2]}
    keep_pos = {pkey(good.coords[n]) for n in used_good}
    keep = np.array([pkey(bad.coords[n]) in keep_pos for n in ids], dtype=np.float32)
    adj_good = set()
    unmapped = 0
    for a, b in sides(good.shells):
        i, j = gmap.get(a), gmap.get(b)
        if i is None or j is None:
            unmapped += 1
            continue
        adj_good.add((min(i, j), max(i, j)))

    g.update(name=pair.name, keep=keep, adj_good=sorted(adj_good),
             unmapped_good_edges=unmapped, after=good)
    return g


def main() -> None:
    from bdf_io import discover_pairs
    from pairs import all_pairs, discover_good_pairs, load_pair

    ap = argparse.ArgumentParser(description="VGAE 학습 데이터 (508쌍 전부)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    vp.ensure_dirs()
    good = discover_good_pairs()
    orig, _, _ = discover_pairs(vp.gp.FE_DIR)
    names = all_pairs()[: a.limit] if a.limit else all_pairs()
    t0 = time.time()
    out = []
    for k, nm in enumerate(names, 1):
        out.append(build_one(load_pair(nm, good, orig)))
        if k % 100 == 0:
            print(f"  ... {k}/{len(names)}  ({time.time() - t0:.0f}s)")

    n_node = np.array([len(d["ids"]) for d in out])
    n_rm = np.array([int((d["keep"] == 0).sum()) for d in out])
    n_chg = np.array([len(set(d["adj_good"]) ^ set(d["adj_bad"])) for d in out])
    print(f"쌍 {len(out)}개 / 절점 수 중앙값 {int(np.median(n_node))} (최대 {n_node.max()})")
    print(f"사라지는 절점: 메쉬당 평균 {n_rm.mean():.2f}, 0개인 메쉬 {int((n_rm == 0).sum())}")
    print(f"바뀌는 변(없어짐+생김): 메쉬당 평균 {n_chg.mean():.2f}, 0개인 메쉬 {int((n_chg == 0).sum())}")
    print(f"Bad 절점으로 못 옮긴 Good 변: 메쉬당 평균 {np.mean([d['unmapped_good_edges'] for d in out]):.2f}")
    with open(vp.DATA_PATH, "wb") as fh:
        pickle.dump(out, fh)
    print(f"저장: {vp.DATA_PATH}")


if __name__ == "__main__":
    main()

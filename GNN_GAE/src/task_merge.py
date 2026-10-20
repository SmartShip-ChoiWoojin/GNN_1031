"""1단계 과제 — "어느 삼각형 쌍을 합칠 것인가".

가장 단순한 수정 종류부터 푼다. `연결만 변경` 196건 중 **97%가 merge_tris 반복**이다
(삼각형 2개 → 사각형 1개를 1~4회). 연산은 이미 `meshedit` 에 있고 회귀 검증도 돼 있다.
남은 문제는 **고르기** 뿐이다.

    각 메쉬마다 합칠 수 있는 삼각형 쌍이 여러 개 있다.
    사람은 그중 일부만 합쳤다. 어느 것인가?

이 과제가 좋은 이유:
  - 정답이 명확하다 (사람이 합친 쌍 = 사라진 변)
  - 행동 공간이 작다 (메쉬당 후보 몇 개)
  - 결과를 정확히 채점할 수 있다 (Good 메쉬와 쉘 집합이 같은가)

    python src/task_merge.py            # 과제 정의 + 베이스라인 채점
    python src/task_merge.py --dump out/task_merge.csv
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass

import numpy as np

import gae_paths as gp
import features as F
from bdf_io import discover_pairs
from meshedit import EditableMesh, EditError, MergeCand, merge_candidates
from meshrep import Edge
from pairs import all_pairs, discover_good_pairs, load_pair


@dataclass
class Sample:
    """메쉬 하나 = 후보 목록 + 사람이 고른 것."""

    name: str
    mesh: EditableMesh
    cands: list[MergeCand]
    #: 사람이 실제로 합친 쌍의 공유 변 (정답)
    truth: set
    #: 사람이 한 merge 횟수
    n_truth: int

    @property
    def n_cand(self) -> int:
        return len(self.cands)

    def cand_edge(self, c: MergeCand) -> Edge:
        return Edge(tuple(set(self.mesh.shells[c.si][2]) & set(self.mesh.shells[c.sj][2])))


def shell_edges(mesh) -> set:
    return {Edge(e) for _, _, n in mesh.shells for e in zip(n, n[1:] + n[:1])}


def is_merge_only(p) -> int | None:
    """순수 merge 반복인가. 맞으면 merge 횟수, 아니면 None."""
    tri = lambda m: sum(1 for _, t, _ in m.shells if t == "CTRIA3")
    quad = lambda m: sum(1 for _, t, _ in m.shells if t == "CQUAD4")
    dt = tri(p.after) - tri(p.before)
    dq = quad(p.after) - quad(p.before)
    if dt < 0 and dt == -2 * dq:
        return dq
    return None


def build(limit: int | None = None) -> list[Sample]:
    """`연결만 변경` 중 순수 merge 인 메쉬를 과제 표본으로 만든다."""
    from app_pairs import classify

    good = discover_good_pairs()
    orig, _, _ = discover_pairs(gp.FE_DIR)
    out = []
    for nm in all_pairs()[:limit] if limit else all_pairs():
        p = load_pair(nm, good, orig)
        if classify(p) != "연결만 변경":
            continue
        k = is_merge_only(p)
        if not k:
            continue
        m = EditableMesh(p.before)
        truth = shell_edges(p.before) - shell_edges(p.after)
        # drop_lines=True 로 열거한다 — 사람은 1D 가 올라탄 변도 없앤다
        cands = [c for c in merge_candidates(m, drop_lines=True) if c.ok]
        out.append(Sample(nm, m, cands, truth, k))
    return out


# ──────────────────────────────────────────────────────────────────────────
# 후보를 메쉬 전체 변으로 여는 모드
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class EdgeCand:
    """내부 변 하나를 후보로 본 것. 합칠 수 없는 변도 포함한다.

    지금까지는 "합칠 수 있는 삼각형 쌍" 만 후보로 줬다(메쉬당 2.8개). 그건 유효성
    판정을 사람이 대신 해 준 셈이라, 나중에 연산을 여러 개로 늘릴 때의 조건과 다르다.
    통합 형태에서는 **모든 변**이 후보이고 그 위에서 무엇을 할지 고르게 된다.
    """

    si: int
    sj: int
    edge: Edge
    #: 삼각형 2개라 합칠 수 있는가 (유효성 — 모델에게 피처로만 준다)
    mergeable: bool
    #: 합친 결과 사각형의 품질. 못 합치면 None
    quality: F.ShellQuality | None
    dihedral: float
    lines_on_edge: int


def all_edge_candidates(m: EditableMesh) -> list[EdgeCand]:
    """쉘 2개가 공유하는 변을 전부 후보로 만든다 (메쉬당 ~100개)."""
    rep = m.rep
    out = []
    for e, v in rep.edge_shells.items():
        if len(v) != 2:
            continue   # 자유변/비매니폴드 변은 '두 요소 사이' 가 아니다
        si, sj = v
        qi, qj = rep.shells[si].quality, rep.shells[sj].quality
        dih = F.dihedral_deg(qi.normal, qj.normal)
        both_tri = (not qi.is_quad) and (not qj.is_quad)
        q = None
        if both_tri:
            try:
                q = m.merged_quality(si, sj, drop_lines=True)
            except EditError:
                q = None
        out.append(EdgeCand(si, sj, Edge(e), q is not None, q, dih, len(m.lines_on(e))))
    return out


def build_all_edges(limit: int | None = None) -> list[Sample]:
    """`build` 와 같은 표본이되 후보를 메쉬 전체 변으로 연다."""
    from app_pairs import classify

    good = discover_good_pairs()
    orig, _, _ = discover_pairs(gp.FE_DIR)
    out = []
    for nm in (all_pairs()[:limit] if limit else all_pairs()):
        p = load_pair(nm, good, orig)
        if classify(p) != "연결만 변경":
            continue
        k = is_merge_only(p)
        if not k:
            continue
        m = EditableMesh(p.before)
        truth = shell_edges(p.before) - shell_edges(p.after)
        out.append(Sample(nm, m, all_edge_candidates(m), truth, k))
    return out


def verify(s: Sample) -> bool:
    """정답 변을 합치면 Good 메쉬의 쉘 집합이 나오는가 (연산 자체의 재현성)."""
    good = discover_good_pairs()
    orig, _, _ = discover_pairs(gp.FE_DIR)
    p = load_pair(s.name, good, orig)
    t = s.mesh.clone()
    for e in s.truth:
        # 매번 다시 찾는다 — 합치면 쉘 인덱스가 밀린다
        hit = next((c for c in merge_candidates(t, drop_lines=True)
                    if Edge(tuple(set(t.shells[c.si][2]) & set(t.shells[c.sj][2]))) == e), None)
        if hit is None:
            return False
        t.merge_tris(hit.si, hit.sj, drop_lines=True)
    return {frozenset(x[2]) for x in t.shells} == {frozenset(n) for _, _, n in p.after.shells}


# ──────────────────────────────────────────────────────────────────────────
# 베이스라인 — 학습 없이 고르기
# ──────────────────────────────────────────────────────────────────────────
def score_quality(s: Sample, c: MergeCand) -> float:
    """합친 사각형이 반듯할수록 낮은 값. 규칙 기반 선택의 기준."""
    q = c.quality
    assert q is not None
    return q.skew + 10.0 * max(0.0, q.aspect - 1.0) + 2.0 * c.dihedral


def score_badness(s: Sample, c: MergeCand) -> float:
    """합쳐질 두 삼각형이 나쁠수록 낮은 값 — '나쁜 것부터 없앤다'."""
    qi = F.shell_quality(s.mesh.corners(c.si))
    qj = F.shell_quality(s.mesh.corners(c.sj))
    return -(max(qi.skew, qj.skew) + 10 * max(qi.aspect, qj.aspect))


def topk_hit(s: Sample, scorer, k: int) -> bool:
    """점수 상위 k개 안에 사람이 고른 변이 하나라도 있나."""
    if not s.cands:
        return False
    ranked = sorted(s.cands, key=lambda c: scorer(s, c))
    return any(s.cand_edge(c) in s.truth for c in ranked[:k])


def exact_hit(s: Sample, scorer) -> bool:
    """상위 n_truth 개를 골랐을 때 사람이 고른 집합과 정확히 일치하나."""
    if len(s.cands) < s.n_truth:
        return False
    ranked = sorted(s.cands, key=lambda c: scorer(s, c))[: s.n_truth]
    return {s.cand_edge(c) for c in ranked} == s.truth


def main() -> None:
    ap = argparse.ArgumentParser(description="1단계 과제 정의와 베이스라인")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dump", default=None, help="표본을 csv 로 저장")
    ap.add_argument("--no-verify", action="store_true", help="재현성 검사를 건너뛴다")
    a = ap.parse_args()

    print("과제 표본을 만드는 중...")
    S = build(a.limit)
    nc = np.array([s.n_cand for s in S])
    nt = np.array([s.n_truth for s in S])
    print(f"\n표본 {len(S)}개 ('연결만 변경' 중 순수 merge)")
    print(f"  사람이 합친 횟수: {dict(collections.Counter(nt.tolist()))}")
    print(f"  메쉬당 후보 수: 중앙값 {np.median(nc):.0f} / 평균 {nc.mean():.1f} / "
          f"최대 {nc.max()} / 0개인 메쉬 {int((nc == 0).sum())}")

    # 정답이 후보 안에 들어 있나 — 이게 안 되면 고르기 이전에 열거가 틀린 것이다
    cover = [len({c for c in s.truth} & {s.cand_edge(x) for x in s.cands}) / s.n_truth
             for s in S if s.n_truth]
    cover = np.array(cover)
    print(f"\n정답이 후보 목록에 들어 있는 비율: 평균 {cover.mean():.1%} / "
          f"전부 들어 있는 메쉬 {int((cover >= 0.999).sum())}/{len(S)}")

    if not a.no_verify:
        okv = sum(verify(s) for s in S)
        print(f"정답 변을 합치면 Good 메쉬가 재현되는 메쉬: {okv}/{len(S)} ({okv/len(S):.1%})")

    print("\n── 학습 없는 베이스라인 ──")
    full = [s for s in S if s.n_truth and
            {c for c in s.truth} <= {s.cand_edge(x) for x in s.cands}]
    print(f"   (정답이 후보에 다 들어 있는 {len(full)}개 기준)")
    print(f"{'방법':20s}{'상위1 적중':>12s}{'상위3':>8s}{'정확히 일치':>12s}")
    for nm_, sc in (("결과가 반듯한 순", score_quality), ("나쁜 삼각형부터", score_badness)):
        h1 = np.mean([topk_hit(s, sc, 1) for s in full])
        h3 = np.mean([topk_hit(s, sc, 3) for s in full])
        ex = np.mean([exact_hit(s, sc) for s in full])
        print(f"  {nm_:18s}{h1:11.1%}{h3:8.1%}{ex:11.1%}")
    rnd = np.mean([1.0 / s.n_cand * s.n_truth if s.n_cand else 0 for s in full])
    print(f"  {'무작위(기댓값)':18s}{rnd:11.1%}")

    if a.dump:
        import pandas as pd

        rows = []
        for s in S:
            for c in s.cands:
                e = s.cand_edge(c)
                q = c.quality
                rows.append(dict(name=s.name, edge=f"{min(e)}-{max(e)}",
                                 y=int(e in s.truth), n_cand=s.n_cand, n_truth=s.n_truth,
                                 dihedral=c.dihedral, lines_on_edge=c.lines_on_edge,
                                 q_skew=q.skew, q_aspect=q.aspect, q_warp=q.warp,
                                 q_angle_min=q.angle_min, q_angle_max=q.angle_max,
                                 q_taper=q.taper, q_area=q.area))
        pd.DataFrame(rows).to_csv(a.dump, index=False, encoding="utf-8-sig")
        print(f"\n저장: {a.dump}  (후보 {len(rows)}개)")


if __name__ == "__main__":
    main()

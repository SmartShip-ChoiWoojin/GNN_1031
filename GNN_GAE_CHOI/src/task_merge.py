"""1단계 과제(초기 시도) — "어느 삼각형 쌍을 합칠 것인가". 지금은 regen_* 의 2단계로 흡수됐다.

`연결만 변경` 196건 중 **97%가 merge_tris 반복**이다 (삼각형 2개 → 사각형 1개를 1~4회).
연산은 이미 `meshedit` 에 있으므로 남은 문제는 **고르기** 뿐이다.

    각 메쉬마다 합칠 수 있는 삼각형 쌍이 여러 개 있다.
    사람은 그중 일부만 합쳤다. 어느 것인가?

    python src/task_merge.py            # 과제 정의 + 학습 없는 기준선 채점
    python src/task_merge.py --dump out/task_merge.csv
"""

from __future__ import annotations

import argparse

import numpy as np

import gae_paths
import features as F
from bdf_io import discover_pairs
from meshedit import EditableMesh, EditError, merge_candidates, shared_edge_of
from meshrep import Edge
from pairs import all_pairs, discover_good_pairs, load_pair
from regen_data import kind_of, shell_edge_set


class Sample:
    """메쉬 하나 = 후보 목록 + 사람이 고른 것."""

    def __init__(self, name: str, mesh: EditableMesh, candidates: list, truth: set, n_truth: int):
        self.name = name
        self.mesh = mesh
        #: 합칠 수 있는 삼각형 쌍 후보 (MergeCand 또는 EdgeCand)
        self.candidates = candidates
        #: 사람이 실제로 합친 쌍의 공유 변 (정답)
        self.truth = truth
        #: 사람이 한 merge 횟수
        self.n_truth = n_truth

    @property
    def n_cand(self) -> int:
        return len(self.candidates)

    def candidate_edge(self, candidate) -> frozenset:
        return shared_edge_of(self.mesh, candidate.si, candidate.sj)


def count_type(mesh, element_type: str) -> int:
    count = 0
    for element_id, shell_type, node_ids in mesh.shells:
        if shell_type == element_type:
            count = count + 1
    return count


def merge_only_count(pair):
    """순수 merge 반복인가 — 삼각형이 2개씩 줄고 사각형이 1개씩 늘었나. 맞으면 merge 횟수, 아니면 None."""
    triangle_change = count_type(pair.after, "CTRIA3") - count_type(pair.before, "CTRIA3")
    quad_change = count_type(pair.after, "CQUAD4") - count_type(pair.before, "CQUAD4")
    if triangle_change < 0 and triangle_change == -2 * quad_change:
        return quad_change
    return None


def merge_only_pairs(limit: int | None = None) -> list:
    """`연결만 변경` 중 순수 merge 인 쌍 [(Pair, merge 횟수)]."""
    good_paths = discover_good_pairs()
    original_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    names = all_pairs()
    if limit:
        names = names[:limit]
    result = []
    for name in names:
        pair = load_pair(name, good_paths, original_paths)
        if kind_of(pair) != "연결만 변경":
            continue
        merge_count = merge_only_count(pair)
        if not merge_count:
            continue
        result.append((pair, merge_count))
    return result


def build(limit: int | None = None) -> list:
    """`연결만 변경` 중 순수 merge 인 메쉬를 과제 표본으로 만든다."""
    samples = []
    for pair, merge_count in merge_only_pairs(limit):
        mesh = EditableMesh(pair.before)
        truth = shell_edge_set(pair.before) - shell_edge_set(pair.after)
        # drop_lines=True 로 열거한다 — 사람은 1D 가 올라탄 변도 없앤다
        candidates = []
        for candidate in merge_candidates(mesh, drop_lines=True):
            if candidate.ok:
                candidates.append(candidate)
        samples.append(Sample(pair.name, mesh, candidates, truth, merge_count))
    return samples


# ──────────────────────────────────────────────────────────────────────────
# 후보를 메쉬 전체 변으로 여는 모드
# ──────────────────────────────────────────────────────────────────────────
class EdgeCand:
    """내부 변 하나를 후보로 본 것. 합칠 수 없는 변도 포함한다."""

    def __init__(self, si, sj, edge, mergeable, quality, dihedral, lines_on_edge):
        self.si = si
        self.sj = sj
        self.edge = edge
        #: 삼각형 2개라 합칠 수 있는가 (유효성 — 모델에게 피처로만 준다)
        self.mergeable = mergeable
        #: 합친 결과 사각형의 품질. 못 합치면 None
        self.quality = quality
        self.dihedral = dihedral
        self.lines_on_edge = lines_on_edge


def all_edge_candidates(mesh: EditableMesh) -> list:
    """쉘 2개가 공유하는 변을 전부 후보로 만든다 (메쉬당 ~100개)."""
    rep = mesh.rep
    result = []
    for edge, shell_indices in rep.edge_shells.items():
        if len(shell_indices) != 2:
            # 자유변/비매니폴드 변은 '두 요소 사이' 가 아니다
            continue
        shell_index_a, shell_index_b = shell_indices
        quality_a = rep.shells[shell_index_a].quality
        quality_b = rep.shells[shell_index_b].quality
        dihedral = F.dihedral_deg(quality_a.normal, quality_b.normal)
        both_triangles = (not quality_a.is_quad) and (not quality_b.is_quad)
        merged_quality = None
        if both_triangles:
            try:
                merged_quality = mesh.merged_quality(shell_index_a, shell_index_b, drop_lines=True)
            except EditError:
                merged_quality = None
        result.append(EdgeCand(shell_index_a, shell_index_b, Edge(edge), merged_quality is not None, merged_quality,
                               dihedral, len(mesh.lines_on(edge))))
    return result


def build_all_edges(limit: int | None = None) -> list:
    """`build` 와 같은 표본이되 후보를 메쉬 전체 변으로 연다."""
    samples = []
    for pair, merge_count in merge_only_pairs(limit):
        mesh = EditableMesh(pair.before)
        truth = shell_edge_set(pair.before) - shell_edge_set(pair.after)
        samples.append(Sample(pair.name, mesh, all_edge_candidates(mesh), truth, merge_count))
    return samples


def verify(sample: Sample) -> bool:
    """정답 변을 합치면 Good 메쉬의 쉘 집합이 나오는가 (연산 자체의 재현성)."""
    pair = load_pair(sample.name, discover_good_pairs(), discover_pairs(gae_paths.FE_DIR)[0])
    mesh = sample.mesh.clone()
    for edge in sample.truth:
        # 매번 다시 찾는다 — 합치면 쉘 인덱스가 밀린다
        found = None
        for candidate in merge_candidates(mesh, drop_lines=True):
            if shared_edge_of(mesh, candidate.si, candidate.sj) == edge:
                found = candidate
                break
        if found is None:
            return False
        mesh.merge_tris(found.si, found.sj, drop_lines=True)

    result_sets = set()
    for shell in mesh.shells:
        result_sets.add(frozenset(shell[2]))
    good_sets = set()
    for element_id, element_type, node_ids in pair.after.shells:
        good_sets.add(frozenset(node_ids))
    return result_sets == good_sets


# ──────────────────────────────────────────────────────────────────────────
# 기준선 — 학습 없이 고르기
# ──────────────────────────────────────────────────────────────────────────
def score_quality(sample: Sample, candidate) -> float:
    """합친 사각형이 반듯할수록 낮은 값. 규칙 기반 선택의 기준."""
    merged_quality = candidate.quality
    return merged_quality.skew + 10.0 * max(0.0, merged_quality.aspect - 1.0) + 2.0 * candidate.dihedral


def score_badness(sample: Sample, candidate) -> float:
    """합쳐질 두 삼각형이 나쁠수록 낮은 값 — '나쁜 것부터 없앤다'."""
    quality_a = F.shell_quality(sample.mesh.corners(candidate.si))
    quality_b = F.shell_quality(sample.mesh.corners(candidate.sj))
    return -(max(quality_a.skew, quality_b.skew) + 10 * max(quality_a.aspect, quality_b.aspect))


def ranked_candidates(sample: Sample, scorer) -> list:
    """점수가 낮은 순으로 정렬한 후보."""
    def score_of(candidate):
        return scorer(sample, candidate)

    return sorted(sample.candidates, key=score_of)


def top_k_hit(sample: Sample, scorer, k: int) -> bool:
    """점수 상위 k개 안에 사람이 고른 변이 하나라도 있나."""
    if len(sample.candidates) == 0:
        return False
    for candidate in ranked_candidates(sample, scorer)[:k]:
        if sample.candidate_edge(candidate) in sample.truth:
            return True
    return False


def exact_hit(sample: Sample, scorer) -> bool:
    """상위 n_truth 개를 골랐을 때 사람이 고른 집합과 정확히 일치하나."""
    if len(sample.candidates) < sample.n_truth:
        return False
    picked_edges = set()
    for candidate in ranked_candidates(sample, scorer)[:sample.n_truth]:
        picked_edges.add(sample.candidate_edge(candidate))
    return picked_edges == sample.truth


def candidate_edge_set(sample: Sample) -> set:
    edges = set()
    for candidate in sample.candidates:
        edges.add(sample.candidate_edge(candidate))
    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description="1단계 과제 정의와 기준선")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dump", default=None, help="표본을 csv 로 저장")
    parser.add_argument("--no-verify", action="store_true", help="재현성 검사를 건너뛴다")
    args = parser.parse_args()

    print("과제 표본을 만드는 중...")
    samples = build(args.limit)
    candidate_counts = []
    truth_counts = []
    for sample in samples:
        candidate_counts.append(sample.n_cand)
        truth_counts.append(sample.n_truth)
    candidate_counts = np.array(candidate_counts)
    truth_counts = np.array(truth_counts)
    print(f"\n표본 {len(samples)}개 ('연결만 변경' 중 순수 merge)")
    truth_count_table = {}
    for value in truth_counts.tolist():
        if value not in truth_count_table:
            truth_count_table[value] = 0
        truth_count_table[value] = truth_count_table[value] + 1
    print(f"  사람이 합친 횟수: {truth_count_table}")
    print(f"  메쉬당 후보 수: 중앙값 {np.median(candidate_counts):.0f} / 평균 {candidate_counts.mean():.1f} / "
          f"최대 {candidate_counts.max()} / 0개인 메쉬 {int((candidate_counts == 0).sum())}")

    # 정답이 후보 안에 들어 있나 — 이게 안 되면 고르기 이전에 열거가 틀린 것이다
    coverage = []
    for sample in samples:
        if sample.n_truth:
            coverage.append(len(sample.truth & candidate_edge_set(sample)) / sample.n_truth)
    coverage = np.array(coverage)
    print(f"\n정답이 후보 목록에 들어 있는 비율: 평균 {coverage.mean():.1%} / "
          f"전부 들어 있는 메쉬 {int((coverage >= 0.999).sum())}/{len(samples)}")

    if not args.no_verify:
        verified_count = 0
        for sample in samples:
            if verify(sample):
                verified_count = verified_count + 1
        print(f"정답 변을 합치면 Good 메쉬가 재현되는 메쉬: {verified_count}/{len(samples)} "
              f"({verified_count / len(samples):.1%})")

    print("\n── 학습 없는 베이스라인 ──")
    full_samples = []
    for sample in samples:
        if sample.n_truth and sample.truth.issubset(candidate_edge_set(sample)):
            full_samples.append(sample)
    print(f"   (정답이 후보에 다 들어 있는 {len(full_samples)}개 기준)")
    print(f"{'방법':20s}{'상위1 적중':>12s}{'상위3':>8s}{'정확히 일치':>12s}")
    for method_name, scorer in (("결과가 반듯한 순", score_quality), ("나쁜 삼각형부터", score_badness)):
        top1_hits = []
        top3_hits = []
        exact_hits = []
        for sample in full_samples:
            top1_hits.append(top_k_hit(sample, scorer, 1))
            top3_hits.append(top_k_hit(sample, scorer, 3))
            exact_hits.append(exact_hit(sample, scorer))
        print(f"  {method_name:18s}{np.mean(top1_hits):11.1%}{np.mean(top3_hits):8.1%}{np.mean(exact_hits):11.1%}")
    random_expectations = []
    for sample in full_samples:
        if sample.n_cand:
            random_expectations.append(1.0 / sample.n_cand * sample.n_truth)
        else:
            random_expectations.append(0)
    print(f"  {'무작위(기댓값)':18s}{np.mean(random_expectations):11.1%}")

    if args.dump:
        import pandas as pd

        rows = []
        for sample in samples:
            for candidate in sample.candidates:
                edge = sample.candidate_edge(candidate)
                merged_quality = candidate.quality
                rows.append(dict(name=sample.name, edge=f"{min(edge)}-{max(edge)}", y=int(edge in sample.truth),
                                 n_cand=sample.n_cand, n_truth=sample.n_truth,
                                 dihedral=candidate.dihedral, lines_on_edge=candidate.lines_on_edge,
                                 q_skew=merged_quality.skew, q_aspect=merged_quality.aspect,
                                 q_warp=merged_quality.warp, q_angle_min=merged_quality.angle_min,
                                 q_angle_max=merged_quality.angle_max, q_taper=merged_quality.taper,
                                 q_area=merged_quality.area))
        pd.DataFrame(rows).to_csv(args.dump, index=False, encoding="utf-8-sig")
        print(f"\n저장: {args.dump}  (후보 {len(rows)}개)")


if __name__ == "__main__":
    main()

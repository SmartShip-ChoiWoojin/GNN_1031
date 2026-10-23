"""Stage 0 — 규칙 기반 수리기. 학습 모델이 넘어야 할 선 (지금은 참고용으로 남긴 초기 접근).

학습 모델을 만들기 전에 "규칙만으로 어디까지 되는가" 를 확정해 두려고 만들었다.
당시 가정: 불량 요소(내각<30° 또는 >150°)의 95%가 삼각형이니 **삼각형을 사각형으로
되돌리는 것**이 수리의 본체다. (사람의 실제 수정 508쌍이 들어온 뒤 이 가정은 틀린 것으로
드러났다 — 사람은 주로 절점을 합친다. README 참고.)

여기 구현한 세 가지 전략:
  1. ``greedy_merge``        — 합쳐서 멀쩡한 사각형이 되는 인접 삼각형 쌍을 좋은 것부터
  2. ``swap_assisted_merge`` — 바로는 못 합치지만 대각선을 한 번 뒤집으면 합쳐지는 쌍까지
  3. ``repair``              — 위 둘을 수렴할 때까지

    python src/repair_rule.py --n 150
"""

from __future__ import annotations

import argparse

import numpy as np

import gae_paths
import features as F
from meshedit import EditableMesh, EditError, check, merge_candidates


class RepairConfig:
    """병합을 허용할 조건. 전부 '합쳐도 되는가' 의 도메인 판단이다."""

    def __init__(self, max_dihedral: float = 10.0, max_skew: float = 45.0, max_aspect: float = 4.0,
                 reject_bad_angle: bool = True, use_swap: bool = True, max_rounds: int = 8):
        #: 두 삼각형 사이 이면각(도). 이보다 크면 서로 다른 패널이라 합치면 형상이 바뀐다.
        self.max_dihedral = max_dihedral
        #: 결과 사각형이 이보다 찌그러지면 합치지 않는다 (90°에서의 이탈)
        self.max_skew = max_skew
        #: 결과 사각형의 최장변/최단변 상한
        self.max_aspect = max_aspect
        #: 결과 사각형이 불량각(내각<30° 또는 >150°)이면 합치지 않는다
        self.reject_bad_angle = reject_bad_angle
        #: swap 을 거친 뒤 합치는 2수 전략을 쓸지
        self.use_swap = use_swap
        #: 수렴 반복 상한
        self.max_rounds = max_rounds


def acceptable(candidate, config: RepairConfig) -> bool:
    """이 병합을 해도 되는가."""
    if not candidate.ok or candidate.quality is None:
        return False
    if candidate.dihedral > config.max_dihedral:
        return False
    merged_quality = candidate.quality
    if merged_quality.skew > config.max_skew or merged_quality.aspect > config.max_aspect:
        return False
    if config.reject_bad_angle and F.is_bad_shell(merged_quality):
        return False
    return True


def merge_score(candidate) -> float:
    """작을수록 먼저 합친다. 결과 사각형이 직사각형에 가까울수록 좋다."""
    merged_quality = candidate.quality
    return merged_quality.skew + 10.0 * max(0.0, merged_quality.aspect - 1.0) + 2.0 * candidate.dihedral


def candidate_dihedral(candidate) -> float:
    return candidate.dihedral


def find_shell(mesh: EditableMesh, node_set: frozenset):
    """절점 집합으로 쉘 인덱스를 찾는다 (편집하면 인덱스가 밀리므로). 없으면 None."""
    for shell_index in range(len(mesh.shells)):
        if frozenset(mesh.shells[shell_index][2]) == node_set:
            return shell_index
    return None


def greedy_merge(mesh: EditableMesh, config: RepairConfig) -> list:
    """합칠 수 있는 삼각형 쌍을 점수 좋은 순으로, 겹치지 않게 합친다. 적용한 연산 목록.

    한 라운드 안에서 쉘 인덱스가 밀리므로 **절점 집합으로** 대상을 기억했다가 매번
    다시 찾는다 — 인덱스를 저장해 두면 조용히 엉뚱한 요소를 합치게 된다.
    """
    operations = []
    for round_number in range(config.max_rounds):
        candidates = []
        for candidate in merge_candidates(mesh):
            if acceptable(candidate, config):
                candidates.append(candidate)
        if len(candidates) == 0:
            break
        candidates.sort(key=merge_score)

        # 이번 라운드에 합칠 쌍을 절점 집합으로 예약 (겹치는 삼각형은 한 번만)
        used_shells = set()
        picked = []
        for candidate in candidates:
            if candidate.si in used_shells or candidate.sj in used_shells:
                continue
            used_shells.add(candidate.si)
            used_shells.add(candidate.sj)
            picked.append((frozenset(mesh.shells[candidate.si][2]), frozenset(mesh.shells[candidate.sj][2])))
        if len(picked) == 0:
            break

        done_count = 0
        for node_set_a, node_set_b in picked:
            shell_index_a = find_shell(mesh, node_set_a)
            shell_index_b = find_shell(mesh, node_set_b)
            if shell_index_a is None or shell_index_b is None:
                continue
            try:
                operations.append(mesh.merge_tris(shell_index_a, shell_index_b))
                done_count = done_count + 1
            except EditError:
                continue
        if done_count == 0:
            break
    return operations


def swap_assisted_merge(mesh: EditableMesh, config: RepairConfig) -> list:
    """바로는 못 합치는 삼각형 쌍을, 대각선을 한 번 뒤집어 합칠 수 있게 만든다.

    뒤집기 자체는 삼각형 수를 안 줄이므로, **뒤집은 뒤 실제로 병합이 성사될 때만**
    확정하고 아니면 되돌린다(사본에서만 해 본다).
    """
    operations = []
    for round_number in range(config.max_rounds):
        gained = False
        candidates = merge_candidates(mesh)
        # 지금은 못 합치는 쌍만 본다
        for candidate in sorted(candidates, key=candidate_dihedral):
            if acceptable(candidate, config) or candidate.blocked is not None:
                continue
            if candidate.dihedral > config.max_dihedral:
                continue
            node_set_a = frozenset(mesh.shells[candidate.si][2])
            node_set_b = frozenset(mesh.shells[candidate.sj][2])
            trial_mesh = mesh.clone()
            shell_index_a = find_shell(trial_mesh, node_set_a)
            shell_index_b = find_shell(trial_mesh, node_set_b)
            if shell_index_a is None or shell_index_b is None:
                continue
            try:
                trial_mesh.swap_edge(shell_index_a, shell_index_b)
            except EditError:
                continue

            after_swap = []
            for trial_candidate in merge_candidates(trial_mesh):
                if acceptable(trial_candidate, config):
                    after_swap.append(trial_candidate)
            if len(after_swap) == 0:
                continue
            after_swap.sort(key=merge_score)
            best_candidate = after_swap[0]
            try:
                trial_mesh.merge_tris(best_candidate.si, best_candidate.sj)
            except EditError:
                continue

            if check(trial_mesh, mesh).ok:
                for operation in trial_mesh.log[len(mesh.log):]:
                    operations.append(operation)
                mesh.shells = trial_mesh.shells
                mesh.coords = trial_mesh.coords
                mesh.lines = trial_mesh.lines
                mesh.log = trial_mesh.log
                mesh._mark_changed()
                gained = True
                break
        if not gained:
            break
    return operations


def repair(mesh: EditableMesh, config: RepairConfig | None = None):
    """규칙 기반 수리 전체. 원본은 건드리지 않고 (수리된 사본, 적용한 연산 목록) 을 돌려준다."""
    if config is None:
        config = RepairConfig()
    repaired = mesh.clone()
    operations = greedy_merge(repaired, config)
    if config.use_swap:
        operations = operations + swap_assisted_merge(repaired, config)
        # swap 이 열어준 새 후보를 다시 훑는다
        operations = operations + greedy_merge(repaired, config)
    return repaired, operations


def count_bad_shells(mesh: EditableMesh) -> int:
    bad_count = 0
    for shell_index in range(len(mesh.shells)):
        if F.is_bad_shell(mesh.quality(shell_index)):
            bad_count = bad_count + 1
    return bad_count


# ──────────────────────────────────────────────────────────────────────────
# 진단 — 규칙으로 어디까지 되는가
# ──────────────────────────────────────────────────────────────────────────
def diagnose(names: list, mesh_paths: dict, config: RepairConfig) -> dict:
    """비정상 메쉬 묶음에 규칙 수리기를 돌리고 상한을 잰다."""
    triangles_before = 0
    triangles_after = 0
    shell_total = 0
    bad_before = 0
    bad_after = 0
    blocked_by_line = 0
    blocked_other = 0
    pair_total = 0
    far_pair_count = 0
    violation_count = 0
    per_mesh_removed_fraction = []

    for name in names:
        bdf_path, json_path = mesh_paths[name]
        mesh = EditableMesh.load(bdf_path, json_path)
        candidates = merge_candidates(mesh)
        pair_total = pair_total + len(candidates)
        for candidate in candidates:
            if candidate.dihedral > config.max_dihedral:
                far_pair_count = far_pair_count + 1
            if candidate.blocked is None:
                continue
            if "1D" in candidate.blocked:
                blocked_by_line = blocked_by_line + 1
            else:
                blocked_other = blocked_other + 1

        mesh_triangles_before = len(mesh.tri_indices())
        mesh_bad_before = count_bad_shells(mesh)
        fixed_mesh, _ = repair(mesh, config)
        if not check(fixed_mesh, mesh).ok:
            violation_count = violation_count + 1
        mesh_triangles_after = len(fixed_mesh.tri_indices())
        mesh_bad_after = count_bad_shells(fixed_mesh)

        triangles_before = triangles_before + mesh_triangles_before
        triangles_after = triangles_after + mesh_triangles_after
        bad_before = bad_before + mesh_bad_before
        bad_after = bad_after + mesh_bad_after
        shell_total = shell_total + len(mesh.shells)
        per_mesh_removed_fraction.append((mesh_triangles_before - mesh_triangles_after) / max(1, mesh_triangles_before))

    return dict(
        n_mesh=len(names),
        tri_before=triangles_before,
        tri_after=triangles_after,
        tri_removed_frac=(triangles_before - triangles_after) / max(1, triangles_before),
        bad_before=bad_before,
        bad_after=bad_after,
        bad_removed_frac=(bad_before - bad_after) / max(1, bad_before),
        tri_frac_before=triangles_before / max(1, shell_total),
        pair_total=pair_total,
        pair_blocked_1d=blocked_by_line,
        pair_blocked_other=blocked_other,
        pair_far=far_pair_count,
        per_mesh_median=float(np.median(per_mesh_removed_fraction)),
        violations=violation_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="규칙 기반 수리기 진단")
    parser.add_argument("--n", type=int, default=150, help="검사할 메쉬 수")
    parser.add_argument("--cls", type=int, default=1, help="1=비정상, 0=정상")
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--max-dihedral", type=float, default=10.0)
    args = parser.parse_args()

    from bdf_io import discover_pairs
    from labels import load_labels

    mesh_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    labels = load_labels(gae_paths.LABEL_XLSX)
    names = []
    for name in sorted(labels):
        if name in mesh_paths and labels[name].cls == args.cls:
            names.append(name)
    names = names[:args.n]

    config = RepairConfig(use_swap=not args.no_swap, max_dihedral=args.max_dihedral)
    report = diagnose(names, mesh_paths, config)

    if args.cls == 1:
        tag = "비정상"
    else:
        tag = "정상"
    if config.use_swap:
        swap_text = "사용"
    else:
        swap_text = "미사용"
    print(f"\n{tag} 메쉬 {report['n_mesh']}개 — 규칙 기반 수리기 (swap {swap_text})")
    print(f"  삼각형        {report['tri_before']} → {report['tri_after']}  "
          f"({report['tri_removed_frac']:.1%} 제거, 메쉬별 중앙값 {report['per_mesh_median']:.1%})")
    print(f"  불량각 요소   {report['bad_before']} → {report['bad_after']}  ({report['bad_removed_frac']:.1%} 제거)")
    triangle_fraction_after = report['tri_frac_before'] * report['tri_after'] / max(1, report['tri_before'])
    print(f"  삼각형 비율   {report['tri_frac_before']:.3f} → {triangle_fraction_after:.3f}")
    print(f"  인접 삼각형 쌍 {report['pair_total']}개 중")
    print(f"    1D 요소가 막음      {report['pair_blocked_1d']} "
          f"({report['pair_blocked_1d'] / max(1, report['pair_total']):.1%})")
    print(f"    기하적으로 불가     {report['pair_blocked_other']}")
    print(f"    다른 패널(이면각 큼) {report['pair_far']} ({report['pair_far'] / max(1, report['pair_total']):.1%})")
    print(f"  불변식 위반 메쉬: {report['violations']}개")


if __name__ == "__main__":
    main()

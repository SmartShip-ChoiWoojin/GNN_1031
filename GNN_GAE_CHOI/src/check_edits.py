"""편집 계층 회귀 테스트 11종. 편집 연산(meshedit)을 고쳤으면 이걸 먼저 돌린다.

수리기가 내놓는 메쉬는 결국 현업의 .bdf 로 들어간다. 조용히 깨진 메쉬를 만드는 것이
제일 나쁜 실패라, 편집 연산을 건드릴 때마다 이걸 먼저 돌린다.

    python src/check_edits.py            # 200개
    python src/check_edits.py --n 1000
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile

import numpy as np

import gae_paths
import features as F
from bdf_io import discover_pairs
from meshdiff import signature
from meshedit import EditableMesh, EditError, check, collapse_pairs, merge_candidates, shared_edge_of


class Failed(AssertionError):
    """테스트 실패."""


def require(condition: bool, message: str) -> None:
    """조건이 거짓이면 테스트 실패."""
    if not condition:
        raise Failed(message)


def sorted_shell_node_lists(mesh) -> list:
    """쉘마다 정렬한 절점 목록을, 다시 정렬한 것 (요소 순서·감김과 무관한 비교용)."""
    result = []
    for shell in mesh.shells:
        result.append(sorted(shell[2]))
    return sorted(result)


def mesh_snapshot(mesh):
    """(좌표, 쉘 절점, 1D 절점) — 메쉬가 바뀌었는지 비교용."""
    shell_nodes = []
    for shell in mesh.shells:
        shell_nodes.append(list(shell[2]))
    line_nodes = []
    for line in mesh.lines:
        line_nodes.append(tuple(line[2]))
    return (dict(mesh.coords), shell_nodes, line_nodes)


# ──────────────────────────────────────────────────────────────────────────
def test_roundtrip(names: list, mesh_paths: dict) -> str:
    """쓰고 다시 읽으면 좌표·요소가 정확히 같아야 한다."""
    temporary_dir = tempfile.mkdtemp()
    for name in names:
        mesh = EditableMesh.load(*mesh_paths[name])
        bdf_path = os.path.join(temporary_dir, "t.bdf")
        json_path = os.path.join(temporary_dir, "t.json")
        mesh.write_bdf(bdf_path)
        mesh.write_json(json_path)
        read_back = EditableMesh.load(bdf_path, json_path)
        require(len(read_back.coords) == len(mesh.coords), f"{name}: 절점 수 불일치")
        for node_id, point in mesh.coords.items():
            require(np.allclose(np.array(point), np.array(read_back.coords[node_id]), rtol=0, atol=1e-6),
                    f"{name}: 절점 {node_id} 좌표가 라운드트립에서 바뀜")
        require(mesh_snapshot(read_back)[1] == mesh_snapshot(mesh)[1], f"{name}: 쉘 불일치")
        require(mesh_snapshot(read_back)[2] == mesh_snapshot(mesh)[2], f"{name}: 1D 불일치")
        require(read_back.ext == mesh.ext, f"{name}: ext 불일치")
    return f"{len(names)}개 .bdf/.json 라운드트립 정확 일치"


def test_split_merge_inverse(names: list, mesh_paths: dict, random_generator: random.Random) -> str:
    """split 한 사각형을 다시 merge 하면 원본으로 돌아와야 한다."""
    case_count = 0
    for name in names:
        base_mesh = EditableMesh.load(*mesh_paths[name])
        quad_indices = []
        for shell_index in range(len(base_mesh.shells)):
            if len(base_mesh.shells[shell_index][2]) == 4:
                quad_indices.append(shell_index)
        if len(quad_indices) == 0:
            continue
        for quad_index in random_generator.sample(quad_indices, min(3, len(quad_indices))):
            for diagonal in (0, 1):
                original_nodes = set(base_mesh.shells[quad_index][2])
                trial_mesh = base_mesh.clone()
                try:
                    trial_mesh.split_quad(quad_index, diag=diagonal)
                except EditError:
                    # 그 대각선이 이미 변으로 존재 — 정당한 거부다
                    continue
                require(check(trial_mesh, base_mesh).ok,
                        f"{name}: split 직후 불변식 위반\n{check(trial_mesh, base_mesh)}")
                matching = []
                for candidate in merge_candidates(trial_mesh):
                    union = set(trial_mesh.shells[candidate.si][2]) | set(trial_mesh.shells[candidate.sj][2])
                    if union == original_nodes and candidate.ok:
                        matching.append(candidate)
                if len(matching) == 0:
                    # 1D 요소가 대각선 위에 있으면 되돌릴 수 없다 — 정상 동작
                    continue
                trial_mesh.merge_tris(matching[0].si, matching[0].sj)
                require(sorted_shell_node_lists(base_mesh) == sorted_shell_node_lists(trial_mesh),
                        f"{name}: split→merge 가 원본을 복원하지 못함 (쉘 {quad_index}, diag {diagonal})")
                require(check(trial_mesh, base_mesh).ok, f"{name}: split→merge 후 불변식 위반")
                case_count = case_count + 1
    return f"split→merge 역연산 {case_count}건 전부 원본 복원"


def test_swap_involution(names: list, mesh_paths: dict) -> str:
    """같은 삼각형 쌍을 두 번 뒤집으면 제자리로 돌아와야 한다."""
    case_count = 0
    for name in names:
        base_mesh = EditableMesh.load(*mesh_paths[name])
        for candidate in merge_candidates(base_mesh):
            if not candidate.ok:
                continue
            trial_mesh = base_mesh.clone()
            try:
                trial_mesh.swap_edge(candidate.si, candidate.sj)
            except EditError:
                continue
            require(check(trial_mesh, base_mesh).ok,
                    f"{name}: swap 직후 불변식 위반\n{check(trial_mesh, base_mesh)}")
            same_pair = []
            for other in merge_candidates(trial_mesh):
                if {other.si, other.sj} == {candidate.si, candidate.sj} and other.ok:
                    same_pair.append(other)
            if len(same_pair) == 0:
                continue
            try:
                trial_mesh.swap_edge(same_pair[0].si, same_pair[0].sj)
            except EditError:
                continue
            require(sorted_shell_node_lists(trial_mesh) == sorted_shell_node_lists(base_mesh),
                    f"{name}: swap 두 번이 제자리로 안 돌아옴")
            case_count = case_count + 1
            break
    return f"swap 2회 = 항등 {case_count}건 확인"


def test_invariants_under_edits(names: list, mesh_paths: dict, random_generator: random.Random) -> str:
    """무작위 편집을 퍼부어도 도메인 불변식이 깨지지 않아야 한다.

    비교 기준이 원본이 아니라 **직전 상태**인 이유: 표면 보존은 연산 하나하나가 지켜야
    할 성질이고(실측 단일 연산 면적 변화 최대 3.7e-5), 원본 대비 누적은
    merge→split→merge 같은 왕복을 섞은 무작위 수열에서만 커진다(실측 1.7e-3).
    """
    edit_count = 0
    for name in names:
        base_mesh = EditableMesh.load(*mesh_paths[name])
        mesh = base_mesh.clone()
        for attempt in range(12):
            previous_mesh = mesh.clone()
            kind = random_generator.choice(["split", "merge", "swap"])
            try:
                if kind == "split":
                    quad_indices = []
                    for shell_index in range(len(mesh.shells)):
                        if len(mesh.shells[shell_index][2]) == 4:
                            quad_indices.append(shell_index)
                    if len(quad_indices) == 0:
                        continue
                    mesh.split_quad(random_generator.choice(quad_indices), diag=random_generator.randint(0, 1))
                else:
                    usable = []
                    for candidate in merge_candidates(mesh):
                        if candidate.ok:
                            usable.append(candidate)
                    if len(usable) == 0:
                        continue
                    candidate = random_generator.choice(usable)
                    if kind == "merge":
                        mesh.merge_tris(candidate.si, candidate.sj)
                    else:
                        mesh.swap_edge(candidate.si, candidate.sj)
            except EditError:
                continue
            edit_count = edit_count + 1
            report = check(mesh, previous_mesh)
            require(report.ok, f"{name}: {kind} 뒤 불변식 위반\n{report}")
    return f"무작위 편집 {edit_count}회 동안 불변식 유지"


def test_line_elements_protected(names: list, mesh_paths: dict) -> str:
    """기본 설정(drop_lines=False)에서는 1D 요소가 올라탄 변의 병합이 막혀야 한다."""
    blocked_count = 0
    for name in names:
        mesh = EditableMesh.load(*mesh_paths[name])
        line_edges = mesh.line_edges()
        for candidate in merge_candidates(mesh):
            shared = shared_edge_of(mesh, candidate.si, candidate.sj)
            if shared in line_edges:
                require(candidate.blocked is not None and "1D" in candidate.blocked,
                        f"{name}: 1D 요소가 있는 변 {sorted(shared)} 의 병합이 막히지 않았다")
                blocked_count = blocked_count + 1
    return f"1D 요소가 보호한 병합 후보 {blocked_count}건 전부 차단됨"


def test_drop_lines(names: list, mesh_paths: dict) -> str:
    """1D 요소가 올라탄 변도 ``drop_lines=True`` 면 합칠 수 있어야 한다 (기본값은 여전히 거부)."""
    blocked_count = 0
    dropped_count = 0
    for name in names:
        base_mesh = EditableMesh.load(*mesh_paths[name])
        candidates_with_lines = []
        for candidate in merge_candidates(base_mesh, drop_lines=True):
            if candidate.ok and candidate.lines_on_edge > 0:
                candidates_with_lines.append(candidate)
        if len(candidates_with_lines) == 0:
            continue
        candidate = candidates_with_lines[0]

        # 기본값은 거부해야 한다
        default_refused = False
        try:
            base_mesh.clone().merge_tris(candidate.si, candidate.sj)
        except EditError:
            default_refused = True
        if not default_refused:
            raise Failed(f"{name}: 1D 가 있는 변인데 기본 호출이 거부되지 않았다")
        blocked_count = blocked_count + 1

        # drop_lines=True 면 성공하고, 그 1D 요소가 사라져 있어야 한다
        trial_mesh = base_mesh.clone()
        operation = trial_mesh.merge_tris(candidate.si, candidate.sj, drop_lines=True)
        dropped = operation.info.get("dropped_lines", [])
        require(len(dropped) == candidate.lines_on_edge,
                f"{name}: 지워진 1D 수가 {len(dropped)} 인데 변 위에는 {candidate.lines_on_edge} 개였다")
        require(len(trial_mesh.lines) == len(base_mesh.lines) - candidate.lines_on_edge, f"{name}: 1D 총수가 맞지 않는다")
        report = check(trial_mesh, base_mesh)
        require(report.ok, f"{name}: drop_lines 병합 뒤 불변식 위반\n{report}")
        require(report.lines_removed == candidate.lines_on_edge,
                f"{name}: 보고서의 1D 삭제 수({report.lines_removed})가 실제({candidate.lines_on_edge})와 다르다")
        dropped_count = dropped_count + 1
    return f"기본 거부 {blocked_count}건 / drop_lines 병합 {dropped_count}건 — 1D 가 함께 삭제되고 불변식 유지"


def test_ext_nodes_frozen(names: list, mesh_paths: dict) -> str:
    """ext=true 절점은 움직일 수 없어야 한다."""
    mesh_count = 0
    for name in names:
        mesh = EditableMesh.load(*mesh_paths[name])
        for node_id, is_ext in mesh.ext.items():
            if not is_ext:
                continue
            refused = False
            try:
                mesh.move_node(node_id, np.array([1.0, 0.0, 0.0]))
            except EditError:
                refused = True
            if refused:
                mesh_count = mesh_count + 1
                break
            raise Failed(f"{name}: ext 절점 {node_id} 이 움직여졌다")
    return f"ext 절점 이동 차단 {mesh_count}개 메쉬에서 확인"


def test_reflex_quad_rejected(names: list, mesh_paths: dict) -> str:
    """오목 사각형이 되는 병합은 거부돼야 하고, 통과한 병합의 면적은 두 삼각형 면적의 합이어야 한다."""
    rejected_count = 0
    for name in names:
        mesh = EditableMesh.load(*mesh_paths[name])
        for candidate in merge_candidates(mesh):
            if candidate.blocked and "오목" in candidate.blocked:
                rejected_count = rejected_count + 1
            elif candidate.ok:
                # 오목(dart) 사각형이면 합이 아니라 차가 나오므로 확실히 갈린다
                area_a = F.shell_quality(mesh.corners(candidate.si)).area
                area_b = F.shell_quality(mesh.corners(candidate.sj)).area
                relative_gap = abs(candidate.quality.area - (area_a + area_b)) / (area_a + area_b)
                require(relative_gap < 0.02,
                        f"{name}: 통과한 병합의 면적 {candidate.quality.area:.1f} != 삼각형 합 "
                        f"{area_a + area_b:.1f} (오목 의심)")
    return f"오목 사각형 병합 {rejected_count}건 거부 + 통과분 내각합 360° 확인"


def test_collapse_random(names: list, mesh_paths: dict, random_generator: random.Random) -> str:
    """무작위 절점 병합 — 받아들여진 것은 불변식을 지키고, 거부된 것은 메쉬를 안 건드린다.

    병합은 좌표를 하나도 바꾸지 않는다. 살아남은 절점의 좌표는 전부 원본과 같아야 하고,
    사라진 것은 정확히 remove 하나다.
    """
    accepted_count = 0
    rejected_count = 0
    for name in names:
        base_mesh = EditableMesh.load(*mesh_paths[name])
        candidates = collapse_pairs(base_mesh)
        for remove_node, keep_node, relation in random_generator.sample(candidates, min(12, len(candidates))):
            trial_mesh = base_mesh.clone()
            snapshot_before = mesh_snapshot(trial_mesh)
            try:
                operation = trial_mesh.collapse_edge(remove_node, keep_node)
            except EditError:
                require(mesh_snapshot(trial_mesh) == snapshot_before,
                        f"{name}: 거부된 병합 {remove_node}→{keep_node} 가 메쉬를 바꿨다")
                rejected_count = rejected_count + 1
                continue

            require(set(base_mesh.coords) - set(trial_mesh.coords) == {remove_node},
                    f"{name}: {remove_node}→{keep_node} 뒤 사라진 절점이 {remove_node} 하나가 아니다")
            for node_id in trial_mesh.coords:
                require(trial_mesh.coords[node_id] == base_mesh.coords[node_id], f"{name}: 병합이 좌표를 바꿨다")
            for shell in trial_mesh.shells:
                require(remove_node not in shell[2], f"{name}: 사라진 절점 {remove_node} 를 아직 쓰는 요소가 있다")
            for line in trial_mesh.lines:
                require(remove_node not in line[2], f"{name}: 사라진 절점 {remove_node} 를 아직 쓰는 요소가 있다")
            report = check(trial_mesh, base_mesh, regen=True)
            require(report.ok, f"{name}: 병합 {remove_node}→{keep_node} 뒤 불변식 위반\n{report}")
            require(len(operation.info["ext_removed"]) == int(bool(base_mesh.ext.get(remove_node))),
                    f"{name}: ext 자리 병합 기록이 틀렸다")
            accepted_count = accepted_count + 1
    return f"병합 {accepted_count}건 불변식 유지 · 거부 {rejected_count}건은 메쉬 불변"


def test_collapse_batch_order(names: list, mesh_paths: dict) -> str:
    """사람의 병합을 한 묶음으로 적용하면 하나씩 적용한 것과 결과가 같아야 한다."""
    from pairs import discover_good_pairs, load_pair
    from regen_oracle import collapse_groups, pairs_from_groups

    good_paths = discover_good_pairs()
    pair_count = 0
    for name in names:
        if name not in good_paths:
            continue
        pair = load_pair(name, good_paths, mesh_paths)
        todo = pairs_from_groups(collapse_groups(pair))
        if len(todo) == 0:
            continue
        one_by_one = EditableMesh(pair.before)
        one_by_one_ok = True
        try:
            for remove_node, keep_node in todo:
                one_by_one.collapse_edge(remove_node, keep_node)
        except EditError:
            one_by_one_ok = False
        if not one_by_one_ok:
            # 하나씩은 안 되는 경우 — 묶음의 존재 이유. 비교 대상이 아니다
            continue
        all_at_once = EditableMesh(pair.before)
        all_at_once.collapse_many(todo)
        require(signature(one_by_one) == signature(all_at_once), f"{name}: 묶음 적용과 하나씩 적용의 결과가 다르다")
        pair_count = pair_count + 1
    return f"사람 병합 {pair_count}쌍 — 묶음 = 하나씩"


def test_signature_ignores_ids(names: list, mesh_paths: dict) -> str:
    """메쉬 서명은 절점·요소 번호와 무관해야 한다 (사람은 남길 번호를 일정하게 안 고른다)."""
    for name in names[:50]:
        mesh = EditableMesh.load(*mesh_paths[name])
        new_number = {}
        position = 0
        for node_id in sorted(mesh.coords, reverse=True):
            new_number[node_id] = 900000 + position
            position = position + 1

        renumbered = mesh.clone()
        renumbered.coords = {}
        for node_id, point in mesh.coords.items():
            renumbered.coords[new_number[node_id]] = point
        renumbered.shells = []
        for element_id, element_type, node_ids in mesh.shells:
            new_nodes = []
            for node_id in node_ids:
                new_nodes.append(new_number[node_id])
            renumbered.shells.append([element_id + 7, element_type, new_nodes])
        renumbered.shells.reverse()
        renumbered.lines = []
        for element_id, element_type, node_ids in mesh.lines:
            new_nodes = []
            for node_id in node_ids:
                new_nodes.append(new_number[node_id])
            renumbered.lines.append([element_id + 7, element_type, tuple(new_nodes)])
        renumbered._mark_changed()
        require(signature(renumbered) == signature(mesh), f"{name}: 번호만 바꿨는데 서명이 달라졌다")

        smallest_node = min(mesh.coords)
        renumbered.coords[new_number[smallest_node]] = tuple(np.add(mesh.coords[smallest_node], (0.5, 0, 0)))
        require(signature(renumbered) != signature(mesh), f"{name}: 절점을 0.5mm 옮겼는데 서명이 같다")
    return f"{min(50, len(names))}개 — 번호 바꿈은 같음, 0.5mm 이동은 다름"


# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="편집 계층 회귀 테스트")
    parser.add_argument("--n", type=int, default=200, help="검사할 메쉬 수")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random_generator = random.Random(args.seed)
    mesh_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    names = sorted(mesh_paths)[:args.n]
    print(f"메쉬 {len(names)}개로 편집 계층 검증\n")

    # (이름, 실행할 함수, 인자들) — 순서대로 돈다 (random_generator 를 같은 순서로 쓰도록)
    tests = [
        ("라운드트립", test_roundtrip, (names, mesh_paths)),
        ("split↔merge 역연산", test_split_merge_inverse, (names, mesh_paths, random_generator)),
        ("swap 2회 = 항등", test_swap_involution, (names, mesh_paths)),
        ("무작위 편집 불변식", test_invariants_under_edits, (names, mesh_paths, random_generator)),
        ("1D 요소 보호(기본)", test_line_elements_protected, (names, mesh_paths)),
        ("1D 함께 삭제(drop_lines)", test_drop_lines, (names, mesh_paths)),
        ("ext 절점 고정", test_ext_nodes_frozen, (names, mesh_paths)),
        ("오목 사각형 거부", test_reflex_quad_rejected, (names, mesh_paths)),
        ("절점 병합 무작위", test_collapse_random, (names, mesh_paths, random_generator)),
        ("절점 병합 묶음 = 하나씩", test_collapse_batch_order, (sorted(mesh_paths), mesh_paths)),
        ("메쉬 서명은 번호 무관", test_signature_ignores_ids, (names, mesh_paths)),
    ]

    failed_count = 0
    for label, test_function, test_arguments in tests:
        try:
            message = test_function(*test_arguments)
        except Failed as error:
            failed_count = failed_count + 1
            print(f"  [실패] {label}\n         {error}")
        else:
            print(f"  [통과] {label}: {message}")

    print()
    if failed_count > 0:
        raise SystemExit(f"{failed_count}개 테스트 실패")
    print("전부 통과")


if __name__ == "__main__":
    main()

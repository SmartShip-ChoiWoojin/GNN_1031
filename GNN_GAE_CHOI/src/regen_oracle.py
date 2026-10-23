"""사람의 수정을 '우리 연산의 순서' 로 다시 쓴다 — 재생성 모델의 정답과 상한.

재생성은 메쉬를 통째로 새로 그리지 않는다. 사람이 한 일을 분해하면 거의 전부가 두
연산이다 (실측):

    1단계  절점 병합 (collapse_edge)   사라진 절점 481개 중 451개
    2단계  삼각형 쌍 병합 (merge_tris)  '연결만 변경' 196건의 97%, 병합 뒤 남은 차이의 대부분

그래서 모델이 할 일은 "어느 변을 어느 쪽으로 합칠지" 와 "그 뒤 어느 삼각형 쌍을 합칠지"
를 고르는 것이다. 이 모듈은 사람의 결과로부터 그 정답 순서(대본, script)를 뽑고, **실제
연산 코드로 다시 돌려서** 사람 결과가 정확히 재현되는지 검사한다. 재현되는 비율이 곧 이
연산 체계로 도달할 수 있는 상한이다.

정답 대본을 만드는 규칙:
  - 절점 병합은 **그룹** 단위로 본다. 살아남은 절점과 그가 흡수한 절점들이 한 그룹이고,
    최종 좌표에 원래 서 있던 그룹원이 '자리가 남는 절점(keep)' 이다. 셋이 하나로 모인
    `_0021` 처럼 남은 번호의 자리가 아닌 곳으로 모이는 경우가 있어서, 쌍 단위로 방향을
    정하면 틀린다.
  - 병합 순서: 지금 이웃인 것부터. 다른 병합이 먼저 일어나야 이웃이 되는 경우가 있다.
  - 삼각형 병합: 병합 뒤 메쉬와 사람 결과를 **좌표**로 비교해, 사람 결과에만 있는
    사각형을 이루는 삼각형 두 개를 찾는다.

    python src/regen_oracle.py              # 508쌍 전부 — 상한과 실패 사유
    python src/regen_oracle.py --name SN2592_FWD_Clean_0021
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import gae_paths
from bdf_io import discover_pairs
from meshdiff import pkey, signature
from meshedit import EditableMesh, EditError, collapse_greedy, shared_edge_of
from pairs import all_pairs, discover_good_pairs, load_pair


class Script:
    """쌍 하나의 정답 대본과 재현 결과."""

    def __init__(self, name: str):
        self.name = name
        #: [(remove, keep)] — 원본 절점 번호, 적용 순서대로
        self.collapses = []
        #: 병합 뒤 메쉬에서 합칠 삼각형 쌍의 공유 변 (절점 번호), 적용 순서대로
        self.merges = []
        #: 연산이 거부된 것 (사유 포함) — 사람이 한 일인데 우리 연산이 못 따라간 것
        self.failed = []
        #: 이 연산 체계 밖의 수정 (절점 추가 / 순수 삭제 / 좌표 이동 / 신규 교체)
        self.outside = []
        #: 요소·1D·좌표까지 사람 결과와 같은가
        self.exact = False
        #: 쉘만 같은가
        self.exact_shells = False
        #: 재현 뒤에도 남은 차이 (요소 수) — 우리만 지운 것 / 사람만 만든 것
        self.res_rm = 0
        self.res_cr = 0
        #: 1D 가 안 맞는 수 (요소는 다 맞았는데)
        self.line_mismatch = 0

    @property
    def reason(self) -> str:
        """재현 결과를 한 마디로."""
        if self.exact:
            return "재현됨"
        if len(self.outside) > 0:
            return self.outside[0]
        if len(self.failed) > 0:
            return "연산 거부"
        if self.exact_shells:
            return "1D 불일치"
        return "재배치(연산 밖)"


# ──────────────────────────────────────────────────────────────────────────
def collapse_groups(pair) -> list:
    """[(자리가 남는 절점, [합쳐질 절점들])] — 사람 결과의 좌표로 방향을 정한다.

    그룹 = 살아남은 절점 + 그가 흡수한 절점들. 그룹원 중 사람 결과의 최종 좌표에 가장
    가까운 자리에 원래 서 있던 절점이 keep 이다 (거리가 같으면 번호가 작은 쪽).
    """
    before = pair.before
    after = pair.after
    groups = []
    for survivor_node, absorbed_list in pair.nodes.survivors.items():
        if survivor_node not in after.coords:
            continue
        group_members = [survivor_node] + list(absorbed_list)
        final_point = np.asarray(after.coords[survivor_node], dtype=float)

        distance_of = {}
        for member in group_members:
            member_point = np.asarray(before.coords[member], float)
            distance_of[member] = float(np.linalg.norm(member_point - final_point))

        keep_node = None
        for member in group_members:
            if keep_node is None:
                keep_node = member
            elif (distance_of[member], member) < (distance_of[keep_node], keep_node):
                keep_node = member

        remove_nodes = []
        for member in group_members:
            if member != keep_node:
                remove_nodes.append(member)
        groups.append((keep_node, remove_nodes))
    return groups


def pairs_from_groups(groups: list) -> list:
    """collapse_groups 결과를 [(remove, keep)] 목록으로 펼친다."""
    collapse_list = []
    for keep_node, remove_nodes in groups:
        for remove_node in remove_nodes:
            collapse_list.append((remove_node, keep_node))
    return collapse_list


def apply_collapses(mesh: EditableMesh, todo: list, failed: list | None = None) -> list:
    """(remove, keep) 들을 한 묶음으로 적용한다. 적용한 것을 순서대로 돌려준다.

    원본에서 이미 이웃인 쌍을 앞에 둔다 — 두 칸 떨어진 쌍(7건)은 앞의 병합으로 이웃이
    된 뒤에야 합칠 수 있다. 묶음이 통째로 거부되면 ``collapse_greedy`` 가 되는 만큼만 남긴다.
    """
    neighbors_now = []
    not_neighbors_now = []
    for remove_node, keep_node in todo:
        if mesh.collapse_relation(remove_node, keep_node) is None:
            not_neighbors_now.append((remove_node, keep_node))
        else:
            neighbors_now.append((remove_node, keep_node))
    ordered = neighbors_now + not_neighbors_now

    applied, rejected = collapse_greedy(mesh, ordered)
    if failed is not None:
        for (remove_node, keep_node), message in rejected.items():
            failed.append(f"병합 {remove_node}→{keep_node}: {message}")
    return applied


def tri_pairs(mesh: EditableMesh) -> list:
    """변을 공유하는 삼각형 쌍 (막힌 것 포함, 유효성은 보지 않는다)."""
    rep = mesh.rep
    result = []
    for edge, shell_indices in rep.edge_shells.items():
        if len(shell_indices) != 2:
            continue
        if len(mesh.shells[shell_indices[0]][2]) != 3:
            continue
        if len(mesh.shells[shell_indices[1]][2]) != 3:
            continue
        result.append((shell_indices[0], shell_indices[1]))
    return result


def oracle_merges(mesh: EditableMesh, good, failed: list | None = None) -> list:
    """사람 결과에만 있는 사각형을 이루는 삼각형 쌍을 찾아 합친다. 공유 변 목록을 돌려준다."""
    good_shell_keys = set()
    for shell in good.shells:
        position_keys = []
        for node_id in shell[2]:
            position_keys.append(pkey(good.coords[node_id]))
        good_shell_keys.add(frozenset(position_keys))

    merged_edges = []
    tried_edges = set()
    while True:
        found = None
        for shell_index_a, shell_index_b in tri_pairs(mesh):
            union_nodes = set(mesh.shells[shell_index_a][2]) | set(mesh.shells[shell_index_b][2])
            position_keys = []
            for node_id in union_nodes:
                position_keys.append(pkey(mesh.coords[node_id]))
            union_key = frozenset(position_keys)
            if len(union_key) != 4:
                continue
            if union_key not in good_shell_keys:
                continue
            edge = shared_edge_of(mesh, shell_index_a, shell_index_b)
            if edge in tried_edges:
                continue
            found = (shell_index_a, shell_index_b, edge)
            break

        if found is None:
            return merged_edges

        shell_index_a, shell_index_b, edge = found
        tried_edges.add(edge)
        try:
            mesh.merge_tris(shell_index_a, shell_index_b, drop_lines=True)
        except EditError as error:
            if failed is not None:
                failed.append(f"삼각형 병합 {sorted(edge)}: {error}")
            continue
        merged_edges.append(edge)


def build_script(pair):
    """(대본, 1단계 뒤 메쉬, 최종 메쉬). 메쉬들은 화면·학습에서 그대로 쓴다."""
    script = Script(pair.name)
    node_match = pair.nodes

    added_not_replacing = node_match.added - set(node_match.replaced_by_new.values())
    if len(added_not_replacing) > 0:
        script.outside.append("절점 추가")
    if len(node_match.replaced_by_new) > 0:
        script.outside.append("신규 절점으로 교체")
    if len(node_match.deleted) > 0:
        script.outside.append("순수 삭제")
    if len(node_match.moved) > 0:
        script.outside.append("좌표 이동")

    mesh = EditableMesh(pair.before)
    todo = pairs_from_groups(collapse_groups(pair))
    script.collapses = apply_collapses(mesh, todo, script.failed)
    mesh_after_stage1 = mesh.clone()
    script.merges = oracle_merges(mesh, pair.after, script.failed)

    predicted_signature = signature(mesh)
    good_signature = signature(pair.after)
    script.exact = predicted_signature == good_signature
    script.exact_shells = predicted_signature.shells == good_signature.shells
    script.res_rm = len(predicted_signature.shells - good_signature.shells)
    script.res_cr = len(good_signature.shells - predicted_signature.shells)
    script.line_mismatch = len(predicted_signature.lines ^ good_signature.lines)
    return script, mesh_after_stage1, mesh


def build_all(names=None, verbose: bool = False) -> list:
    """쌍 여러 개의 대본. names 를 안 주면 508쌍 전부."""
    good_paths = discover_good_pairs()
    original_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    if names is None:
        names = all_pairs()
    scripts = []
    count = 0
    for name in names:
        count = count + 1
        script, _, _ = build_script(load_pair(name, good_paths, original_paths))
        scripts.append(script)
        if verbose and count % 100 == 0:
            print(f"  ... {count}")
    return scripts


def collapses_text(collapses: list) -> str:
    """[(remove, keep)] → "r>k;r>k" (csv 한 칸)."""
    parts = []
    for remove_node, keep_node in collapses:
        parts.append(f"{remove_node}>{keep_node}")
    return ";".join(parts)


def edges_text(edges: list) -> str:
    """[공유 변] → "a-b;a-b" (csv 한 칸, 각 변은 번호 순)."""
    parts = []
    for edge in edges:
        node_texts = []
        for node_id in sorted(edge):
            node_texts.append(str(node_id))
        parts.append("-".join(node_texts))
    return ";".join(parts)


def parse_collapses(text: str) -> list:
    """collapses_text 의 반대: "r>k;r>k" → [(r, k), ...]."""
    result = []
    for part in str(text).split(";"):
        if part == "":
            continue
        remove_text, keep_text = part.split(">")
        result.append((int(remove_text), int(keep_text)))
    return result


def parse_edges(text: str) -> list:
    """edges_text 의 반대: "a-b;a-b" → [frozenset({a, b}), ...]."""
    result = []
    for part in str(text).split(";"):
        if part == "":
            continue
        node_ids = []
        for node_text in part.split("-"):
            node_ids.append(int(node_text))
        result.append(frozenset(node_ids))
    return result


def to_frame(scripts: list):
    """대본 목록 → pandas 표."""
    import pandas as pd

    rows = []
    for script in scripts:
        rows.append(dict(
            name=script.name, exact=script.exact, exact_shells=script.exact_shells,
            reason=script.reason,
            n_collapse=len(script.collapses), n_merge=len(script.merges), n_failed=len(script.failed),
            res_rm=script.res_rm, res_cr=script.res_cr, line_mismatch=script.line_mismatch,
            outside=";".join(script.outside), failed=" | ".join(script.failed),
            collapses=collapses_text(script.collapses),
            merges=edges_text(script.merges),
        ))
    return pd.DataFrame(rows)


def count_in_order(values: list) -> list:
    """[(값, 개수)] — 개수가 많은 순. 개수가 같으면 처음 나온 순 (Counter.most_common 과 같다)."""
    counts = {}
    for value in values:
        if value not in counts:
            counts[value] = 0
        counts[value] = counts[value] + 1
    items = list(counts.items())

    def count_of(item):
        return item[1]

    return sorted(items, key=count_of, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="사람 수정 → 연산 대본, 재현 상한")
    parser.add_argument("--name", help="한 쌍만")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.name:
        script = build_all([args.name])[0]
        print(f"{script.name}: {script.reason}")
        print(f"  절점 병합 {len(script.collapses)}: {script.collapses}")
        merge_list = []
        for edge in script.merges:
            merge_list.append(sorted(edge))
        print(f"  삼각형 병합 {len(script.merges)}: {merge_list}")
        for failure in script.failed:
            print(f"  거부: {failure}")
        print(f"  남은 차이: 요소 -{script.res_rm} +{script.res_cr}, 1D {script.line_mismatch}")
        return

    print("508쌍의 정답 대본을 만들고 재현하는 중...")
    scripts = build_all(verbose=True)
    table = to_frame(scripts)
    pair_count = len(table)
    print(f"\n정확히 재현 (요소+1D+좌표): {int(table.exact.sum())}/{pair_count} ({table.exact.mean():.1%})")
    print(f"요소만 재현:               {int(table.exact_shells.sum())}/{pair_count} ({table.exact_shells.mean():.1%})")

    print("\n재현 못 한 사유:")
    for reason, count in count_in_order(list(table.reason)):
        print(f"  {reason:16s} {count}")
    print(f"\n연산 수 — 절점 병합 {int(table.n_collapse.sum())} / 삼각형 병합 {int(table.n_merge.sum())} / "
          f"거부 {int(table.n_failed.sum())}")

    all_failures = []
    for script in scripts:
        for failure in script.failed:
            all_failures.append(failure)
    if len(all_failures) > 0:
        print("\n거부된 연산 (사람은 했는데 우리 연산이 막은 것):")
        short_reasons = []
        for failure in all_failures:
            # "병합 1→2: 사유 (자세히)" 에서 '사유' 만 뽑는다
            after_colon = failure.split(": ", 1)[1]
            short_reasons.append(after_colon.split(" (")[0])
        for reason, count in count_in_order(short_reasons)[:10]:
            print(f"  {count:4d}  {reason}")

    gae_paths.ensure_dirs()
    out_path = os.path.join(gae_paths.OUT_DIR, "regen_oracle.csv")
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()

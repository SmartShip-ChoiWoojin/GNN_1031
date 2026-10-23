"""디버거로 '쌍 읽기 → 절점 대응 → 정답 대본 → 재현 검사' 흐름을 따라가는 스크립트.

VSCode 에서 'GNN_GAE_CHOI 쌍·정답 대본 디버그' (F5) 로 실행하거나 터미널에서:

    python src/debug_pair.py --name SN2592_FWD_Clean_0021

각 단계에서 F11(Step Into) 로 들어가면 실제 함수로 이어진다. regen_oracle.build_script 가
하는 일을 한 줄씩 풀어 써 두었다.
  0. 원본(.bdf, 쉼표 구분)과 사람이 고친 것(.bdf, 고정폭)을 읽는다        bdf_io / bdf_fixed
  1. 절점 대응 — 사라진 절점이 누구에게 흡수됐나 (연결관계로 판정)       pairs.match_nodes → nodemap
  2. 정답 절점 병합 그룹 — 어느 자리가 남나 (사람 결과 좌표로 판정)        regen_oracle.collapse_groups
  3. 병합 적용 (지금 이웃인 것부터, 안 되면 되는 만큼)                    meshedit.collapse_greedy
  4. 사람 결과에만 있는 사각형을 이루는 삼각형 쌍을 찾아 합친다          regen_oracle.oracle_merges
  5. 사람 결과와 좌표 기준으로 비교                                     meshdiff.signature
"""

from __future__ import annotations

import argparse

import gae_paths
from bdf_fixed import load_fixed_mesh
from bdf_io import discover_pairs, load_raw_mesh
from meshdiff import signature
from meshedit import EditableMesh, check
from pairs import discover_good_pairs, match_nodes
from regen_oracle import apply_collapses, collapse_groups, oracle_merges, pairs_from_groups


class SimplePair:
    """pairs.Pair 와 같은 모양 (name, before, after, nodes)."""

    def __init__(self, name, before, after, nodes):
        self.name = name
        self.before = before
        self.after = after
        self.nodes = nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="쌍·정답 대본 흐름 디버그 (메쉬 하나)")
    parser.add_argument("--name", default="SN2592_FWD_Clean_0021", help="메쉬 이름")
    args = parser.parse_args()

    # ── 0. 읽기 ──────────────────────────────────────────────────────
    good_paths = discover_good_pairs()
    original_paths = discover_pairs(gae_paths.FE_DIR)[0]
    original_bdf, original_json = original_paths[args.name]
    good_bdf, good_json = good_paths[args.name]
    before = load_raw_mesh(original_bdf, original_json)      # 원본 (쉼표 구분 .bdf)
    after = load_fixed_mesh(good_bdf, good_json)              # 사람이 고친 것 (고정폭 .bdf)
    print(f"[0] 원본  node {len(before.coords)} / shell {len(before.shells)} / 1D {len(before.lines)}")
    print(f"    수정본 node {len(after.coords)} / shell {len(after.shells)} / 1D {len(after.lines)}")

    # ── 1. 절점 대응 ─────────────────────────────────────────────────
    node_match = match_nodes(before, after)
    print(f"[1] {node_match.summary()}")
    for absorbed_node, survivor_node in node_match.absorbed.items():
        print(f"    흡수: {absorbed_node} → {survivor_node} (근거 {node_match.basis.get(absorbed_node)})")
    print(f"    검산 {node_match.explained:.2f}")
    pair = SimplePair(args.name, before, after, node_match)

    # ── 2. 정답 병합 그룹 ────────────────────────────────────────────
    groups = collapse_groups(pair)
    for keep_node, remove_nodes in groups:
        print(f"[2] 그룹: 자리가 남는 {keep_node} ← 합쳐질 {remove_nodes}")
    todo = pairs_from_groups(groups)

    # ── 3. 병합 적용 ─────────────────────────────────────────────────
    mesh = EditableMesh(before)
    failed = []
    applied = apply_collapses(mesh, todo, failed)
    print(f"[3] 절점 병합 적용 {len(applied)}개: {applied}")
    for failure in failed:
        print(f"    거부: {failure}")

    # ── 4. 삼각형 쌍 병합 ────────────────────────────────────────────
    merged_edges = oracle_merges(mesh, after, failed)
    merged_list = []
    for edge in merged_edges:
        merged_list.append(sorted(edge))
    print(f"[4] 삼각형 쌍 병합 {len(merged_edges)}개: {merged_list}")

    # ── 5. 비교 ──────────────────────────────────────────────────────
    result_signature = signature(mesh)
    human_signature = signature(after)
    print(f"[5] 사람 결과와 정확히 일치: {result_signature == human_signature}")
    print(f"    우리만 가진 요소 {len(result_signature.shells - human_signature.shells)}개 / "
          f"사람만 가진 요소 {len(human_signature.shells - result_signature.shells)}개 / "
          f"1D 차이 {len(result_signature.lines ^ human_signature.lines)}개")
    print(f"[검사] {check(mesh, EditableMesh(before), regen=True)}")


if __name__ == "__main__":
    main()

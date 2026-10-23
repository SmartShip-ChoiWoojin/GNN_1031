"""디버거로 재생성 흐름을 따라가는 스크립트 — 메쉬 하나를 최종 모델로 다시 만든다.

VSCode 에서 'GNN_GAE_CHOI 재생성 디버그' (F5) 로 실행하거나 터미널에서:

    python src/debug_regen.py --name SN2592_FWD_Clean_0199 --single

아래 각 단계에서 F11(Step Into) 로 들어가면 실제 함수로 이어진다. 채점용 코드
(regen_model.RegenModel.regenerate)와 같은 함수를 부르되, 단계를 한 줄씩 풀어 써 두었다.
1단계는 decode_collapses 를 부르지 않고 그 안의 반복을 여기에 그대로 펼쳐 두었다 —
매 반복에서 후보 수, 1위 후보, 점수, 멈춤 점수를 눈으로 볼 수 있다.

--single : 배포 모델은 fold 5 × seed 3 = 15개 모델의 평균이라 CandidateNet.forward 가
           매번 15번 불린다. 디버깅할 때는 첫 번째 모델 하나만 써서 한 번씩만 들어가게 한다.
           (결과는 최종 모델과 다를 수 있다)
--cpu    : GPU 대신 CPU 로
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import gae_paths
from bdf_io import discover_pairs
from meshdiff import compare
from meshedit import REGEN_AREA_TOL, EditableMesh, apply_collapse_seq, apply_merges, check
from pairs import discover_good_pairs, load_pair
from regen_oracle import edges_text
from regen_model import MAX_COLLAPSES, RegenModel, pick_merges, stage1_item, total_area


def main() -> None:
    parser = argparse.ArgumentParser(description="재생성 흐름 디버그 (메쉬 하나)")
    parser.add_argument("--run", default="runs/regen_final", help="학습된 모델 폴더")
    parser.add_argument("--name", default="SN2592_FWD_Clean_0199", help="메쉬 이름")
    parser.add_argument("--single", action="store_true", help="15개 평균 대신 첫 번째 모델 하나만")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if os.path.isabs(args.run):
        run_dir = args.run
    else:
        run_dir = os.path.join(gae_paths.ROOT, args.run)

    # ── 0. 모델과 메쉬 불러오기 ────────────────────────────────────────
    # stage1_scorer = GNN 15개 평균, stage2_scorer = GBM 15개 평균
    model = RegenModel.load(run_dir, device)
    if args.single:
        model.stage1_scorer = model.stage1_scorer.members[0]
        model.stage2_scorer = model.stage2_scorer.members[0]

    good_paths = discover_good_pairs()
    original_paths = discover_pairs(gae_paths.FE_DIR)[0]
    pair = load_pair(args.name, good_paths, original_paths)
    bad_mesh = EditableMesh(pair.before)       # 원본 (비정상 메쉬)
    human_mesh = EditableMesh(pair.after)      # 사람 수정 결과
    print(f"[0] {args.name}: node {len(bad_mesh.coords)}, shell {len(bad_mesh.shells)}, 1D {len(bad_mesh.lines)}")
    print(f"    1단계 임계값(여유 하한) {model.stage1_threshold:.2f} · 2단계 임계값(확률) {model.stage2_threshold:.2f}")

    # ── 1단계: edge collapse 반복 (regen_model.decode_collapses 를 펼친 것) ──
    current_mesh = bad_mesh.clone()
    current_mesh.log.clear()
    original_area = total_area(bad_mesh)
    chosen_collapses = []
    for step in range(MAX_COLLAPSES):
        # 후보 만들기: collapse_candidates(후보·피처) → mesh_graph(그래프)
        item = stage1_item(bad_mesh.name, current_mesh, model.normalizer, model.graph_options,
                           use_context=model.stage1_scorer.config.use_context)
        if item.candidate_count == 0:
            print(f"[1-{step + 1}] 적용 가능한 후보 없음 → 멈춤")
            break

        # 점수 매기기: GNN (CandidateNet.forward → 인코더 encode)
        candidate_scores, stop_score = model.stage1_scorer.scores([item])[0]
        order = np.argsort(-candidate_scores, kind="stable")
        best_index = order[0]
        print(f"[1-{step + 1}] 후보 {item.candidate_count}개 · 1위 {item.candidates.pairs[best_index]} "
              f"점수 {float(candidate_scores[best_index]):.2f} · 멈춤 점수 {stop_score:.2f}")

        picked = None
        for candidate_index in order:
            margin = float(candidate_scores[candidate_index] - stop_score)
            if margin < model.stage1_threshold:
                break
            remove_node, keep_node = item.candidates.pairs[candidate_index]
            trial_mesh = current_mesh.clone()
            trial_mesh.collapse_edge(remove_node, keep_node)
            area_change = abs(total_area(trial_mesh) - original_area) / original_area
            if area_change > REGEN_AREA_TOL:
                print(f"      {remove_node}→{keep_node} 은 누적 면적 변화 {area_change:.1%} 라 건너뜀")
                continue
            picked = ((remove_node, keep_node), margin, trial_mesh)
            break

        if picked is None:
            print(f"      여유가 임계값 {model.stage1_threshold:.2f} 보다 작음 → 멈춤")
            break
        chosen_pair, margin, current_mesh = picked
        chosen_collapses.append(chosen_pair)
        print(f"      적용: {chosen_pair[0]} → {chosen_pair[1]} (여유 {margin:.2f})")

    # 고른 순서대로 원본에 다시 적용 (채점 코드와 같은 방식)
    mesh_after_stage1 = apply_collapse_seq(bad_mesh, chosen_collapses)
    print(f"[1] edge collapse {len(chosen_collapses)}개: {chosen_collapses}")

    # ── 2단계: triangle-pair merge ────────────────────────────────────
    # merge_cands(후보) → GBM 확률 → 확률 ≥ 임계값, 같은 삼각형 중복 없이 → 공유 변으로 찾아 merge_tris
    stage2_candidates, stage2_probabilities = model.stage2(bad_mesh.name, mesh_after_stage1)
    for candidate_index in range(len(stage2_candidates.pairs)):
        edge_nodes = sorted(stage2_candidates.edges[candidate_index])
        print(f"[2] 후보 공유 변 {edge_nodes} 확률 {float(stage2_probabilities[candidate_index]):.3f}")
    picked_edges = pick_merges(stage2_candidates, stage2_probabilities, model.stage2_threshold)
    final_mesh = mesh_after_stage1.clone()
    merged_edges, notes = apply_merges(final_mesh, picked_edges)
    print(f"[2] triangle-pair merge 후보 {len(stage2_candidates.pairs)}개 중 {len(merged_edges)}개 적용: "
          + edges_text(merged_edges))
    for note in notes:
        print(f"    못 합침: {note}")

    # ── 결과 확인 ─────────────────────────────────────────────────────
    result = compare(bad_mesh, final_mesh, human_mesh)
    print(f"[결과] 사람과 정확히 일치: {result['exact']} · 바뀐 요소 IoU {result['change_iou']:.2f}")
    print(f"[검사] {check(final_mesh, bad_mesh, regen=True)}")


if __name__ == "__main__":
    main()

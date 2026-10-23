"""디버거로 '학습 → 임계값별 채점' 흐름을 따라가는 스크립트 — 작은 규모로 빨리 돈다.

VSCode 에서 'GNN_GAE_CHOI 학습 디버그' (F5) 로 실행하거나 터미널에서:

    python src/debug_train.py                     # 학습 셋 앞 40쌍, GNN 1단계 2 epoch
    python src/debug_train.py --n 80 --epochs 5

regen_train.main 이 하는 일을 fold 하나만, 한 줄씩 풀어 써 두었다. 결과 숫자는 의미가 없다
(데이터도 epoch 도 적다). 흐름을 눈으로 따라가는 용도다. 아무 파일도 저장하지 않는다.
  0. 분할 파일에서 학습 셋 이름을 읽고 표본을 만든다                   regen_data.load_samples
  1. 중간 상태(사람의 병합을 일부 적용한 메쉬)를 만든다                regen_train.make_states
  2. 1단계·2단계 채점 단위(후보 + 피처 + 그래프 + 정답)를 만든다       regen_train.build_items
  3. 1단계 GNN 채점기, 2단계 GBM 채점기를 학습한다                    GNNScorer.fit / GBMScorer.fit
  4. 남겨 둔 쌍을 모든 임계값 조합으로 재생성해 사람과 비교한다         regen_train.grid_eval
"""

from __future__ import annotations

import argparse

import torch

from regen_data import load_samples, load_split
from regen_model import GBMScorer, GNNScorer, TrainConfig, load_encoder
from regen_train import build_items, grid_eval, make_states, mean_of


def main() -> None:
    parser = argparse.ArgumentParser(description="학습 흐름 디버그 (작은 규모)")
    parser.add_argument("--n", type=int, default=40, help="학습 셋 앞에서 쓸 쌍 수")
    parser.add_argument("--epochs", type=int, default=2, help="1단계 GNN epoch")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ── 0. 표본 ──────────────────────────────────────────────────────
    split_table = load_split()
    names = sorted(split_table[split_table.split == "train"].name)[:args.n]
    samples = load_samples(names, verbose=False)
    print(f"[0] 표본 {len(samples)}쌍")
    first_sample = samples[0]
    print(f"    예: {first_sample.name} · 정답 절점 병합 {first_sample.collapses} · "
          f"정답 삼각형 병합 {len(first_sample.merges)}개 · 정답 대본 재현 {first_sample.oracle_reason}")

    # ── 1. 중간 상태 (한 쌍만 보기) ──────────────────────────────────
    states = make_states(first_sample)
    for cache_key, mesh, remaining in states:
        print(f"[1] 상태 {cache_key}: 남은 정답 병합 {remaining}")

    # ── 2. 채점 단위 ─────────────────────────────────────────────────
    _, normalizer, graph_options, _ = load_encoder(TrainConfig().encoder_run, device)
    stage1_items, stage2_items, first_candidates = build_items(samples, normalizer, graph_options, False,
                                                               use_states=True)
    stage1_unit_count = 0
    for items_of_mesh in stage1_items:
        stage1_unit_count = stage1_unit_count + len(items_of_mesh)
    print(f"[2] 1단계 단위 {stage1_unit_count}개 (원본 + 중간 상태) · 2단계 단위 {len(stage2_items)}개")
    first_item = stage1_items[0][0]
    print(f"    예: {first_item.name} 후보 {first_item.candidate_count}개, 피처 {first_item.candidates.feats.shape}, "
          f"정답 {int(first_item.labels.sum())}개")

    # ── 3. 학습 — 뒤 20% 를 채점용으로 남긴다 ─────────────────────────
    scoring_count = max(1, len(samples) // 5)
    fitting_stage1_items = []
    fitting_stage2_items = []
    for sample_index in range(len(samples) - scoring_count):
        for item in stage1_items[sample_index]:
            fitting_stage1_items.append(item)
        fitting_stage2_items.append(stage2_items[sample_index])
    stage1_config = TrainConfig(epochs=args.epochs, loss="list")
    stage2_config = TrainConfig(epochs=args.epochs)
    stage1_scorer = GNNScorer(1, stage1_config, device).fit(fitting_stage1_items, print)
    stage2_scorer = GBMScorer(2, stage2_config, device).fit(fitting_stage2_items, print)
    print(f"[3] 학습 끝 (1단계 단위 {len(fitting_stage1_items)}개, 2단계 단위 {len(fitting_stage2_items)}개)")

    # ── 4. 임계값 조합별 채점 ────────────────────────────────────────
    scoring_samples = samples[len(samples) - scoring_count:]
    results = grid_eval(scoring_samples, first_candidates, stage1_scorer, stage2_scorer, normalizer, graph_options,
                        False, "iter")
    best_pair = None
    best_exact = None
    for threshold_pair, result_list in results.items():
        exact_rate = mean_of(result_list, "exact")
        if best_exact is None or exact_rate > best_exact:
            best_pair = threshold_pair
            best_exact = exact_rate
    print(f"[4] 남긴 {scoring_count}쌍 · 최고 정확일치 {best_exact:.1%} @ 임계값 {best_pair}")


if __name__ == "__main__":
    main()

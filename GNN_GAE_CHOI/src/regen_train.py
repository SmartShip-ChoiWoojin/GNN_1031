"""재생성 모델 학습 — 학습 셋(90%) 안에서만 모든 결정을 내린다.

    1. 학습 셋을 5-fold 로 나눠, 각 메쉬를 **자기를 학습에 안 쓴 모델**로 재생성한다
       (out-of-fold). fold 는 분할 파일에 묶음 단위로 미리 적혀 있다. 이걸로
         - 임계값 두 개(1단계·2단계)를 고르고
         - 학습 셋 안에서의 정직한 성능(CV)을 잰다.
    2. 배포 모델은 **그 fold 모델들의 평균**이다 (``--deploy ensemble``, 기본). 학습 셋
       전부로 다시 학습하면(``--deploy refit``) 점수 눈금이 달라져 CV 에서 고른 임계값이
       어긋난다 — 실제로 그래서 테스트에서 병합을 너무 적게 골랐다 (regen_nested.py 로 확인).
    3. 테스트 셋은 여기서 **한 번도 열지 않는다** → `regen_eval.py` 가 처음으로 연다.

1단계 선택 방식 (``--decode``):
  iter     '다음에 합칠 병합 하나 또는 멈춤' 을 반복 (기본). 학습에는 원본 말고도
           **사람의 병합을 일부 적용한 중간 상태**를 함께 쓴다 — 남은 병합이 정답이고,
           다 적용한 상태의 정답은 '멈춤' 이다.
  oneshot  원본에서 후보 전부에 한 번에 점수를 매겨 임계값 이상을 묶어 적용.

2단계는 학습할 때 **정답 1단계를 적용한 메쉬**로 배운다 (teacher forcing). 쓸 때는 모델이
만든 1단계 결과 위에서 돈다 — 그 차이까지 포함해 채점되도록 CV 평가는 실제 파이프라인을
그대로 돌린다.

    python src/regen_train.py --tag final --loss list --scorer2 gbm --seeds 0,1,2   # 최종 설정 (~25분)
    python src/regen_train.py --tag gbm_oneshot --scorer gbm --decode oneshot      # 피처만 쓰는 기준선 (빠름)
"""

from __future__ import annotations

import argparse
import os
import time
import zlib

import numpy as np
import pandas as pd
import torch

import gae_paths
from meshdiff import compare
from meshedit import EditableMesh, EditError, apply_collapse_seq, apply_merges, collapse_greedy
from regen_data import load_samples, load_split
from regen_model import (SCORERS, EnsembleScorer, RegenModel, TrainConfig, cached_collapse_cands,
                         decode_collapses, load_encoder, pick_collapses, pick_merges, stage1_item, stage2_item)
from regen_oracle import collapses_text, edges_text


def make_grid(start: float, stop: float, step: float) -> list:
    """np.arange(start, stop, step) 를 소수 둘째 자리로 반올림한 목록."""
    grid = []
    for value in np.arange(start, stop, step):
        grid.append(round(float(value), 2))
    return grid


#: oneshot 의 1단계 임계값 = 확률
GRID1 = make_grid(0.15, 0.96, 0.05)
#: iter 의 1단계 임계값 = 여유(최고 후보 점수 - 멈춤 점수)
GRID1_ITER = make_grid(-2.0, 4.01, 0.25)
#: 2단계 임계값 = 확률
GRID2 = make_grid(0.05, 0.91, 0.05)


def ops_str(collapses: list) -> str:
    """적용한 절점 병합을 csv 한 칸으로 — 화면이 이 기록만으로 결과를 재현한다."""
    return collapses_text(collapses)


def edges_str(edges: list) -> str:
    return edges_text(edges)


# ──────────────────────────────────────────────────────────────────────────
# 학습 표본
# ──────────────────────────────────────────────────────────────────────────
def make_states(sample, n_orders: int = 2) -> list:
    """[(캐시 키, 메쉬, 남은 정답 병합)] — 원본 + 사람의 병합을 일부 적용한 중간 상태.

    사람이 병합한 순서는 알 수 없으므로 무작위 순서 ``n_orders`` 개를 따라가며 앞부분을
    적용한 상태를 모은다 (같은 집합이면 한 번만 — 적용 순서가 달라도 메쉬는 같다).
    다 적용한 상태는 남은 정답이 없으므로 '멈춤' 을 가르친다.
    """
    true_collapses = []
    for pair in sample.collapses:
        true_collapses.append(tuple(pair))
    bad_mesh = EditableMesh(sample.pair.before)
    states = [(sample.name, bad_mesh, true_collapses)]
    if len(true_collapses) == 0:
        return states

    # hash() 는 실행마다 바뀐다 — 재현되도록 이름의 crc 를 씨앗으로 쓴다
    random_generator = np.random.default_rng(zlib.crc32(sample.name.encode()))
    seen_keys = {()}
    for order_number in range(n_orders):
        mesh = bad_mesh.clone()
        applied = []
        for collapse_index in random_generator.permutation(len(true_collapses)):
            remove_node, keep_node = true_collapses[collapse_index]
            try:
                mesh.collapse_edge(remove_node, keep_node)
            except EditError:
                continue
            applied.append(true_collapses[collapse_index])
            state_key = tuple(sorted(applied))
            if state_key in seen_keys:
                continue
            seen_keys.add(state_key)

            tag_parts = []
            for applied_remove, applied_keep in state_key:
                tag_parts.append(f"{applied_remove}>{applied_keep}")
            cache_key = f"{sample.name}|" + ".".join(tag_parts)

            remaining = []
            for pair in true_collapses:
                if pair not in applied:
                    remaining.append(pair)
            states.append((cache_key, mesh.clone(), remaining))
    return states


def build_items(samples: list, normalizer, graph_options: dict, use_context: bool, use_states: bool,
                verbose: bool = True):
    """(1단계 표본 [메쉬별 목록], 2단계 표본, {이름: 원본의 1단계 후보}).

    use_states=True (반복형) 면 1단계는 메쉬마다 중간 상태 여러 개를 쓴다.
    """
    states_per_mesh = []
    for sample in samples:
        if use_states:
            states_per_mesh.append(make_states(sample))
        else:
            states_per_mesh.append([(sample.name, EditableMesh(sample.pair.before), list(sample.collapses))])

    all_keys_and_meshes = []
    for states in states_per_mesh:
        for cache_key, mesh, remaining in states:
            all_keys_and_meshes.append((cache_key, mesh))
    candidates_of_key = cached_collapse_cands(all_keys_and_meshes, verbose=verbose)

    stage1_items = []
    for states in states_per_mesh:
        items_of_mesh = []
        for cache_key, mesh, remaining in states:
            item = stage1_item(cache_key, mesh, normalizer, graph_options, candidates_of_key[cache_key], remaining,
                               use_context=use_context)
            # 남은 정답이 있는데 후보에 하나도 없으면(두 칸 떨어진 병합 등) '멈춤' 으로
            # 잘못 가르치게 되므로 뺀다
            if len(remaining) > 0 and item.labels.sum() == 0:
                continue
            items_of_mesh.append(item)
        stage1_items.append(items_of_mesh)

    stage2_items = []
    for sample in samples:
        stage2_items.append(stage2_item(sample.name, sample.mesh_after_stage1, normalizer, graph_options,
                                        sample.merges, use_context=use_context))

    first_candidates = {}
    for sample in samples:
        first_candidates[sample.name] = candidates_of_key[sample.name]
    return stage1_items, stage2_items, first_candidates


# ──────────────────────────────────────────────────────────────────────────
# OOF 채점 — 모든 임계값 조합
# ──────────────────────────────────────────────────────────────────────────
def finish_stage2(sample, mesh_after_stage1, applied_collapses, stage2_item_of_mesh, stage2_probabilities) -> dict:
    """1단계 결과에 2단계 임계값별로 병합을 적용해 사람과 비교한다. {2단계 임계값: compare 결과}."""
    true_collapse_set = set()
    for pair in sample.collapses:
        true_collapse_set.add(tuple(pair))

    result_by_threshold = {}
    result_by_choice = {}
    for stage2_threshold in GRID2:
        picked_edges = pick_merges(stage2_item_of_mesh.candidates, stage2_probabilities, stage2_threshold)
        sorted_edge_tuples = []
        for edge in picked_edges:
            sorted_edge_tuples.append(tuple(sorted(edge)))
        choice_key = tuple(sorted(sorted_edge_tuples))

        if choice_key not in result_by_choice:
            final_mesh = mesh_after_stage1.clone()
            merged_edges, _ = apply_merges(final_mesh, picked_edges)
            result = compare(sample.pair.before, final_mesh, sample.pair.after)
            result.update(name=sample.name, n_collapse=len(applied_collapses), n_merge=len(merged_edges),
                          collapse_exact=set(applied_collapses) == true_collapse_set,
                          collapses=ops_str(applied_collapses), merges=edges_str(merged_edges))
            result_by_choice[choice_key] = result
        result_by_threshold[stage2_threshold] = result_by_choice[choice_key]
    return result_by_threshold


def grid_eval(samples: list, first_candidates: dict, stage1_scorer, stage2_scorer, normalizer, graph_options: dict,
              use_context: bool, decode: str) -> dict:
    """OOF 메쉬마다 모든 (임계값1, 임계값2) 조합으로 재생성하고 사람 결과와 비교한다.

    반복형은 낮은 임계값으로 한 번 끝까지 풀고 임계값마다 그 앞부분을 쓴다. 같은 1단계
    결과는 한 번만 2단계를 돌린다. 반환: {(임계값1, 임계값2): [메쉬별 compare 결과]}
    """
    if decode == "iter":
        stage1_grid = GRID1_ITER
    else:
        stage1_grid = GRID1

    results = {}
    for stage1_threshold in stage1_grid:
        for stage2_threshold in GRID2:
            results[(stage1_threshold, stage2_threshold)] = []

    for sample in samples:
        bad_mesh = EditableMesh(sample.pair.before)

        # 1단계 임계값마다 고를 병합 목록
        choice_by_threshold = {}
        if decode == "iter":
            sequence = decode_collapses(stage1_scorer, sample.name, bad_mesh, normalizer, graph_options,
                                        first_candidates[sample.name], min_margin=stage1_grid[0])
            for stage1_threshold in stage1_grid:
                prefix_length = 0
                while prefix_length < len(sequence) and sequence[prefix_length][1] >= stage1_threshold:
                    prefix_length = prefix_length + 1
                chosen = []
                for pair, margin in sequence[:prefix_length]:
                    chosen.append(pair)
                choice_by_threshold[stage1_threshold] = chosen
        else:
            item = stage1_item(sample.name, bad_mesh, normalizer, graph_options, first_candidates[sample.name],
                               use_context=use_context)
            probabilities = stage1_scorer.predict([item])[0]
            for stage1_threshold in stage1_grid:
                choice_by_threshold[stage1_threshold] = pick_collapses(item.candidates, probabilities,
                                                                       stage1_threshold)

        # 같은 1단계 선택은 한 번만 2단계를 돈다
        stage2_results_by_choice = {}
        for stage1_threshold in stage1_grid:
            choice_key = tuple(choice_by_threshold[stage1_threshold])
            if choice_key not in stage2_results_by_choice:
                if decode == "iter":
                    mesh_after_stage1 = apply_collapse_seq(bad_mesh, choice_by_threshold[stage1_threshold])
                    applied_collapses = choice_by_threshold[stage1_threshold]
                else:
                    mesh_after_stage1 = bad_mesh.clone()
                    applied_collapses, _ = collapse_greedy(mesh_after_stage1, choice_by_threshold[stage1_threshold])
                item2 = stage2_item(sample.name, mesh_after_stage1, normalizer, graph_options,
                                    use_context=use_context)
                stage2_probabilities = stage2_scorer.predict([item2])[0]
                stage2_results_by_choice[choice_key] = finish_stage2(sample, mesh_after_stage1, applied_collapses,
                                                                     item2, stage2_probabilities)
            for stage2_threshold, result in stage2_results_by_choice[choice_key].items():
                results[(stage1_threshold, stage2_threshold)].append(result)
    return results


def mean_of(results: list, key: str) -> float:
    """compare 결과 목록에서 key 값의 평균."""
    values = []
    for result in results:
        values.append(result[key])
    return np.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="재생성 모델 학습 (CV + 최종)")
    parser.add_argument("--tag", required=True, help="실행 이름 → runs/regen_<tag>/")
    parser.add_argument("--scorer", choices=["gbm", "gnn"], default="gnn", help="1단계 채점기")
    parser.add_argument("--scorer2", choices=["gbm", "gnn"], default=None,
                        help="2단계 채점기 (기본: --scorer 와 같음). 2단계는 후보가 학습 셋 전체에 ~950개뿐이라 "
                             "gbm 이 GNN 보다 낫게 나왔다")
    parser.add_argument("--decode", choices=["iter", "oneshot"], default="iter")
    parser.add_argument("--loss", choices=["bce", "list"], default="bce",
                        help="1단계 손실. list = 메쉬마다 [후보들, 멈춤] softmax (gnn 만)")
    parser.add_argument("--epochs1", type=int, default=60, help="1단계(절점 병합) epoch")
    parser.add_argument("--epochs2", type=int, default=100, help="2단계(삼각형 병합) epoch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=4.0)
    parser.add_argument("--no-pretrain", action="store_true", help="인코더를 처음부터 학습")
    parser.add_argument("--ctx", action="store_true",
                        help="문맥 피처를 붙인다 — 반대 방향 결과와의 비교, 메쉬 안 순위, 메쉬 상태")
    parser.add_argument("--train-on", choices=["all", "reproducible"], default="all",
                        help="reproducible = 정답 대본이 사람 결과를 정확히 재현하는 쌍만으로 학습 "
                             "(채점은 언제나 전부로 한다)")
    parser.add_argument("--deploy", choices=["ensemble", "refit"], default="ensemble",
                        help="배포할 모델. ensemble = CV fold 모델들의 평균 (임계값을 고른 바로 그 모델들). "
                             "refit = 학습 셋 전부로 다시 학습 — 점수 눈금이 달라져 임계값이 어긋날 수 있다")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default=None,
                        help="쉼표로 여러 씨앗 (예: 0,1,2). fold 마다 씨앗별로 학습해 평균한다 — 같은 설정을 "
                             "두 번 돌려도 CV 가 ±2%%p 흔들려서(GPU 비결정성) 그 잡음을 평균으로 줄인다")
    parser.add_argument("--limit", type=int, default=None,
                        help="(디버그용) 학습 셋 앞에서 N쌍만 쓴다 — 흐름을 빨리 돌려 볼 때")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.loss == "list":
        if args.scorer != "gnn" or args.decode != "iter":
            parser.error("--loss list 는 --scorer gnn --decode iter 에서만 쓴다")

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    run_dir = os.path.join(gae_paths.RUNS_DIR, f"regen_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    start_time = time.time()

    def log(message: str) -> None:
        print(f"[{time.time() - start_time:6.0f}s] {message}", flush=True)

    stage2_scorer_name = args.scorer2
    if stage2_scorer_name is None:
        stage2_scorer_name = args.scorer

    split_table = load_split()
    train_names = sorted(split_table[split_table.split == "train"].name)
    if args.limit:
        train_names = train_names[:args.limit]
    test_count = int((split_table.split == "test").sum())
    log(f"학습 셋 {len(train_names)}쌍 (테스트 {test_count}쌍은 열지 않음) · "
        f"채점기={args.scorer}/{stage2_scorer_name} · 1단계={args.decode}/{args.loss} · 문맥피처={args.ctx} · "
        f"device={device}")

    samples = load_samples(train_names, verbose=False)
    _, normalizer, graph_options, _ = load_encoder(TrainConfig().encoder_run, device)
    stage1_items, stage2_items, first_candidates = build_items(samples, normalizer, graph_options, args.ctx,
                                                               use_states=args.decode == "iter")

    state_count = 0
    stage1_candidate_count = 0
    stage1_positive_count = 0
    for items_of_mesh in stage1_items:
        state_count = state_count + len(items_of_mesh)
        for item in items_of_mesh:
            stage1_candidate_count = stage1_candidate_count + item.candidate_count
            stage1_positive_count = stage1_positive_count + item.labels.sum()
    stage2_candidate_count = 0
    stage2_positive_count = 0
    for item in stage2_items:
        stage2_candidate_count = stage2_candidate_count + item.candidate_count
        stage2_positive_count = stage2_positive_count + item.labels.sum()
    log(f"1단계 표본 {state_count}개 (원본 {len(samples)} + 중간 상태 {state_count - len(samples)}) · "
        f"후보 {stage1_candidate_count}개 (정답 {int(stage1_positive_count)}) · "
        f"2단계 후보 {stage2_candidate_count}개 (정답 {int(stage2_positive_count)})")

    # 학습에 쓸 쌍 (reproducible 이면 정답 대본이 정확한 쌍만)
    use_for_fit = []
    for sample in samples:
        use_for_fit.append(args.train_on == "all" or sample.oracle_exact)

    stage1_config = TrainConfig(epochs=args.epochs1, learning_rate=args.lr, pos_weight=args.pos_weight,
                                seed=args.seed, pretrained=not args.no_pretrain, use_context=args.ctx)
    stage2_config = TrainConfig(epochs=args.epochs2, learning_rate=args.lr, pos_weight=args.pos_weight,
                                seed=args.seed, pretrained=not args.no_pretrain, use_context=args.ctx)
    stage1_config.loss = args.loss
    seeds = [args.seed]
    if args.seeds:
        seeds = []
        for seed_text in args.seeds.split(","):
            seeds.append(int(seed_text))
    Stage1ScorerClass = SCORERS[args.scorer]
    Stage2ScorerClass = SCORERS[stage2_scorer_name]

    # ── 1. 학습 셋 안에서 5-fold ────────────────────────────────────────
    # fold 는 분할 파일에 적힌 것을 쓴다 — 같은 결함을 고친 쌍이 학습과 채점에 갈라지지
    # 않도록 묶음 단위로 미리 정해 둔 것이다 (regen_data.defect_groups).
    fold_of_name = split_table.set_index("name").cv_fold
    fold_of_sample = []
    for sample in samples:
        fold_of_sample.append(int(fold_of_name[sample.name]))
    fold_count = max(fold_of_sample)

    all_results = {}
    fold_models = []
    for fold_number in range(1, fold_count + 1):
        scoring_indices = []
        fitting_indices = []
        for sample_index in range(len(samples)):
            if fold_of_sample[sample_index] == fold_number:
                scoring_indices.append(sample_index)
            elif use_for_fit[sample_index]:
                fitting_indices.append(sample_index)
        log(f"fold {fold_number}/{fold_count}: 학습 {len(fitting_indices)} / 채점 {len(scoring_indices)}")

        fitting_stage1_items = []
        fitting_stage2_items = []
        for sample_index in fitting_indices:
            for item in stage1_items[sample_index]:
                fitting_stage1_items.append(item)
            fitting_stage2_items.append(stage2_items[sample_index])

        models_of_fold = []
        for seed in seeds:
            stage1_scorer = Stage1ScorerClass(1, stage1_config.copy_with_seed(seed), device)
            stage1_scorer.fit(fitting_stage1_items, log)
            stage2_scorer = Stage2ScorerClass(2, stage2_config.copy_with_seed(seed), device)
            stage2_scorer.fit(fitting_stage2_items, log)
            models_of_fold.append((stage1_scorer, stage2_scorer))

        if len(models_of_fold) == 1:
            fold_stage1_scorer, fold_stage2_scorer = models_of_fold[0]
        else:
            stage1_members = []
            stage2_members = []
            for stage1_scorer, stage2_scorer in models_of_fold:
                stage1_members.append(stage1_scorer)
                stage2_members.append(stage2_scorer)
            fold_stage1_scorer = EnsembleScorer(stage1_members)
            fold_stage2_scorer = EnsembleScorer(stage2_members)

        scoring_samples = []
        for sample_index in scoring_indices:
            scoring_samples.append(samples[sample_index])
        fold_results = grid_eval(scoring_samples, first_candidates, fold_stage1_scorer, fold_stage2_scorer,
                                 normalizer, graph_options, args.ctx, args.decode)
        for threshold_pair, results in fold_results.items():
            if threshold_pair not in all_results:
                all_results[threshold_pair] = []
            all_results[threshold_pair].extend(results)

        best_pair = None
        best_exact = None
        for threshold_pair, results in fold_results.items():
            exact_rate = mean_of(results, "exact")
            if best_exact is None or exact_rate > best_exact:
                best_pair = threshold_pair
                best_exact = exact_rate
        log(f"  이 fold 최고 정확일치 {best_exact:.1%} @ {best_pair}")
        for model_pair in models_of_fold:
            fold_models.append(model_pair)

    # ── 임계값 선택 — CV 정확일치 최대, 동률이면 바뀐 요소 IoU ──────────────
    table_rows = []
    for (stage1_threshold, stage2_threshold), results in all_results.items():
        table_rows.append(dict(t1=stage1_threshold, t2=stage2_threshold, exact=mean_of(results, "exact"),
                               iou=mean_of(results, "change_iou"),
                               collapse_exact=mean_of(results, "collapse_exact")))
    grid_table = pd.DataFrame(table_rows).sort_values(["exact", "iou"], ascending=False)
    chosen_stage1_threshold = float(grid_table.iloc[0].t1)
    chosen_stage2_threshold = float(grid_table.iloc[0].t2)

    cv_table = pd.DataFrame(all_results[(chosen_stage1_threshold, chosen_stage2_threshold)])
    cv_table = cv_table.merge(split_table[["name", "cv_fold", "group", "kind", "oracle"]], on="name")
    cv_table.to_csv(os.path.join(run_dir, "cv_pred.csv"), index=False, encoding="utf-8-sig")
    grid_table.to_csv(os.path.join(run_dir, "cv_grid.csv"), index=False, encoding="utf-8-sig")

    reproducible = cv_table.oracle == "재현됨"
    log(f"선택한 임계값: 1단계 {chosen_stage1_threshold:.2f} / 2단계 {chosen_stage2_threshold:.2f}")
    log(f"CV 정확일치 {cv_table.exact.mean():.1%} ({int(cv_table.exact.sum())}/{len(cv_table)}) · "
        f"재현 가능한 쌍 중 {cv_table[reproducible].exact.mean():.1%} · "
        f"바뀐 요소 IoU {cv_table.change_iou.mean():.3f} · 1단계만 정확 {cv_table.collapse_exact.mean():.1%}")

    # ── 2. 배포 모델 ────────────────────────────────────────────────────
    all_fit_indices = []
    for sample_index in range(len(samples)):
        if use_for_fit[sample_index]:
            all_fit_indices.append(sample_index)

    if args.deploy == "ensemble":
        log(f"배포 모델 = CV fold 모델 {len(fold_models)}개의 평균 (fold {fold_count} × 씨앗 {len(seeds)}, "
            f"다시 학습하지 않는다)")
        stage1_members = []
        stage2_members = []
        for stage1_scorer, stage2_scorer in fold_models:
            stage1_members.append(stage1_scorer)
            stage2_members.append(stage2_scorer)
        deploy_stage1_scorer = EnsembleScorer(stage1_members)
        deploy_stage2_scorer = EnsembleScorer(stage2_members)
    else:
        log("배포 모델 = 학습 셋 전부로 다시 학습")
        all_stage1_items = []
        all_stage2_items = []
        for sample_index in all_fit_indices:
            for item in stage1_items[sample_index]:
                all_stage1_items.append(item)
            all_stage2_items.append(stage2_items[sample_index])
        deploy_stage1_scorer = Stage1ScorerClass(1, stage1_config, device).fit(all_stage1_items, log)
        deploy_stage2_scorer = Stage2ScorerClass(2, stage2_config, device).fit(all_stage2_items, log)

    meta = dict(tag=args.tag, scorer=args.scorer, scorer2=stage2_scorer_name, deploy=args.deploy,
                decode=args.decode, loss=args.loss,
                encoder_run=stage1_config.encoder_run, pretrained=not args.no_pretrain, ctx=args.ctx,
                train_on=args.train_on, folds=fold_count, epochs1=args.epochs1, epochs2=args.epochs2,
                lr=args.lr, pos_weight=args.pos_weight, seed=args.seed, seeds=seeds, n_train=len(samples),
                n_fit=len(all_fit_indices), n_state=state_count,
                cv_exact=float(cv_table.exact.mean()), cv_iou=float(cv_table.change_iou.mean()),
                cv_exact_reproducible=float(cv_table[reproducible].exact.mean()),
                split_file=os.path.relpath(os.path.join(gae_paths.RUNS_DIR, "regen_split.csv"), gae_paths.ROOT))
    RegenModel(deploy_stage1_scorer, deploy_stage2_scorer, chosen_stage1_threshold, chosen_stage2_threshold,
               normalizer, graph_options, meta).save(run_dir)
    log(f"저장: {run_dir}")
    log(f"테스트 셋 채점:  python src/regen_eval.py --run {os.path.relpath(run_dir, gae_paths.ROOT)}")


if __name__ == "__main__":
    main()

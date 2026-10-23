"""배포 방식 검증 — '다시 학습(refit)' 과 'fold 모델 평균(ensemble)' 을 학습 셋 안에서만 비교한다.

왜 필요했나: 처음 배포 모델은 CV 로 임계값을 고른 뒤 학습 셋 전부로 **다시 학습**했다.
테스트에서 절점삭제 메쉬의 병합을 평균 0.70개만 골랐는데(CV 의 fold 모델은 1.14개, 정답
1.63개), 다시 학습한 모델의 점수 눈금이 달라져 임계값이 어긋났다고 의심했다.

테스트 셋을 이미 한 번 열었으므로, 이 의심을 **테스트를 다시 쓰지 않고** 확인한다:

    학습 셋의 CV fold 하나(k)를 '가짜 테스트' 로 떼어 놓는다
    나머지 4 fold 로  (1) 안쪽 CV → 임계값 선택
                      (2-a) 4 fold 전부로 다시 학습 → k 채점        (refit)
                      (2-b) 안쪽 CV 의 fold 모델 4개 평균 → k 채점  (ensemble)
    k = 1..5 를 다 돌면 학습 셋 456쌍 모두가 한 번씩 가짜 테스트가 된다

    python src/regen_nested.py --outer 1      # fold 하나 (병렬로 5개를 띄운다)
    python src/regen_nested.py --summary      # 다 끝난 뒤 합친다
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import pandas as pd
import torch

import gae_paths
from meshdiff import compare
from meshedit import EditableMesh
from regen_data import load_samples, load_split
from regen_model import SCORERS, EnsembleScorer, RegenModel, TrainConfig, load_encoder
from regen_train import build_items, grid_eval, mean_of

#: 결과 폴더
OUT = os.path.join(gae_paths.RUNS_DIR, "regen_nested")


def run_outer(outer_fold: int, args) -> None:
    """바깥 fold 하나를 가짜 테스트로 떼고 refit / ensemble 을 비교한다."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    start_time = time.time()

    def log(message: str) -> None:
        print(f"[outer {outer_fold} {time.time() - start_time:5.0f}s] {message}", flush=True)

    split_table = load_split()
    train_table = split_table[split_table.split == "train"]
    samples = load_samples(sorted(train_table.name), verbose=False)
    fold_of_name = train_table.set_index("name").cv_fold
    _, normalizer, graph_options, _ = load_encoder(TrainConfig().encoder_run, device)
    stage1_items, stage2_items, first_candidates = build_items(samples, normalizer, graph_options, False,
                                                               use_states=True, verbose=False)

    fold_of_sample = []
    for sample in samples:
        fold_of_sample.append(int(fold_of_name[sample.name]))
    inner_folds = []
    for fold_number in sorted(set(fold_of_sample)):
        if fold_number != outer_fold:
            inner_folds.append(fold_number)

    stage1_config = TrainConfig(epochs=args.epochs1, loss=args.loss)
    stage2_config = TrainConfig(epochs=args.epochs2)
    Stage1ScorerClass = SCORERS[args.scorer]
    Stage2ScorerClass = SCORERS[args.scorer2]

    # ── 안쪽 CV — 임계값 고르기 ──────────────────────────────────────────
    all_results = {}
    inner_models = []
    for inner_fold in inner_folds:
        fitting_stage1_items = []
        fitting_stage2_items = []
        scoring_samples = []
        for sample_index in range(len(samples)):
            fold_number = fold_of_sample[sample_index]
            if fold_number != outer_fold and fold_number != inner_fold:
                for item in stage1_items[sample_index]:
                    fitting_stage1_items.append(item)
                fitting_stage2_items.append(stage2_items[sample_index])
            if fold_number == inner_fold:
                scoring_samples.append(samples[sample_index])

        stage1_scorer = Stage1ScorerClass(1, stage1_config, device).fit(fitting_stage1_items)
        stage2_scorer = Stage2ScorerClass(2, stage2_config, device).fit(fitting_stage2_items)
        fold_results = grid_eval(scoring_samples, first_candidates, stage1_scorer, stage2_scorer,
                                 normalizer, graph_options, False, "iter")
        for threshold_pair, results in fold_results.items():
            if threshold_pair not in all_results:
                all_results[threshold_pair] = []
            all_results[threshold_pair].extend(results)
        inner_models.append((stage1_scorer, stage2_scorer))
        log(f"안쪽 fold {inner_fold} 끝")

    best_pair = None
    best_score = None
    for threshold_pair, results in all_results.items():
        score = (mean_of(results, "exact"), mean_of(results, "change_iou"))
        if best_score is None or score > best_score:
            best_pair = threshold_pair
            best_score = score
    log(f"안쪽 CV 로 고른 임계값 {best_pair}  (안쪽 OOF 정확일치 {best_score[0]:.1%})")

    # ── refit 과 ensemble 만들기 ──────────────────────────────────────────
    all_fit_stage1_items = []
    all_fit_stage2_items = []
    for sample_index in range(len(samples)):
        if fold_of_sample[sample_index] != outer_fold:
            for item in stage1_items[sample_index]:
                all_fit_stage1_items.append(item)
            all_fit_stage2_items.append(stage2_items[sample_index])
    refit_model = RegenModel(Stage1ScorerClass(1, stage1_config, device).fit(all_fit_stage1_items),
                             Stage2ScorerClass(2, stage2_config, device).fit(all_fit_stage2_items),
                             best_pair[0], best_pair[1], normalizer, graph_options, {"decode": "iter"})

    stage1_members = []
    stage2_members = []
    for stage1_scorer, stage2_scorer in inner_models:
        stage1_members.append(stage1_scorer)
        stage2_members.append(stage2_scorer)
    ensemble_model = RegenModel(EnsembleScorer(stage1_members), EnsembleScorer(stage2_members),
                                best_pair[0], best_pair[1], normalizer, graph_options, {"decode": "iter"})

    # ── 가짜 테스트 채점 ──────────────────────────────────────────────────
    rows = []
    for sample_index in range(len(samples)):
        if fold_of_sample[sample_index] != outer_fold:
            continue
        sample = samples[sample_index]
        bad_mesh = EditableMesh(sample.pair.before)
        for deploy_name, model in (("refit", refit_model), ("ensemble", ensemble_model)):
            regen_result = model.regenerate(bad_mesh, first_candidates[sample.name])
            row = compare(sample.pair.before, regen_result.mesh, sample.pair.after)
            row.update(name=sample.name, outer=outer_fold, deploy=deploy_name, t1=best_pair[0], t2=best_pair[1],
                       n_collapse=len(regen_result.collapses), n_collapse_true=len(sample.collapses),
                       n_merge=len(regen_result.merges), n_merge_true=len(sample.merges))
            rows.append(row)

    os.makedirs(OUT, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, f"outer{outer_fold}.csv"), index=False, encoding="utf-8-sig")
    for deploy_name, deploy_table in table.groupby("deploy"):
        log(f"{deploy_name:8s} 정확일치 {deploy_table.exact.mean():.1%}  "
            f"병합 {deploy_table.n_collapse.mean():.2f}개/메쉬 (정답 {deploy_table.n_collapse_true.mean():.2f})")


def summary() -> None:
    """outer*.csv 를 합쳐 refit 과 ensemble 을 비교한다."""
    tables = []
    for path in sorted(glob.glob(os.path.join(OUT, "outer*.csv"))):
        tables.append(pd.read_csv(path, encoding="utf-8-sig"))
    table = pd.concat(tables)
    split_table = load_split().set_index("name")
    table["kind"] = table.name.map(split_table.kind)
    print(f"가짜 테스트 {table.name.nunique()}쌍 (outer fold {sorted(table.outer.unique())})")

    by_deploy = table.groupby("deploy").agg(exact=("exact", "mean"), iou=("change_iou", "mean"),
                                            pred_collapse=("n_collapse", "mean"),
                                            true_collapse=("n_collapse_true", "mean"),
                                            pred_merge=("n_merge", "mean"), true_merge=("n_merge_true", "mean"))
    print(by_deploy.round(3).to_string())

    print("\n절점삭제 메쉬만:")
    node_delete_table = table[table.kind == "절점삭제"]
    print(node_delete_table.groupby("deploy").agg(n=("name", "size"), exact=("exact", "mean"),
                                                  pred_collapse=("n_collapse", "mean"),
                                                  true_collapse=("n_collapse_true", "mean")).round(3).to_string())

    pivot = table.pivot_table(index="name", columns="deploy", values="exact")
    both = pivot.dropna()
    only_ensemble = int(((both.ensemble == 1) & (both.refit == 0)).sum())
    only_refit = int(((both.ensemble == 0) & (both.refit == 1)).sum())
    print(f"\n같은 메쉬끼리: ensemble 만 맞힘 {only_ensemble} / refit 만 맞힘 {only_refit}")
    table.to_csv(os.path.join(OUT, "all.csv"), index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="refit vs ensemble — 학습 셋 안 중첩 검증")
    parser.add_argument("--outer", type=int)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--scorer", default="gnn")
    parser.add_argument("--scorer2", default="gbm")
    parser.add_argument("--loss", default="list")
    parser.add_argument("--epochs1", type=int, default=60)
    parser.add_argument("--epochs2", type=int, default=100)
    args = parser.parse_args()
    if args.summary:
        summary()
    elif args.outer:
        run_outer(args.outer, args)
    else:
        parser.error("--outer k 또는 --summary")


if __name__ == "__main__":
    main()

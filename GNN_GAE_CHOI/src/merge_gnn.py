"""1단계 수리 모델(초기 시도) — "어느 삼각형 쌍을 합칠 것인가" 를 GNN 으로 고른다.

과제 정의는 `task_merge.py` 참고. 요약하면:

    입력   비정상 메쉬 + 합칠 수 있는 삼각형 쌍 후보 (메쉬당 평균 2.8개)
    출력   어느 후보를 합칠 것인가 (1~5개)
    채점   Good 메쉬와 쉘 집합이 정확히 일치하는가

**요소기반 그래프에서 병합 후보는 인접한 두 삼각형 '노드' 다.** 그래서 분류기가 쓰던
그래프를 그대로 쓰고, 노드 임베딩 두 개를 묶어 후보 점수를 낸다.

표본이 191개뿐이라 두 가지를 쓴다:
  - **사전학습 인코더 전이** — 심판 분류기(`deploy_10_14`)의 가중치로 시작한다.
  - **메쉬 단위 5-fold** — 같은 메쉬가 학습과 채점에 동시에 들어가지 않게 한다.

    python src/merge_gnn.py                    # 5-fold 학습 + 채점 → out/merge_pred.csv
    python src/merge_gnn.py --no-pretrain      # 전이 없이
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch_geometric.data import Batch

import gae_paths
import dataset as DS
import features as F
import graphs
import paths as parent_paths
from explain import load_full_model
from meshrep import MeshRep
from models import mlp
from task_merge import build as build_samples

#: 후보 자체의 피처 — 그래프에는 없는, "합쳤을 때 어떻게 되는가" 정보
CAND_FEATS = ["dihedral", "q_skew", "q_aspect", "q_warp", "q_angle_min", "q_angle_max",
              "q_taper", "q_area_rel", "lines_on_edge", "tri_skew_max", "tri_aspect_max",
              "tri_angle_min"]


def candidate_vector(mesh, candidate, median_edge: float) -> list:
    """후보 하나를 숫자 벡터로. 길이 피처는 메쉬의 중앙 변길이로 상대화한다."""
    merged_quality = candidate.quality
    quality_a = F.shell_quality(mesh.corners(candidate.si))
    quality_b = F.shell_quality(mesh.corners(candidate.sj))
    return [
        candidate.dihedral, merged_quality.skew, merged_quality.aspect, merged_quality.warp,
        merged_quality.angle_min, merged_quality.angle_max, merged_quality.taper,
        merged_quality.area / (median_edge * median_edge + 1e-9), float(candidate.lines_on_edge),
        max(quality_a.skew, quality_b.skew), max(quality_a.aspect, quality_b.aspect),
        min(quality_a.angle_min, quality_b.angle_min),
    ]


# ──────────────────────────────────────────────────────────────────────────
class MergeScorer(nn.Module):
    """메쉬를 인코딩하고, 후보(그래프 노드 쌍)마다 점수를 낸다."""

    def __init__(self, encoder, hidden: int, candidate_feature_count: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        # 후보는 **순서 없는 쌍**이다. (i,j) 와 (j,i) 가 같은 점수가 나오도록
        # 합과 절대차만 쓴다 — 이어 붙이면 순서에 따라 답이 달라진다.
        self.candidate_head = mlp(2 * hidden + candidate_feature_count, hidden, hidden // 2, 1, dropout=dropout)

    def forward(self, data, node_pairs: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        node_embeddings = self.encoder.encode(data)
        embedding_a = node_embeddings[node_pairs[:, 0]]
        embedding_b = node_embeddings[node_pairs[:, 1]]
        head_input = torch.cat([embedding_a + embedding_b, (embedding_a - embedding_b).abs(), candidate_features],
                               dim=-1)
        return self.candidate_head(head_input).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────
def prepare(samples: list, normalizer, classifier_args, device) -> list:
    """표본을 학습용 dict 로 바꾼다 (PyG Data, 후보 노드쌍, 후보피처, 정답, 메타)."""
    graph_options = {}
    if classifier_args.graph == "elem":
        graph_options = {"corner_edges": not classifier_args.no_corner_edges,
                         "include_lines": not classifier_args.no_lines}
    items = []
    for sample in samples:
        rep = MeshRep(sample.mesh.to_raw())
        graph_arrays = graphs.build(rep, classifier_args.graph, **graph_options)
        data = DS.to_pyg(graph_arrays, 0, sample.name, normalizer)
        median_edge = rep.median_edge_length
        node_pairs = []
        candidate_features = []
        labels = []
        edges = []
        for candidate in sample.candidates:
            # 요소기반 그래프는 쉘을 0..n_shell-1 로 먼저 깔므로 쉘 인덱스 = 그래프 노드 인덱스
            node_pairs.append([candidate.si, candidate.sj])
            candidate_features.append(candidate_vector(sample.mesh, candidate, median_edge))
            edge = sample.candidate_edge(candidate)
            edges.append(edge)
            if edge in sample.truth:
                labels.append(1.0)
            else:
                labels.append(0.0)
        if len(sample.candidates) >= sample.n_truth:
            choice_count = math.comb(len(sample.candidates), sample.n_truth)
        else:
            choice_count = 1
        items.append(dict(
            name=sample.name, data=data,
            pairs=torch.tensor(node_pairs, dtype=torch.long),
            cand=torch.tensor(candidate_features, dtype=torch.float32),
            y=torch.tensor(labels, dtype=torch.float32),
            edges=edges, cands=sample.candidates, truth=sample.truth, n_truth=sample.n_truth,
            n_choice=choice_count,
        ))
    return items


def normalize_candidates(items: list, fit_indices) -> None:
    """후보 피처를 학습 fold 통계로 표준화한다 (fold 밖 정보가 새지 않도록). 결과는 item["cand_n"]."""
    feature_tensors = []
    for item_index in fit_indices:
        feature_tensors.append(items[item_index]["cand"])
    all_features = torch.cat(feature_tensors, dim=0)
    mean = all_features.mean(0, keepdim=True)
    std = all_features.std(0, keepdim=True).clamp_min(1e-6)
    for item in items:
        item["cand_n"] = (item["cand"] - mean) / std


def pick(scores: np.ndarray, candidates: list, k: int | None = None, threshold: float = 0.5) -> list:
    """점수 순으로 고르되 **삼각형이 겹치는 후보는 같이 못 쓴다**. 고른 후보 번호 목록.

    k 를 주면 k 개까지, 안 주면 점수 ≥ threshold 인 것까지.
    """
    order = np.argsort(-scores)
    used_shells = set()
    picked = []
    for candidate_index in order:
        if k is not None and len(picked) >= k:
            break
        if k is None and scores[candidate_index] < threshold:
            break
        shell_index_a = candidates[candidate_index].si
        shell_index_b = candidates[candidate_index].sj
        if shell_index_a in used_shells or shell_index_b in used_shells:
            continue
        used_shells.add(shell_index_a)
        used_shells.add(shell_index_b)
        picked.append(candidate_index)
    return picked


def collate(items: list, indices: list, device):
    """메쉬 여러 개를 한 배치로 묶는다.

    PyG 가 노드를 이어 붙이면서 번호를 밀므로, 후보 노드쌍도 그 메쉬의 시작 위치(ptr)만큼 더해 준다.
    """
    data_list = []
    for item_index in indices:
        data_list.append(items[item_index]["data"])
    batch = Batch.from_data_list(data_list).to(device)
    offsets = batch.ptr[:-1]
    shifted_pairs = []
    normalized_features = []
    labels = []
    sizes = []
    for position in range(len(indices)):
        item = items[indices[position]]
        shifted_pairs.append(item["pairs"] + int(offsets[position]))
        normalized_features.append(item["cand_n"])
        labels.append(item["y"])
        sizes.append(len(item["y"]))
    node_pairs = torch.cat(shifted_pairs, 0)
    candidate_features = torch.cat(normalized_features, 0)
    label_tensor = torch.cat(labels, 0)
    return batch, node_pairs.to(device), candidate_features.to(device), label_tensor.to(device), sizes


def raw_scores(items: list, indices, model, device) -> list:
    """메쉬별 후보 확률 배열 목록."""
    model.eval()
    with torch.no_grad():
        batch, node_pairs, candidate_features, _, sizes = collate(items, list(indices), device)
        all_probabilities = torch.sigmoid(model(batch, node_pairs, candidate_features)).cpu().numpy()
    result = []
    position = 0
    for size in sizes:
        result.append(all_probabilities[position: position + size])
        position = position + size
    return result


def picked_edge_set(item: dict, picked_indices: list) -> set:
    edges = set()
    for candidate_index in picked_indices:
        edges.add(item["edges"][candidate_index])
    return edges


def tune_threshold(items: list, indices, model, device) -> float:
    """**학습 fold 에서** 임계값을 고른다 (채점 데이터로 고르면 숫자가 낙관적이 된다)."""
    scores = raw_scores(items, indices, model, device)
    best_correct = -1
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.96, 0.05):
        correct_count = 0
        for position in range(len(indices)):
            item = items[indices[position]]
            picked_indices = pick(scores[position], item["cands"], k=None, threshold=float(threshold))
            if picked_edge_set(item, picked_indices) == item["truth"]:
                correct_count = correct_count + 1
        if correct_count > best_correct:
            best_correct = correct_count
            best_threshold = float(threshold)
    return best_threshold


def evaluate(items: list, indices, model, device, use_truth_count: bool, threshold: float = 0.5) -> dict:
    """채점. ``use_truth_count`` 면 사람이 합친 개수를 알려주고 고르게 한다(낙관적)."""
    exact_count = 0
    top1_count = 0
    real_choice_count = 0
    real_choice_exact = 0
    picked_total = 0
    scores = raw_scores(items, indices, model, device)
    for position in range(len(indices)):
        item = items[indices[position]]
        item_scores = scores[position]
        if use_truth_count:
            picked_indices = pick(item_scores, item["cands"], k=item["n_truth"], threshold=threshold)
        else:
            picked_indices = pick(item_scores, item["cands"], k=None, threshold=threshold)
        is_correct = picked_edge_set(item, picked_indices) == item["truth"]
        if is_correct:
            exact_count = exact_count + 1
        picked_total = picked_total + len(picked_indices)
        if item["edges"][int(np.argmax(item_scores))] in item["truth"]:
            top1_count = top1_count + 1
        if item["n_choice"] > 1:
            real_choice_count = real_choice_count + 1
            if is_correct:
                real_choice_exact = real_choice_exact + 1
    return dict(exact=exact_count / max(1, len(indices)), top1=top1_count / max(1, len(indices)),
                exact_real=real_choice_exact / max(1, real_choice_count), n_real=real_choice_count,
                n_pick=picked_total / max(1, len(indices)), thr=threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="병합 후보 선택 GNN")
    parser.add_argument("--run", default=None, help="전이할 분류기 (기본: deploy_10_14)")
    parser.add_argument("--no-pretrain", action="store_true", help="인코더를 처음부터 학습")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--enc-lr-scale", type=float, default=0.1,
                        help="사전학습 인코더의 학습률 배율 (너무 크면 전이가 깨진다)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-pred", default=None,
                        help="out-of-fold 예측을 csv 로 저장 (기본: out/merge_pred.csv). app_merge.py 가 읽는다.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    run_dir = args.run
    if run_dir is None:
        run_dir = os.path.join(parent_paths.RUNS_TRAIN, "deploy_10_14")
    base_model, normalizer, classifier_args, _ = load_full_model(run_dir, device)
    print(f"[merge_gnn] 전이 원본: {run_dir} · device={device} · graph={classifier_args.graph}")

    print("표본을 만드는 중...")
    samples = build_samples()
    all_items = prepare(samples, normalizer, classifier_args, device)
    # 정답이 후보에 다 들어 있는 메쉬만 학습/채점한다 (열거가 못 담은 것은 과제 밖)
    items = []
    for item in all_items:
        if item["truth"].issubset(set(item["edges"])) and item["n_truth"]:
            items.append(item)
    candidate_total = 0
    positive_total = 0
    for item in items:
        candidate_total = candidate_total + len(item["y"])
        positive_total = positive_total + item["y"].sum()
    print(f"  메쉬 {len(items)}개 · 후보 {candidate_total}개 · 정답 {int(positive_total)}개")

    random_generator = np.random.default_rng(args.seed)
    order = random_generator.permutation(len(items))
    folds = np.array_split(order, args.folds)

    fold_results = []
    prediction_rows = []
    for fold_number in range(1, len(folds) + 1):
        scoring_indices = folds[fold_number - 1]
        other_folds = []
        for fold_index in range(len(folds)):
            if fold_index != fold_number - 1:
                other_folds.append(folds[fold_index])
        fitting_indices = np.concatenate(other_folds)
        normalize_candidates(items, fitting_indices)

        torch.manual_seed(args.seed + fold_number)
        if args.no_pretrain:
            if base_model.separate_encoder:
                node_input_dim = base_model.node_enc_shell[0].in_features
            else:
                node_input_dim = base_model.node_enc[0].in_features
            encoder = type(base_model)(
                d_node=node_input_dim, d_edge=base_model.edge_enc[0].in_features, d_graph=1,
                conv=classifier_args.conv, hidden=base_model.hidden, layers=len(base_model.convs),
                dropout=classifier_args.dropout, heads=classifier_args.heads,
                use_graph_features=False, separate_encoder=base_model.separate_encoder,
            ).to(device)
        else:
            encoder = load_full_model(run_dir, device)[0]
        model = MergeScorer(encoder, base_model.hidden, len(CAND_FEATS), classifier_args.dropout).to(device)

        if args.no_pretrain:
            encoder_learning_rate = args.lr * 1.0
        else:
            encoder_learning_rate = args.lr * args.enc_lr_scale
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": encoder_learning_rate},
            {"params": model.candidate_head.parameters(), "lr": args.lr},
        ], weight_decay=1e-4)

        for epoch in range(args.epochs):
            model.train()
            random_generator.shuffle(fitting_indices)
            for batch_start in range(0, len(fitting_indices), args.batch_size):
                batch_indices = list(fitting_indices[batch_start: batch_start + args.batch_size])
                if len(batch_indices) < 2:
                    # BatchNorm 은 배치 1 에서 못 돈다
                    continue
                batch, node_pairs, candidate_features, labels, _ = collate(items, batch_indices, device)
                scores = model(batch, node_pairs, candidate_features)
                loss = torch_functional.binary_cross_entropy_with_logits(scores, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 임계값은 **학습 fold 에서만** 고른다
        threshold = tune_threshold(items, fitting_indices, model, device)
        # 채점 fold 의 후보 점수를 모아 둔다 — 각 메쉬는 자기를 학습에 쓰지 않은 모델의 점수만 갖는다
        for item_index in scoring_indices:
            item = items[item_index]
            item_scores = raw_scores(items, [item_index], model, device)[0]
            picked_indices = set(pick(item_scores, item["cands"], k=None, threshold=threshold))
            for candidate_index in range(len(item["edges"])):
                edge = item["edges"][candidate_index]
                if candidate_index in picked_indices:
                    picked_flag = 1
                else:
                    picked_flag = 0
                if edge in item["truth"]:
                    truth_flag = 1
                else:
                    truth_flag = 0
                prediction_rows.append(dict(
                    name=item["name"], fold=fold_number, edge=f"{min(edge)}-{max(edge)}",
                    si=item["cands"][candidate_index].si, sj=item["cands"][candidate_index].sj,
                    score=float(item_scores[candidate_index]), picked=picked_flag, truth=truth_flag, thr=threshold,
                    n_cand=len(item["edges"]), n_truth=item["n_truth"], n_choice=item["n_choice"]))
        result_with_count = evaluate(items, scoring_indices, model, device, use_truth_count=True)
        result_self = evaluate(items, scoring_indices, model, device, use_truth_count=False, threshold=threshold)
        fold_results.append((result_with_count, result_self))
        print(f"  fold {fold_number}: [개수 알려줌] 전체 {result_with_count['exact']:.1%} "
              f"실제선택 {result_with_count['exact_real']:.1%}  |  [스스로 정함 thr={threshold:.2f}] "
              f"전체 {result_self['exact']:.1%} 실제선택 {result_self['exact_real']:.1%} "
              f"(평균 {result_self['n_pick']:.2f}개 선택)")

    print(f"\n── {args.folds}-fold 평균 ──")
    print(f"  {'':32s}{'개수 알려줌':>16s}{'스스로 정함':>18s}")
    for key, label in (("exact", "정확히 일치 (전체)"), ("top1", "상위1 적중"),
                       ("exact_real", "정확히 일치 (실제 선택분)")):
        values_with_count = []
        values_self = []
        for result_with_count, result_self in fold_results:
            values_with_count.append(result_with_count[key])
            values_self.append(result_self[key])
        values_with_count = np.array(values_with_count)
        values_self = np.array(values_self)
        print(f"  {label:30s} {values_with_count.mean():7.1%} ±{values_with_count.std():4.1%}  "
              f"{values_self.mean():9.1%} ±{values_self.std():4.1%}")
    picked_per_mesh = []
    for result_with_count, result_self in fold_results:
        picked_per_mesh.append(result_self["n_pick"])
    picked_per_mesh = np.array(picked_per_mesh)
    truth_counts = []
    for item in items:
        truth_counts.append(item["n_truth"])
    print(f"  {'메쉬당 선택 개수':30s} {float(np.mean(truth_counts)):7.2f} (정답)  {picked_per_mesh.mean():9.2f}")
    print("\n  비교 기준 — 무작위 31.4% / 규칙 47.1% (실제로 골라야 하는 121개 기준)")

    import pandas as pd

    gae_paths.ensure_dirs()
    out_path = args.save_pred
    if out_path is None:
        out_path = os.path.join(gae_paths.OUT_DIR, "merge_pred.csv")
    pd.DataFrame(prediction_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n예측 저장: {out_path}  (후보 {len(prediction_rows)}개)")
    print("  화면으로 보려면:  streamlit run GNN_GAE_CHOI/src/app_merge.py")


if __name__ == "__main__":
    main()

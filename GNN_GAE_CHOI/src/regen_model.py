"""재생성 모델 — 후보에 점수를 매기는 두 개의 채점기와, 그것으로 메쉬를 다시 만드는 파이프라인.

    원본(Bad)
      │  1단계: 절점 병합 — "다음에 합칠 것 하나, 또는 멈춤" 을 반복 (iter)
      │         또는 후보 전부에 한 번에 점수를 매겨 임계값 이상을 묶어 적용 (oneshot)
      ▼
    중간 메쉬
      │  2단계: 삼각형 쌍 후보 ~2개에 점수 → 임계값 이상을 겹치지 않게 적용
      ▼
    재생성 결과

1단계를 왜 반복형으로 바꿨나 (oneshot 의 CV 를 쪼개 본 결과):
  - 절점 병합이 2개 이상인 메쉬에서 정답 개수를 못 채웠다 (2개짜리에 평균 1.4개).
    첫 병합을 하고 나면 두 번째 자리의 모양이 바뀌는데, 한 번에 점수를 매기면 그걸 못 본다.
  - "이 메쉬에 병합이 필요한가" 를 임계값 하나로 가르기 어려웠다.
그래서 반복형은 매 단계 후보들과 **'멈춤'** 을 한 줄에 세워 고르게 하고(listwise), 하나를
적용한 뒤 메쉬를 다시 보고 다음을 고른다.

채점기 종류 (같은 사용법):
  ``gnn``  부모의 요소기반 그래프 + 심판 분류기(`deploy_10_14`) 인코더를 이어받아 미세조정.
           요소 임베딩 → 절점 임베딩(붙은 요소의 평균·최대) → 후보 = [remove, keep, 피처].
           2단계 후보 = [두 삼각형 임베딩의 합, 차의 절댓값, 피처] — 순서 없는 쌍이라 대칭.
           ``loss="list"`` 면 '멈춤' 점수를 메쉬 임베딩(요소 평균·최대 + 메쉬 상태)으로 낸다.
  ``gbm``  후보 피처만 쓰는 gradient boosting. 그래프 문맥 없이 어디까지 되나의 기준선.
  ``ens``  같은 설정으로 학습한 채점기 여러 개(CV 의 fold 모델들)의 평균. 배포 모델이 이것이다.

채점기가 공통으로 가진 메서드:
  fit(items)       학습
  predict(items)   후보별 확률 (oneshot 선택, 2단계 선택에 쓴다)
  scores(items)    [(후보별 점수, 멈춤 점수)] — 반복형 선택은 '최고 후보 점수 - 멈춤 점수' 로 판단한다.
                   BCE 로 배운 채점기는 멈춤 점수가 0 이다 (= 확률 0.5 기준의 logit).

**저장 파일 형식**(model.pt / model.json)은 원래 GNN_GAE 코드와 똑같이 유지한다. 그래서
원래 코드로 학습한 runs/regen_final 을 그대로 불러올 수 있다. 파일 안의 짧은 키 이름
(s1, s2, thr1, cfg, lr, enc ...)은 저장/불러오기 함수에서만 쓰고, 코드 안에서는 풀어 쓴
이름(stage1_scorer, stage1_threshold, config, learning_rate, encoder ...)으로 다룬다.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

import gae_paths
import paths as parent_paths
from meshedit import (REGEN_AREA_TOL, EditableMesh, apply_collapse_seq, apply_merges, collapse_greedy,
                      total_shell_area)
from models import mlp
from regen_feats import (COLLAPSE_CTX_FEATS, COLLAPSE_FEATS, MERGE_CTX_FEATS, MERGE_FEATS, MESH_FEATS,
                         CollapseCands, MergeCands, collapse_candidates, merge_cands, mesh_graph,
                         mesh_state_row, with_context)

#: 인코더를 이어받을 분류기. 수정 쌍 원본을 학습에서 뺀 심판이다.
DEFAULT_ENCODER_RUN = os.path.join(parent_paths.RUNS_TRAIN, "deploy_10_14")

#: 메쉬 하나에서 고를 절점 병합의 상한. 사람은 최대 7개였다 (508쌍). 임계값이 낮게
#: 잡혀도 메쉬를 통째로 무너뜨리지 않게 하는 안전장치다.
MAX_COLLAPSES = 10


# ──────────────────────────────────────────────────────────────────────────
# 피처 정규화
# ──────────────────────────────────────────────────────────────────────────
class FeatureNormalizer:
    """부호 보존 log1p 뒤 z-score. 통계는 **학습 데이터에서만** 적합한다.

    aspect·면적비처럼 꼬리가 수십 배로 긴 피처와 각도(0~180)가 섞여 있다. log1p 로 꼬리를
    누르고 나서 표준화하면 한 규칙으로 다 다룰 수 있다.
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    @staticmethod
    def signed_log(values: np.ndarray) -> np.ndarray:
        """sign(x) * log(1 + |x|)."""
        return np.sign(values) * np.log1p(np.abs(values))

    def fit(self, values: np.ndarray) -> "FeatureNormalizer":
        transformed = self.signed_log(values)
        self.mean = transformed.mean(0)
        self.std = np.maximum(transformed.std(0), 1e-6)
        return self

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return ((self.signed_log(values) - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        """파일에 저장할 dict (키 이름은 원래 형식 mu / sd)."""
        return {"mu": self.mean.tolist(), "sd": self.std.tolist()}

    @classmethod
    def from_state(cls, saved: dict) -> "FeatureNormalizer":
        return cls(np.array(saved["mu"], np.float32), np.array(saved["sd"], np.float32))


# ──────────────────────────────────────────────────────────────────────────
# 학습/추론 단위
# ──────────────────────────────────────────────────────────────────────────
class ScoringItem:
    """메쉬(또는 중간 상태) 하나 × 단계 하나. 후보와 (있으면) 정답."""

    def __init__(self, name: str, stage: int, graph, candidates, labels=None, mesh_features=None):
        self.name = name
        #: 1 (절점 병합) 또는 2 (삼각형 쌍 병합)
        self.stage = stage
        #: regen_feats.MeshGraph
        self.graph = graph
        #: CollapseCands 또는 MergeCands
        self.candidates = candidates
        #: 후보마다 정답이면 1, 아니면 0
        if labels is None:
            labels = np.zeros(0, np.float32)
        self.labels = labels
        #: 메쉬 상태 (MESH_FEATS) — '멈춤' 점수의 입력
        if mesh_features is None:
            mesh_features = np.zeros(len(MESH_FEATS), np.float32)
        self.mesh_features = mesh_features

    @property
    def candidate_count(self) -> int:
        return len(self.candidates.pairs)


def feature_count(stage: int, use_context: bool) -> int:
    """후보 피처 수 (문맥 피처를 붙이면 그만큼 늘어난다)."""
    if stage == 1:
        count = len(COLLAPSE_FEATS)
        if use_context:
            count = count + len(COLLAPSE_CTX_FEATS)
        return count
    count = len(MERGE_FEATS)
    if use_context:
        count = count + len(MERGE_CTX_FEATS)
    return count


def stage1_item(name: str, mesh: EditableMesh, normalizer, graph_options: dict,
                candidates: CollapseCands | None = None, truth=None, use_context: bool = False) -> ScoringItem:
    """1단계 채점 단위. candidates 를 주면(캐시) 다시 계산하지 않는다. truth = 정답 병합 목록."""
    if candidates is None:
        candidates = collapse_candidates(mesh)
    mesh_features = np.array(mesh_state_row(mesh, len(candidates.pairs)), np.float32)
    if use_context:
        candidates = with_context(mesh, candidates, 1)
    graph = mesh_graph(mesh, normalizer, graph_options)

    wanted = set()
    if truth is not None:
        for pair in truth:
            wanted.add(tuple(pair))
    label_list = []
    for pair in candidates.pairs:
        if pair in wanted:
            label_list.append(1.0)
        else:
            label_list.append(0.0)
    labels = np.array(label_list, np.float32)
    return ScoringItem(name, 1, graph, candidates, labels, mesh_features)


def stage2_item(name: str, mesh: EditableMesh, normalizer, graph_options: dict, truth=None,
                use_context: bool = False) -> ScoringItem:
    """2단계 채점 단위. truth = 정답 공유 변 목록."""
    candidates = merge_cands(mesh)
    mesh_features = np.array(mesh_state_row(mesh, len(candidates.pairs)), np.float32)
    if use_context:
        candidates = with_context(mesh, candidates, 2)
    graph = mesh_graph(mesh, normalizer, graph_options)

    wanted = set()
    if truth is not None:
        for edge in truth:
            wanted.add(edge)
    label_list = []
    for edge in candidates.edges:
        if edge in wanted:
            label_list.append(1.0)
        else:
            label_list.append(0.0)
    labels = np.array(label_list, np.float32)
    return ScoringItem(name, 2, graph, candidates, labels, mesh_features)


class BatchInput:
    """메쉬 여러 개를 한 번에 신경망에 넣기 위한 묶음."""

    def __init__(self, graph_batch, incidence, vertex_count, candidate_index, features, labels,
                 candidate_mesh, mesh_features):
        #: torch_geometric Batch
        self.graph_batch = graph_batch
        #: (2, nnz) [절점 인덱스, 그래프 노드 인덱스] — 배치 전체 번호로 밀어 둔 것
        self.incidence = incidence
        #: 배치 전체 절점 수
        self.vertex_count = vertex_count
        #: (후보 수, 2) 후보가 가리키는 절점(1단계) 또는 그래프 노드(2단계) 인덱스
        self.candidate_index = candidate_index
        #: (후보 수, 피처 수) 정규화된 후보 피처
        self.features = features
        #: (후보 수,) 정답
        self.labels = labels
        #: (후보 수,) 후보마다 몇 번째 메쉬인가
        self.candidate_mesh = candidate_mesh
        #: (메쉬 수, len(MESH_FEATS)) 정규화된 메쉬 상태
        self.mesh_features = mesh_features


def collate_items(items: list, feature_normalizer, mesh_feature_normalizer, device) -> BatchInput:
    """메쉬 여러 개를 한 배치로. 요소·절점 번호를 메쉬마다 밀어 준다."""
    data_list = []
    for item in items:
        data_list.append(item.graph.data)
    graph_batch = Batch.from_data_list(data_list).to(device)
    element_offsets = graph_batch.ptr[:-1].tolist()

    vertex_offsets = []
    vertex_total = 0
    for item in items:
        vertex_offsets.append(vertex_total)
        vertex_total = vertex_total + len(item.graph.vertex_ids)

    shifted_incidence = []
    for item_index in range(len(items)):
        offset = torch.tensor([[vertex_offsets[item_index]], [element_offsets[item_index]]])
        shifted_incidence.append(items[item_index].graph.incidence + offset)
    incidence = torch.cat(shifted_incidence, 1)

    normalized_features = []
    label_arrays = []
    for item in items:
        normalized_features.append(feature_normalizer(item.candidates.feats))
        label_arrays.append(item.labels)
    features = torch.from_numpy(np.concatenate(normalized_features, 0))
    labels = torch.from_numpy(np.concatenate(label_arrays, 0))

    candidate_index_list = []
    if items[0].stage == 1:
        for item_index in range(len(items)):
            item = items[item_index]
            vertex_index_of = {}
            for vertex_index in range(len(item.graph.vertex_ids)):
                vertex_index_of[item.graph.vertex_ids[vertex_index]] = vertex_index
            for remove_node, keep_node in item.candidates.pairs:
                candidate_index_list.append([vertex_index_of[remove_node] + vertex_offsets[item_index],
                                             vertex_index_of[keep_node] + vertex_offsets[item_index]])
    else:
        for item_index in range(len(items)):
            item = items[item_index]
            for shell_index_a, shell_index_b in item.candidates.pairs:
                candidate_index_list.append([shell_index_a + element_offsets[item_index],
                                             shell_index_b + element_offsets[item_index]])
    candidate_index = torch.tensor(candidate_index_list, dtype=torch.long).reshape(-1, 2)

    candidate_mesh_list = []
    for item_index in range(len(items)):
        for _ in range(items[item_index].candidate_count):
            candidate_mesh_list.append(item_index)
    candidate_mesh = torch.tensor(candidate_mesh_list, dtype=torch.long)

    mesh_feature_list = []
    for item in items:
        mesh_feature_list.append(item.mesh_features)
    mesh_feature_matrix = np.stack(mesh_feature_list, 0)
    if mesh_feature_normalizer is not None:
        mesh_features = torch.from_numpy(mesh_feature_normalizer(mesh_feature_matrix))
    else:
        mesh_features = torch.from_numpy(mesh_feature_matrix.astype(np.float32))

    return BatchInput(graph_batch, incidence.to(device), vertex_total, candidate_index.to(device),
                      features.to(device), labels.to(device), candidate_mesh.to(device), mesh_features.to(device))


# ──────────────────────────────────────────────────────────────────────────
# 신경망
# ──────────────────────────────────────────────────────────────────────────
class CandidateNet(nn.Module):
    """요소 임베딩 → 후보 점수 (+ 선택적으로 메쉬마다 '멈춤' 점수).

    1단계(절점 쌍, 방향 있음) / 2단계(삼각형 쌍, 대칭) 공용.
    """

    def __init__(self, encoder, hidden: int, stage: int, feature_dim: int, dropout: float = 0.3,
                 use_stop: bool = False):
        super().__init__()
        self.encoder = encoder
        self.stage = stage
        # 1단계: [remove 절점(평균·최대), keep 절점(평균·최대), 피처] — 방향이 결과를 가르므로
        #        remove/keep 자리를 고정해 비대칭을 그대로 둔다.
        # 2단계: [합, 차의 절댓값, 피처] — (i,j) 와 (j,i) 가 같은 점수가 나와야 한다.
        if stage == 1:
            input_dim = 4 * hidden + feature_dim
        else:
            input_dim = 2 * hidden + feature_dim
        self.candidate_head = mlp(input_dim, hidden, hidden // 2, 1, dropout=dropout)
        # '멈춤' — 메쉬 전체를 보고 "더 고칠 게 없다" 의 점수. 후보 점수와 같은 줄에 세운다.
        if use_stop:
            self.stop_head = mlp(2 * hidden + len(MESH_FEATS), hidden, hidden // 2, 1, dropout=dropout)
        else:
            self.stop_head = None

    def forward(self, batch_input: BatchInput):
        element_embeddings = self.encoder.encode(batch_input.graph_batch)

        if self.stage == 1:
            # 절점 임베딩 = 그 절점에 붙은 요소 임베딩의 평균과 최대
            embedding_per_incidence = element_embeddings[batch_input.incidence[1]]
            vertex_mean = scatter(embedding_per_incidence, batch_input.incidence[0], dim=0,
                                  dim_size=batch_input.vertex_count, reduce="mean")
            vertex_max = scatter(embedding_per_incidence, batch_input.incidence[0], dim=0,
                                 dim_size=batch_input.vertex_count, reduce="max")
            vertex_embeddings = torch.cat([vertex_mean, vertex_max], 1)
            remove_embedding = vertex_embeddings[batch_input.candidate_index[:, 0]]
            keep_embedding = vertex_embeddings[batch_input.candidate_index[:, 1]]
            head_input = torch.cat([remove_embedding, keep_embedding, batch_input.features], 1)
        else:
            embedding_a = element_embeddings[batch_input.candidate_index[:, 0]]
            embedding_b = element_embeddings[batch_input.candidate_index[:, 1]]
            head_input = torch.cat([embedding_a + embedding_b, (embedding_a - embedding_b).abs(),
                                    batch_input.features], 1)

        candidate_scores = self.candidate_head(head_input).squeeze(-1)
        if self.stop_head is None:
            return candidate_scores, None

        mesh_count = batch_input.mesh_features.size(0)
        mesh_mean = scatter(element_embeddings, batch_input.graph_batch.batch, dim=0, dim_size=mesh_count,
                            reduce="mean")
        mesh_max = scatter(element_embeddings, batch_input.graph_batch.batch, dim=0, dim_size=mesh_count,
                           reduce="max")
        pooled = torch.cat([mesh_mean, mesh_max, batch_input.mesh_features], 1)
        stop_scores = self.stop_head(pooled).squeeze(-1)
        return candidate_scores, stop_scores


#: 신경망 가중치 이름 — 파일(원래 형식)의 접두어 → 코드 안 접두어
WEIGHT_PREFIX_FILE_TO_CODE = {"enc.": "encoder.", "head.": "candidate_head.", "stop_head.": "stop_head."}


def weights_from_file_format(saved_weights: dict) -> dict:
    """파일에 저장된 가중치 이름(enc.*, head.*)을 코드의 이름(encoder.*, candidate_head.*)으로."""
    renamed = {}
    for key, value in saved_weights.items():
        new_key = key
        for file_prefix, code_prefix in WEIGHT_PREFIX_FILE_TO_CODE.items():
            if key.startswith(file_prefix):
                new_key = code_prefix + key[len(file_prefix):]
                break
        renamed[new_key] = value
    return renamed


def weights_to_file_format(code_weights: dict) -> dict:
    """코드의 가중치 이름을 파일 형식(원래 GNN_GAE 와 같은 이름)으로."""
    renamed = {}
    for key, value in code_weights.items():
        new_key = key
        for file_prefix, code_prefix in WEIGHT_PREFIX_FILE_TO_CODE.items():
            if key.startswith(code_prefix):
                new_key = file_prefix + key[len(code_prefix):]
                break
        renamed[new_key] = value
    return renamed


def segment_logsumexp(values: torch.Tensor, segment: torch.Tensor, segment_count: int) -> torch.Tensor:
    """구간별 log-sum-exp. segment[i] = values[i] 가 속한 구간 번호."""
    segment_max = scatter(values.detach(), segment, dim=0, dim_size=segment_count, reduce="max")
    exp_sum = scatter((values - segment_max[segment]).exp(), segment, dim=0, dim_size=segment_count, reduce="sum")
    return exp_sum.log() + segment_max


def listwise_loss(candidate_scores, stop_scores, labels, candidate_mesh, mesh_count: int) -> torch.Tensor:
    """메쉬마다 [후보들, 멈춤] 위의 softmax. 정답 = 남은 사람 병합 중 아무거나(여럿이면 합), 없으면 멈춤.

    -inf 대신 큰 음수로 가린다 — torch.where 의 안 쓰는 쪽에 log(0) 이 있으면 기울기가 NaN 이 된다.
    """
    device = candidate_scores.device
    all_scores = torch.cat([candidate_scores, stop_scores])
    all_segments = torch.cat([candidate_mesh, torch.arange(mesh_count, device=device)])
    has_positive = scatter(labels, candidate_mesh, dim=0, dim_size=mesh_count, reduce="sum") > 0
    target_mask = torch.cat([labels > 0, ~has_positive])
    masked_scores = torch.where(target_mask, all_scores, torch.full_like(all_scores, -1e4))
    target_lse = segment_logsumexp(masked_scores, all_segments, mesh_count)
    return (segment_logsumexp(all_scores, all_segments, mesh_count) - target_lse).mean()


def load_encoder(run: str, device, pretrained: bool = True):
    """(인코더, 그래프 정규화기, 그래프 옵션, hidden 크기).

    pretrained=False 면 같은 구조를 새로 만든다 (가중치는 무작위).
    분류 head·pooling 은 쓰지 않는다 — encode() 까지만 쓴다.
    """
    from explain import load_full_model

    base_model, normalizer, classifier_args, _ = load_full_model(run, device)
    graph_options = {"corner_edges": not classifier_args.no_corner_edges,
                     "include_lines": not classifier_args.no_lines}
    if not pretrained:
        from models import MeshGNN

        if base_model.separate_encoder:
            node_input_dim = base_model.node_enc_shell[0].in_features
        else:
            node_input_dim = base_model.node_enc[0].in_features
        base_model = MeshGNN(d_node=node_input_dim, d_edge=base_model.edge_enc[0].in_features, d_graph=1,
                             conv=classifier_args.conv, hidden=base_model.hidden, layers=len(base_model.convs),
                             dropout=classifier_args.dropout, heads=classifier_args.heads,
                             use_graph_features=False,
                             separate_encoder=base_model.separate_encoder).to(device)
    return base_model, normalizer, graph_options, base_model.hidden


# ──────────────────────────────────────────────────────────────────────────
# 학습 설정
# ──────────────────────────────────────────────────────────────────────────
#: 설정 — 코드 안 이름 → 파일(원래 형식)의 키 이름. 파일에 저장하는 순서도 이 순서다.
CONFIG_CODE_TO_FILE = {
    "epochs": "epochs",
    "learning_rate": "lr",
    "encoder_lr_scale": "enc_lr_scale",
    "batch_size": "batch_size",
    "pos_weight": "pos_weight",
    "dropout": "dropout",
    "weight_decay": "weight_decay",
    "seed": "seed",
    "pretrained": "pretrained",
    "encoder_run": "encoder_run",
    "use_context": "ctx",
    "loss": "loss",
}


class TrainConfig:
    """채점기 학습 설정."""

    def __init__(self, epochs: int = 60, learning_rate: float = 1e-3, encoder_lr_scale: float = 0.1,
                 batch_size: int = 16, pos_weight: float = 4.0, dropout: float = 0.3,
                 weight_decay: float = 1e-4, seed: int = 0, pretrained: bool = True,
                 encoder_run: str = DEFAULT_ENCODER_RUN, use_context: bool = False, loss: str = "bce"):
        self.epochs = epochs
        self.learning_rate = learning_rate
        #: 사전학습 인코더의 학습률 배율 (head 보다 작게 — 너무 크면 이어받은 것이 깨진다)
        self.encoder_lr_scale = encoder_lr_scale
        self.batch_size = batch_size
        #: 정답 후보의 손실 가중치 (정답이 드물어서)
        self.pos_weight = pos_weight
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.seed = seed
        #: 심판 분류기 인코더의 가중치로 시작하나
        self.pretrained = pretrained
        self.encoder_run = encoder_run
        #: 문맥 피처(반대 방향 비교·메쉬 안 순위·메쉬 상태)를 붙이나 (regen_feats.with_context)
        self.use_context = use_context
        #: "bce" = 후보마다 독립 이진분류 / "list" = 메쉬마다 [후보들, 멈춤] softmax (1단계용)
        self.loss = loss

    def copy_with_seed(self, seed: int) -> "TrainConfig":
        """씨앗만 바꾼 사본."""
        copy = TrainConfig()
        for code_name in CONFIG_CODE_TO_FILE:
            setattr(copy, code_name, getattr(self, code_name))
        copy.seed = seed
        return copy

    def to_file_dict(self) -> dict:
        """파일에 저장할 dict (키 이름은 원래 형식)."""
        saved = {}
        for code_name, file_name in CONFIG_CODE_TO_FILE.items():
            saved[file_name] = getattr(self, code_name)
        return saved

    @classmethod
    def from_file_dict(cls, saved: dict) -> "TrainConfig":
        config = cls()
        for code_name, file_name in CONFIG_CODE_TO_FILE.items():
            if file_name in saved:
                setattr(config, code_name, saved[file_name])
        return config


def probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# 채점기
# ──────────────────────────────────────────────────────────────────────────
class GNNScorer:
    """그래프 신경망 채점기."""

    kind = "gnn"

    def __init__(self, stage: int, config: TrainConfig, device):
        self.stage = stage
        self.config = config
        self.device = device
        self.net = None
        #: 후보 피처 정규화기
        self.feature_normalizer = None
        #: 메쉬 상태 피처 정규화기
        self.mesh_feature_normalizer = None

    def build_net(self, pretrained: bool) -> None:
        encoder, _, _, hidden = load_encoder(self.config.encoder_run, self.device, pretrained)
        use_stop = self.config.loss == "list"
        self.net = CandidateNet(encoder, hidden, self.stage, feature_count(self.stage, self.config.use_context),
                                self.config.dropout, use_stop=use_stop).to(self.device)

    def fit(self, items: list, log=None) -> "GNNScorer":
        config = self.config
        torch.manual_seed(config.seed)
        random_generator = np.random.default_rng(config.seed)
        self.build_net(config.pretrained)

        usable_items = []
        for item in items:
            if item.candidate_count > 0:
                usable_items.append(item)
        items = usable_items

        all_features = []
        all_mesh_features = []
        for item in items:
            all_features.append(item.candidates.feats)
            all_mesh_features.append(item.mesh_features)
        self.feature_normalizer = FeatureNormalizer().fit(np.concatenate(all_features, 0))
        self.mesh_feature_normalizer = FeatureNormalizer().fit(np.stack(all_mesh_features, 0))

        if config.pretrained:
            encoder_learning_rate = config.learning_rate * config.encoder_lr_scale
        else:
            encoder_learning_rate = config.learning_rate
        head_parameters = list(self.net.candidate_head.parameters())
        if self.net.stop_head is not None:
            head_parameters = head_parameters + list(self.net.stop_head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": self.net.encoder.parameters(), "lr": encoder_learning_rate},
            {"params": head_parameters, "lr": config.learning_rate},
        ], weight_decay=config.weight_decay)
        positive_weight = torch.tensor(config.pos_weight, device=self.device)

        item_order = np.arange(len(items))
        for epoch in range(config.epochs):
            self.net.train()
            random_generator.shuffle(item_order)
            loss_total = 0.0
            for batch_start in range(0, len(item_order), config.batch_size):
                batch_items = []
                for item_position in item_order[batch_start: batch_start + config.batch_size]:
                    batch_items.append(items[item_position])
                if len(batch_items) < 2:
                    # BatchNorm 은 배치 1 에서 못 돈다
                    continue
                batch_input = collate_items(batch_items, self.feature_normalizer, self.mesh_feature_normalizer,
                                            self.device)
                if len(batch_input.labels) < 2:
                    continue
                candidate_scores, stop_scores = self.net(batch_input)
                if config.loss == "list":
                    loss = listwise_loss(candidate_scores, stop_scores, batch_input.labels,
                                         batch_input.candidate_mesh, len(batch_items))
                else:
                    loss = torch_functional.binary_cross_entropy_with_logits(
                        candidate_scores, batch_input.labels, pos_weight=positive_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_total = loss_total + float(loss.detach()) * len(batch_items)
            if log is not None and (epoch + 1) % 20 == 0:
                log(f"      [{self.stage}단계] epoch {epoch + 1}/{config.epochs}  loss {loss_total / len(items):.4f}")
        return self

    @torch.no_grad()
    def scores(self, items: list) -> list:
        """[(후보별 점수 배열, 멈춤 점수)] — 후보가 없는 항목은 (빈 배열, 0.0)."""
        self.net.eval()
        results = []
        for _ in range(len(items)):
            results.append((np.zeros(0, np.float32), 0.0))

        live_indices = []
        for item_index in range(len(items)):
            if items[item_index].candidate_count > 0:
                live_indices.append(item_index)

        for chunk_start in range(0, len(live_indices), 32):
            chunk_indices = live_indices[chunk_start: chunk_start + 32]
            chunk_items = []
            for item_index in chunk_indices:
                chunk_items.append(items[item_index])
            batch_input = collate_items(chunk_items, self.feature_normalizer, self.mesh_feature_normalizer,
                                        self.device)
            candidate_scores, stop_scores = self.net(batch_input)
            candidate_scores = candidate_scores.cpu().numpy()
            if stop_scores is not None:
                stop_scores = stop_scores.cpu().numpy()
            else:
                stop_scores = np.zeros(len(chunk_items), np.float32)

            position = 0
            for chunk_position in range(len(chunk_indices)):
                item_index = chunk_indices[chunk_position]
                count = chunk_items[chunk_position].candidate_count
                results[item_index] = (candidate_scores[position: position + count],
                                       float(stop_scores[chunk_position]))
                position = position + count
        return results

    def predict(self, items: list) -> list:
        """후보별 확률 = sigmoid(후보 점수 - 멈춤 점수)."""
        probabilities = []
        for candidate_scores, stop_score in self.scores(items):
            probabilities.append(1.0 / (1.0 + np.exp(-(candidate_scores - stop_score))))
        return probabilities

    def state(self) -> dict:
        if self.mesh_feature_normalizer is not None:
            mesh_feature_state = self.mesh_feature_normalizer.state()
        else:
            mesh_feature_state = None
        return {"kind": self.kind, "stage": self.stage, "cfg": self.config.to_file_dict(),
                "net": weights_to_file_format(self.net.state_dict()), "fn": self.feature_normalizer.state(),
                "fng": mesh_feature_state}

    @classmethod
    def from_state(cls, saved: dict, device) -> "GNNScorer":
        scorer = cls(saved["stage"], TrainConfig.from_file_dict(saved["cfg"]), device)
        scorer.build_net(pretrained=True)
        scorer.net.load_state_dict(weights_from_file_format(saved["net"]))
        scorer.net.eval()
        scorer.feature_normalizer = FeatureNormalizer.from_state(saved["fn"])
        if saved.get("fng"):
            scorer.mesh_feature_normalizer = FeatureNormalizer.from_state(saved["fng"])
        else:
            scorer.mesh_feature_normalizer = None
        return scorer


class GBMScorer:
    """후보 피처만 쓰는 gradient boosting 채점기 (그래프 문맥 없음)."""

    kind = "gbm"

    def __init__(self, stage: int, config: TrainConfig, device=None):
        self.stage = stage
        self.config = config
        self.classifier = None

    def fit(self, items: list, log=None) -> "GBMScorer":
        from sklearn.ensemble import HistGradientBoostingClassifier

        feature_arrays = []
        label_arrays = []
        for item in items:
            if item.candidate_count > 0:
                feature_arrays.append(item.candidates.feats)
                label_arrays.append(item.labels)
        features = np.concatenate(feature_arrays, 0)
        labels = np.concatenate(label_arrays, 0)
        sample_weights = np.where(labels > 0, self.config.pos_weight, 1.0)
        self.classifier = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                         random_state=self.config.seed)
        self.classifier.fit(features, labels, sample_weight=sample_weights)
        return self

    def predict(self, items: list) -> list:
        probabilities = []
        for item in items:
            if item.candidate_count > 0:
                probabilities.append(self.classifier.predict_proba(item.candidates.feats)[:, 1].astype(np.float32))
            else:
                probabilities.append(np.zeros(0, np.float32))
        return probabilities

    def scores(self, items: list) -> list:
        results = []
        for probability in self.predict(items):
            results.append((probability_to_logit(probability), 0.0))
        return results

    def state(self) -> dict:
        return {"kind": self.kind, "stage": self.stage, "cfg": self.config.to_file_dict(),
                "clf": pickle.dumps(self.classifier)}

    @classmethod
    def from_state(cls, saved: dict, device=None) -> "GBMScorer":
        scorer = cls(saved["stage"], TrainConfig.from_file_dict(saved["cfg"]))
        scorer.classifier = pickle.loads(saved["clf"])
        return scorer


class EnsembleScorer:
    """같은 설정으로 학습한 채점기 여러 개(CV 의 fold 모델들)의 평균.

    왜 이걸 배포하나: 임계값은 **fold 모델들의** out-of-fold 점수로 골랐다. 학습 셋 전부로
    다시 학습한 모델은 점수의 눈금이 달라져서(실측: 테스트의 절점삭제 메쉬에서 병합을 평균
    0.70개만 골랐다 — CV 의 fold 모델은 1.14개, 정답 1.63개) 그 임계값이 맞지 않는다.
    fold 모델들을 그대로 평균하면 임계값을 고른 바로 그 모델들의 눈금이 유지된다.
    점수(후보 점수·멈춤 점수)를 평균하고, 확률은 확률끼리 평균한다.
    """

    kind = "ens"

    def __init__(self, members: list):
        self.members = members
        self.stage = members[0].stage
        self.config = members[0].config

    def scores(self, items: list) -> list:
        member_outputs = []
        for member in self.members:
            member_outputs.append(member.scores(items))

        results = []
        for item_index in range(len(items)):
            candidate_score_list = []
            stop_score_list = []
            for output in member_outputs:
                candidate_score_list.append(output[item_index][0])
                stop_score_list.append(output[item_index][1])
            mean_candidate_scores = np.mean(candidate_score_list, axis=0).astype(np.float32)
            mean_stop_score = float(np.mean(stop_score_list))
            results.append((mean_candidate_scores, mean_stop_score))
        return results

    def predict(self, items: list) -> list:
        member_outputs = []
        for member in self.members:
            member_outputs.append(member.predict(items))

        results = []
        for item_index in range(len(items)):
            probability_list = []
            for output in member_outputs:
                probability_list.append(output[item_index])
            results.append(np.mean(probability_list, axis=0).astype(np.float32))
        return results

    def state(self) -> dict:
        member_states = []
        for member in self.members:
            member_states.append(member.state())
        return {"kind": self.kind, "stage": self.stage, "members": member_states}

    @classmethod
    def from_state(cls, saved: dict, device=None) -> "EnsembleScorer":
        members = []
        for member_state in saved["members"]:
            members.append(scorer_from_state(member_state, device))
        return cls(members)


#: 채점기 종류 이름 → 클래스
SCORERS = {"gnn": GNNScorer, "gbm": GBMScorer, "ens": EnsembleScorer}


def scorer_from_state(saved: dict, device):
    """저장된 dict 의 kind 를 보고 맞는 채점기 클래스로 불러온다."""
    scorer_class = SCORERS[saved["kind"]]
    return scorer_class.from_state(saved, device)


# ──────────────────────────────────────────────────────────────────────────
# 선택 규칙 — 점수를 실제 편집으로
# ──────────────────────────────────────────────────────────────────────────
def pick_collapses(candidates: CollapseCands, probabilities: np.ndarray, threshold: float) -> list:
    """(oneshot) 임계값 이상을 점수 순으로 (최대 ``MAX_COLLAPSES``). 남길지는 ``collapse_greedy`` 가 정한다."""
    order = np.argsort(-probabilities, kind="stable")
    picked = []
    for candidate_index in order:
        if probabilities[candidate_index] >= threshold:
            picked.append(candidates.pairs[candidate_index])
    return picked[:MAX_COLLAPSES]


def total_area(mesh: EditableMesh) -> float:
    """메쉬 전체 쉘 면적."""
    return total_shell_area(mesh.rep)


def decode_collapses(stage1_scorer, name: str, bad: EditableMesh, normalizer, graph_options: dict,
                     first_candidates: CollapseCands | None = None, min_margin: float = -np.inf,
                     max_steps: int = MAX_COLLAPSES) -> list:
    """(iter) '최고 후보 하나를 합친다 → 메쉬를 다시 본다' 를 반복한다. [((remove, keep), 여유)].

    여유(margin) = 최고 후보 점수 - 멈춤 점수. ``min_margin`` 밑으로 떨어지면 멈춘다.
    고르는 **순서**는 임계값과 무관하므로, 낮은 임계값으로 한 번 끝까지 풀어 두면 어떤
    임계값의 결과든 그 앞부분(prefix)이다 — CV 에서 임계값을 고를 때 이걸 쓴다.
    후보는 매번 '지금 메쉬에서 적용 가능한 것' 이라 고른 병합은 항상 적용된다.
    first_candidates 는 원본의 1단계 후보 (캐시에서 온 것, 없으면 계산한다).
    """
    current_mesh = bad.clone()
    current_mesh.log.clear()
    original_area = total_area(bad)
    sequence = []
    candidates = first_candidates

    for step in range(max_steps):
        item = stage1_item(name, current_mesh, normalizer, graph_options, candidates,
                           use_context=stage1_scorer.config.use_context)
        if item.candidate_count == 0:
            break
        candidate_scores, stop_score = stage1_scorer.scores([item])[0]

        picked = None
        for candidate_index in np.argsort(-candidate_scores, kind="stable"):
            margin = float(candidate_scores[candidate_index] - stop_score)
            if margin < min_margin:
                break
            remove_node, keep_node = item.candidates.pairs[candidate_index]
            trial_mesh = current_mesh.clone()
            trial_mesh.collapse_edge(remove_node, keep_node)
            # 면적 가드는 **원본 대비 누적**으로 본다. 병합 하나하나는 5% 안이어도 여러 번
            # 쌓이면 넘을 수 있다 (실측 `_0231`: 병합 4번에 +6.3%). 넘으면 다음 후보로.
            if abs(total_area(trial_mesh) - original_area) / original_area > REGEN_AREA_TOL:
                continue
            picked = ((remove_node, keep_node), margin, trial_mesh)
            break

        if picked is None:
            break
        chosen_pair, margin, current_mesh = picked
        sequence.append((chosen_pair, margin))
        # 다음 단계부터는 메쉬가 바뀌었으니 후보를 새로 계산한다
        candidates = None
    return sequence


def pick_merges(candidates: MergeCands, probabilities: np.ndarray, threshold: float) -> list:
    """임계값 이상을 점수 순으로, **삼각형이 겹치지 않게**. 한 삼각형이 여러 후보에 걸린다."""
    used_shells = set()
    picked_edges = []
    for candidate_index in np.argsort(-probabilities, kind="stable"):
        if probabilities[candidate_index] < threshold:
            break
        shell_index_a, shell_index_b = candidates.pairs[candidate_index]
        if shell_index_a in used_shells or shell_index_b in used_shells:
            continue
        used_shells.add(shell_index_a)
        used_shells.add(shell_index_b)
        picked_edges.append(candidates.edges[candidate_index])
    return picked_edges


class RegenResult:
    """재생성 결과 하나."""

    def __init__(self, mesh, mesh_after_stage1, collapses=None, merges=None, rejected=None, margins=None,
                 stage2_probabilities=None, stage2_candidates=None):
        #: 재생성 결과 메쉬
        self.mesh = mesh
        #: 1단계 뒤 메쉬
        self.mesh_after_stage1 = mesh_after_stage1
        if collapses is None:
            collapses = []
        if merges is None:
            merges = []
        if rejected is None:
            rejected = {}
        if margins is None:
            margins = []
        #: 적용한 (remove, keep), 적용 순서
        self.collapses = collapses
        #: 적용한 공유 변
        self.merges = merges
        #: 고른 병합 중 거부된 것 → 사유 (oneshot)
        self.rejected = rejected
        #: 병합마다 여유 (iter)
        self.margins = margins
        self.stage2_probabilities = stage2_probabilities
        self.stage2_candidates = stage2_candidates

    @property
    def absorbed(self) -> dict:
        """{사라진 절점: 이어받은 절점} — 화면 설명용 (병합 사슬을 끝까지 따라간다)."""
        forward = {}
        for remove_node, keep_node in self.collapses:
            forward[remove_node] = keep_node
        result = {}
        for remove_node in forward:
            final_node = forward[remove_node]
            while final_node in forward:
                final_node = forward[final_node]
            result[remove_node] = final_node
        return result


class RegenModel:
    """두 채점기 + 임계값 + 그래프 정규화. 학습이 끝난 실행 폴더 하나에 대응한다.

    decode = "iter" 면 stage1_threshold 는 여유(최고 후보 - 멈춤)의 하한이고, "oneshot" 이면 확률이다.
    """

    def __init__(self, stage1_scorer, stage2_scorer, stage1_threshold: float, stage2_threshold: float,
                 normalizer, graph_options: dict, meta: dict | None = None):
        self.stage1_scorer = stage1_scorer
        self.stage2_scorer = stage2_scorer
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold
        self.normalizer = normalizer
        self.graph_options = graph_options
        if meta is None:
            meta = {}
        self.meta = meta

    @property
    def decode(self) -> str:
        return self.meta.get("decode", "oneshot")

    def stage2(self, name: str, mesh_after_stage1: EditableMesh):
        """(2단계 후보, 후보별 확률)."""
        item = stage2_item(name, mesh_after_stage1, self.normalizer, self.graph_options,
                           use_context=self.stage2_scorer.config.use_context)
        probabilities = self.stage2_scorer.predict([item])[0]
        return item.candidates, probabilities

    def regenerate(self, bad: EditableMesh, first_candidates: CollapseCands | None = None,
                   stage1_threshold: float | None = None, stage2_threshold: float | None = None) -> RegenResult:
        """원본 하나를 다시 만든다. 원본은 건드리지 않는다."""
        if stage1_threshold is None:
            stage1_threshold = self.stage1_threshold
        if stage2_threshold is None:
            stage2_threshold = self.stage2_threshold

        rejected = {}
        margins = []
        if self.decode == "iter":
            sequence = decode_collapses(self.stage1_scorer, bad.name, bad, self.normalizer, self.graph_options,
                                        first_candidates, stage1_threshold)
            applied = []
            for pair, margin in sequence:
                applied.append(pair)
                margins.append(margin)
            mesh = apply_collapse_seq(bad, applied)
        else:
            item = stage1_item(bad.name, bad, self.normalizer, self.graph_options, first_candidates,
                               use_context=self.stage1_scorer.config.use_context)
            probabilities = self.stage1_scorer.predict([item])[0]
            mesh = bad.clone()
            mesh.log.clear()
            applied, rejected = collapse_greedy(mesh, pick_collapses(item.candidates, probabilities,
                                                                     stage1_threshold))

        mesh_after_stage1 = mesh.clone()
        stage2_candidates, stage2_probabilities = self.stage2(bad.name, mesh_after_stage1)
        picked_edges = pick_merges(stage2_candidates, stage2_probabilities, stage2_threshold)
        merged_edges, _ = apply_merges(mesh, picked_edges)
        return RegenResult(mesh, mesh_after_stage1, applied, merged_edges, rejected, margins,
                           stage2_probabilities, stage2_candidates)

    # ── 저장 / 불러오기 (원래 GNN_GAE 와 같은 파일 형식) ──────────────────
    def save(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        torch.save({"s1": self.stage1_scorer.state(), "s2": self.stage2_scorer.state()},
                   os.path.join(run_dir, "model.pt"))
        meta = dict(self.meta)
        meta["thr1"] = self.stage1_threshold
        meta["thr2"] = self.stage2_threshold
        meta["gopts"] = self.graph_options
        with open(os.path.join(run_dir, "model.json"), "w", encoding="utf-8") as file_handle:
            json.dump(meta, file_handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, run_dir: str, device=None) -> "RegenModel":
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        with open(os.path.join(run_dir, "model.json"), encoding="utf-8") as file_handle:
            meta = json.load(file_handle)
        saved = torch.load(os.path.join(run_dir, "model.pt"), map_location=device, weights_only=False)
        stage1_scorer = scorer_from_state(saved["s1"], device)
        stage2_scorer = scorer_from_state(saved["s2"], device)
        _, normalizer, graph_options, _ = load_encoder(meta.get("encoder_run", DEFAULT_ENCODER_RUN), device)
        return cls(stage1_scorer, stage2_scorer, meta["thr1"], meta["thr2"], normalizer, graph_options, meta)


# ──────────────────────────────────────────────────────────────────────────
# 1단계 후보 캐시 — 미리보기를 계산하느라 메쉬당 0.5초 걸린다
# ──────────────────────────────────────────────────────────────────────────
def feature_code_version() -> str:
    """1단계 후보·피처를 만드는 **함수들의 코드**가 바뀌면 캐시를 버린다.

    파일 통째로 해시하면 무관한 수정에도 수백 개를 다시 계산하느라 5분을 쓴다.
    그래서 결과에 영향을 주는 함수의 소스만 본다.
    """
    import features
    import meshedit
    import regen_feats

    functions = [regen_feats.collapse_candidates, regen_feats.region_summary, regen_feats.vertex_table,
                 regen_feats.shells_area,
                 meshedit.collapse_pairs, meshedit.compact_nodes, meshedit.is_convex_quad, meshedit.node_relation,
                 meshedit.count_edge_use, meshedit.follow_forward, meshedit.total_shell_area,
                 meshedit.EditableMesh.collapse_many,
                 meshedit.EditableMesh.collapse_edge, meshedit.EditableMesh.clone, meshedit.EditableMesh.points_of,
                 features.shell_quality, features.area_normal]
    hasher = hashlib.sha1()
    for function in functions:
        hasher.update(inspect.getsource(function).encode())
    hasher.update(repr(regen_feats.COLLAPSE_FEATS).encode())
    hasher.update(repr(meshedit.REGEN_AREA_TOL).encode())
    return hasher.hexdigest()[:10]


def cached_collapse_cands(names_and_meshes: list, verbose: bool = True) -> dict:
    """{키: 1단계 후보}. 키는 메쉬 이름 — 중간 상태는 ``이름|적용한 병합`` 으로 부른다."""
    gae_paths.ensure_dirs()
    cache_path = os.path.join(gae_paths.CACHE_DIR, f"regen_c1_{feature_code_version()}.pkl")
    store = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as file_handle:
            store = pickle.load(file_handle)

    new_count = 0
    for key, mesh in names_and_meshes:
        if key in store:
            continue
        store[key] = collapse_candidates(mesh)
        new_count = new_count + 1
        if verbose and new_count % 50 == 0:
            print(f"    1단계 후보 계산 {new_count}개 ...", flush=True)

    if new_count > 0:
        # 임시 파일에 쓰고 바꿔 끼운다 — 학습 여러 개를 동시에 돌릴 때 반쯤 쓴 파일을
        # 다른 프로세스가 읽지 않도록
        temporary_path = f"{cache_path}.{os.getpid()}.tmp"
        with open(temporary_path, "wb") as file_handle:
            pickle.dump(store, file_handle)
        os.replace(temporary_path, cache_path)

    result = {}
    for key, mesh in names_and_meshes:
        result[key] = store[key]
    return result

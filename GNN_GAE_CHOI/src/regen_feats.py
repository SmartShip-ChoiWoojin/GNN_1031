"""재생성 모델의 입력 — 메쉬 그래프 + 두 단계의 후보와 후보 피처.

두 단계 모두 "후보를 전부 늘어놓고 각 후보에 점수를 매기는" 구조다.

  1단계 후보 = **절점 병합** (remove → keep)
      변으로 이어진 이웃과 사각형의 대각 모서리, **양방향** 전부. 어느 자리가 남느냐에
      따라 결과가 완전히 달라서 (a→b) 와 (b→a) 는 다른 후보다.
  2단계 후보 = **삼각형 쌍 병합**
      1단계를 적용한 메쉬에서 변을 공유하는 삼각형 쌍.

후보는 **실제로 적용할 수 있는 것만** 남긴다 (적용해 보고 거부되면 뺀다).

피처에는 **미리보기**가 들어간다 — 후보를 사본에 실제로 적용해 보고 주변이 어떻게
바뀌는지(나쁜 요소가 사라지나, 삼각형이 줄어드나, 각이 펴지나) 잰다. 사람이 고르는
기준이 그것이라서다.

GNN 쪽 입력은 부모 프로젝트의 **요소기반 그래프** 그대로다 (그래프 노드 = 쉘·1D 요소).
그래야 심판 분류기(`deploy_10_14`)의 인코더를 그대로 이어받을 수 있다. 절점 병합 후보는
절점 쌍이라, 절점마다 붙은 요소들의 임베딩을 모아(평균·최대) 절점 임베딩을 만든다
(``MeshGraph.incidence``).
"""

from __future__ import annotations

import numpy as np
import torch

import gae_paths  # noqa: F401  (부모 src 를 sys.path 에 넣는다)
import dataset as DS
import features as F
import graphs
from meshedit import EditableMesh, EditError, collapse_pairs, merge_candidates, shared_edge_of
from meshrep import Edge, MeshRep

#: 절점 하나를 설명하는 피처 (1단계 후보는 remove 쪽, keep 쪽에 하나씩 붙인다)
VERTEX_FEATS = ["ext", "deg_shell", "n_tri", "n_line", "n_free", "n_nonman",
                "angle_dev", "ang_min", "ang_max", "skew_max", "aspect_max", "n_bad"]

#: 주변(remove·keep 을 쓰는 요소들) 품질 요약 — 병합 전/후 둘 다 잰다
REGION_FEATS = ["ang_min", "ang_max", "skew_max", "aspect_max", "n_bad", "n_tri", "n_shell"]

#: 1단계 후보 피처 이름 (열 순서)
COLLAPSE_FEATS = ["is_diag", "len_rel", "lines_on_edge", "n_shared", "edge_free", "edge_nonman"]
for feature_name in VERTEX_FEATS:
    COLLAPSE_FEATS.append("r_" + feature_name)
for feature_name in VERTEX_FEATS:
    COLLAPSE_FEATS.append("k_" + feature_name)
for feature_name in REGION_FEATS:
    COLLAPSE_FEATS.append("pre_" + feature_name)
for feature_name in REGION_FEATS:
    COLLAPSE_FEATS.append("post_" + feature_name)
COLLAPSE_FEATS = COLLAPSE_FEATS + ["d_n_bad", "d_n_tri", "d_ang_min", "d_skew_max", "d_aspect_max",
                                   "n_removed", "n_lines_dropped", "area_rel"]

#: 2단계 후보 피처 이름 (열 순서)
MERGE_FEATS = [
    "dihedral", "q_skew", "q_aspect", "q_warp", "q_angle_min", "q_angle_max", "q_taper",
    "q_area_rel", "lines_on_edge", "tri_skew_max", "tri_aspect_max", "tri_angle_min",
    "tri_bad", "any_ext", "area_ratio", "n_compete",
]


# ──────────────────────────────────────────────────────────────────────────
# 메쉬 그래프
# ──────────────────────────────────────────────────────────────────────────
class MeshGraph:
    """GNN 입력 하나. 그래프 노드 순서 = 쉘(mesh.shells 순서) 다음 1D(mesh.lines 순서)."""

    def __init__(self, data, vertex_ids: list, incidence: torch.Tensor, n_shell: int):
        #: torch_geometric Data (정규화 끝)
        self.data = data
        #: 절점 번호 목록 (정렬) — 절점 인덱스의 기준
        self.vertex_ids = vertex_ids
        #: (2, nnz) [절점 인덱스, 그래프 노드(요소) 인덱스] — 절점마다 붙은 요소
        self.incidence = incidence
        self.n_shell = n_shell


def mesh_graph(mesh: EditableMesh, normalizer, graph_options: dict) -> MeshGraph:
    """메쉬 → GNN 입력. normalizer 는 심판 분류기의 정규화기, graph_options 는 그래프 옵션."""
    rep = MeshRep(mesh.to_raw())
    graph_arrays = graphs.build(rep, "elem", **graph_options)
    data = DS.to_pyg(graph_arrays, 0, mesh.name, normalizer)

    vertex_ids = sorted(mesh.coords)
    vertex_index_of = {}
    for vertex_index in range(len(vertex_ids)):
        vertex_index_of[vertex_ids[vertex_index]] = vertex_index

    n_shell = len(rep.shells)
    include_lines = graph_options.get("include_lines", True)
    vertex_rows = []
    element_columns = []
    for vertex_id in vertex_ids:
        for shell_index in rep.node_shells.get(vertex_id, ()):
            vertex_rows.append(vertex_index_of[vertex_id])
            element_columns.append(shell_index)
        if include_lines:
            for line_index in rep.node_lines.get(vertex_id, ()):
                vertex_rows.append(vertex_index_of[vertex_id])
                element_columns.append(n_shell + line_index)
    incidence = torch.tensor([vertex_rows, element_columns], dtype=torch.long)
    return MeshGraph(data, vertex_ids, incidence, n_shell)


# ──────────────────────────────────────────────────────────────────────────
# 1단계 — 절점 병합 후보
# ──────────────────────────────────────────────────────────────────────────
def region_summary(qualities: list) -> list:
    """요소 품질 목록 → REGION_FEATS 순서의 요약 7개.

    [최소 내각의 최소, 최대 내각의 최대, skew 최대, aspect 최대, 불량각 요소 수, 삼각형 수, 요소 수]
    """
    if len(qualities) == 0:
        return [90.0, 90.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    smallest_angle = qualities[0].angle_min
    largest_angle = qualities[0].angle_max
    largest_skew = qualities[0].skew
    largest_aspect = qualities[0].aspect
    bad_count = 0
    triangle_count = 0
    for quality in qualities:
        if quality.angle_min < smallest_angle:
            smallest_angle = quality.angle_min
        if quality.angle_max > largest_angle:
            largest_angle = quality.angle_max
        if quality.skew > largest_skew:
            largest_skew = quality.skew
        if quality.aspect > largest_aspect:
            largest_aspect = quality.aspect
        if F.is_bad_shell(quality):
            bad_count = bad_count + 1
        if not quality.is_quad:
            triangle_count = triangle_count + 1
    return [smallest_angle, largest_angle, largest_skew, largest_aspect,
            float(bad_count), float(triangle_count), float(len(qualities))]


def vertex_table(mesh: EditableMesh, rep: MeshRep) -> dict:
    """{절점: VERTEX_FEATS 순서의 피처 12개}."""
    table = {}
    for vertex_id in mesh.coords:
        shell_indices = rep.node_shells.get(vertex_id, [])
        qualities = []
        for shell_index in shell_indices:
            qualities.append(rep.shells[shell_index].quality)

        touching_edges = []
        for edge in rep.edge_shells:
            if vertex_id in edge:
                touching_edges.append(edge)
        free_edge_count = 0
        nonmanifold_edge_count = 0
        for edge in touching_edges:
            if rep.is_free_edge(edge):
                free_edge_count = free_edge_count + 1
            if rep.is_nonmanifold_edge(edge):
                nonmanifold_edge_count = nonmanifold_edge_count + 1

        triangle_count = 0
        for quality in qualities:
            if not quality.is_quad:
                triangle_count = triangle_count + 1

        if len(shell_indices) > 0:
            angle_deviation = abs(rep.node_angle_sum(vertex_id) - 360.0)
        else:
            angle_deviation = 0.0

        region = region_summary(qualities)
        table[vertex_id] = [
            float(mesh.ext.get(vertex_id, False)), float(len(shell_indices)),
            float(triangle_count), float(len(rep.node_lines.get(vertex_id, []))),
            float(free_edge_count), float(nonmanifold_edge_count),
            angle_deviation,
            region[0], region[1], region[2], region[3], region[4],
        ]
    return table


class CollapseCands:
    """1단계 후보 묶음."""

    def __init__(self, pairs: list | None = None, feats: np.ndarray | None = None):
        if pairs is None:
            pairs = []
        if feats is None:
            feats = np.zeros((0, len(COLLAPSE_FEATS)))
        #: [(remove, keep)]
        self.pairs = pairs
        #: (후보 수, 피처 수) 배열
        self.feats = feats


def shells_area(mesh: EditableMesh, shell_indices: list) -> float:
    """쉘 여러 개의 면적 합 (내장 sum() — 오차 보정 덧셈. for 문 덧셈과 마지막 자리가 다를 수 있다)."""
    areas = []
    for shell_index in shell_indices:
        areas.append(F.area_normal(mesh.corners(shell_index))[0])
    return sum(areas)


def collapse_candidates(mesh: EditableMesh) -> CollapseCands:
    """적용 가능한 절점 병합 후보 전부와 미리보기 피처."""
    rep = mesh.rep
    median_edge = rep.median_edge_length
    vertex_features = vertex_table(mesh, rep)

    candidate_pairs = []
    feature_rows = []
    for remove_node, keep_node, relation in collapse_pairs(mesh):
        # 사본에 실제로 적용해 본다 — 거부되면 후보가 아니다
        preview = mesh.clone()
        try:
            operation = preview.collapse_edge(remove_node, keep_node)
        except EditError:
            continue
        edge = Edge((remove_node, keep_node))

        # 병합 전: remove 나 keep 을 쓰는 요소
        before_indices = sorted(set(rep.node_shells.get(remove_node, []))
                                | set(rep.node_shells.get(keep_node, [])))
        before_qualities = []
        for shell_index in before_indices:
            before_qualities.append(rep.shells[shell_index].quality)

        # 병합 후: keep 을 쓰는 요소
        after_indices = []
        for shell_index in range(len(preview.shells)):
            if keep_node in preview.shells[shell_index][2]:
                after_indices.append(shell_index)
        after_qualities = []
        for shell_index in after_indices:
            after_qualities.append(preview.quality(shell_index))

        before_region = region_summary(before_qualities)
        after_region = region_summary(after_qualities)
        area_before = shells_area(mesh, before_indices)
        area_after = shells_area(preview, after_indices)

        is_diagonal = relation == "diag"
        is_edge = relation == "edge"
        if is_edge:
            lines_on_edge = float(len(rep.edge_lines.get(edge, ())))
        else:
            lines_on_edge = 0.0
        shared_shell_count = 0
        for shell_index in rep.node_shells.get(remove_node, []):
            if keep_node in rep.shells[shell_index].nodes:
                shared_shell_count = shared_shell_count + 1
        edge_is_free = is_edge and rep.is_free_edge(edge)
        edge_is_nonmanifold = is_edge and rep.is_nonmanifold_edge(edge)
        edge_length = float(np.linalg.norm(mesh.point(remove_node) - mesh.point(keep_node)))

        row = [float(is_diagonal), edge_length / (median_edge + 1e-9), lines_on_edge,
               float(shared_shell_count), float(edge_is_free), float(edge_is_nonmanifold)]
        row = row + vertex_features[remove_node] + vertex_features[keep_node] + before_region + after_region
        row = row + [
            after_region[4] - before_region[4],          # 불량각 요소 수 변화
            after_region[5] - before_region[5],          # 삼각형 수 변화
            after_region[0] - before_region[0],          # 최소 내각 변화
            after_region[2] - before_region[2],          # skew 최대 변화
            after_region[3] - before_region[3],          # aspect 최대 변화
            float(len(operation.info["removed_shells"])),
            float(len(operation.info["dropped_lines"])),
            (area_after - area_before) / (area_before + 1e-9),
        ]
        feature_rows.append(row)
        candidate_pairs.append((remove_node, keep_node))

    if len(feature_rows) > 0:
        features = np.array(feature_rows, dtype=np.float32)
    else:
        features = np.zeros((0, len(COLLAPSE_FEATS)), np.float32)
    return CollapseCands(candidate_pairs, features)


# ──────────────────────────────────────────────────────────────────────────
# 문맥 피처 — 후보 하나만 봐서는 모르는 것
# ──────────────────────────────────────────────────────────────────────────
# 후보 피처는 후보를 **혼자** 본다. 그런데 사람의 선택은 비교다:
#   - 이 변을 합친다면 어느 쪽 자리를 남기나 → 반대 방향 결과와 비교해야 안다
#   - 이 메쉬에서 제일 나쁜 곳이 여기인가 → 같은 메쉬의 다른 후보와 비교해야 안다
#   - 이 메쉬는 손댈 곳이 많은가 → 메쉬 전체 상태
# 이것들은 이미 계산한 후보 피처와 메쉬에서 싸게 나오므로, 비싼 1단계 캐시(미리보기)는
# 그대로 두고 학습/추론 때 덧붙인다.

#: 반대 방향 후보와 차이를 볼 피처
REVERSE_COMPARE_FEATS = ["post_ang_min", "post_skew_max", "post_aspect_max", "post_n_bad", "post_n_tri",
                         "area_rel", "n_removed"]
#: 메쉬 안 순위를 볼 피처 (1단계)
RANK_FEATS = ["d_n_bad", "d_ang_min", "d_skew_max", "len_rel", "pre_ang_min", "post_ang_min"]
#: 메쉬 상태 피처 — '멈춤' 점수의 입력이기도 하다
MESH_FEATS = ["m_n_shell", "m_n_tri", "m_n_bad", "m_ang_min", "m_n_cand"]

COLLAPSE_CTX_FEATS = ["rev_valid"]
for feature_name in REVERSE_COMPARE_FEATS:
    COLLAPSE_CTX_FEATS.append("rev_d_" + feature_name)
for feature_name in RANK_FEATS:
    COLLAPSE_CTX_FEATS.append("rank_" + feature_name)
COLLAPSE_CTX_FEATS = COLLAPSE_CTX_FEATS + MESH_FEATS

MERGE_CTX_FEATS = MESH_FEATS + ["rank_q_skew", "rank_q_angle_min", "rank_tri_angle_min"]


def mesh_state_row(mesh: EditableMesh, candidate_count: int) -> list:
    """MESH_FEATS 순서: [요소 수, 삼각형 수, 불량각 요소 수, 최소 내각, 후보 수]."""
    triangle_count = 0
    bad_count = 0
    smallest_angle = None
    for shell_index in range(len(mesh.shells)):
        quality = mesh.quality(shell_index)
        if not quality.is_quad:
            triangle_count = triangle_count + 1
        if F.is_bad_shell(quality):
            bad_count = bad_count + 1
        if smallest_angle is None or quality.angle_min < smallest_angle:
            smallest_angle = quality.angle_min
    if smallest_angle is None:
        smallest_angle = 90.0
    return [float(len(mesh.shells)), float(triangle_count), float(bad_count), smallest_angle, float(candidate_count)]


def rank_in_mesh(values: np.ndarray) -> np.ndarray:
    """메쉬 안에서의 순위를 0~1 로 (동률은 평균 순위)."""
    if len(values) <= 1:
        return np.zeros(len(values), np.float32)
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(len(values))
    for unique_value in np.unique(values):
        same_mask = values == unique_value
        if same_mask.sum() > 1:
            ranks[same_mask] = ranks[same_mask].mean()
    return (ranks / (len(values) - 1)).astype(np.float32)


def collapse_context(mesh: EditableMesh, candidates: CollapseCands) -> np.ndarray:
    """1단계 문맥 피처 (후보 수, len(COLLAPSE_CTX_FEATS))."""
    candidate_count = len(candidates.pairs)
    if candidate_count == 0:
        return np.zeros((0, len(COLLAPSE_CTX_FEATS)), np.float32)

    column_of = {}
    for column_index in range(len(COLLAPSE_FEATS)):
        column_of[COLLAPSE_FEATS[column_index]] = column_index
    features = candidates.feats
    position_of_pair = {}
    for candidate_index in range(candidate_count):
        position_of_pair[candidates.pairs[candidate_index]] = candidate_index

    reverse_part = np.zeros((candidate_count, 1 + len(REVERSE_COMPARE_FEATS)), np.float32)
    for candidate_index in range(candidate_count):
        remove_node, keep_node = candidates.pairs[candidate_index]
        reverse_index = position_of_pair.get((keep_node, remove_node))
        if reverse_index is None:
            continue
        reverse_part[candidate_index, 0] = 1.0
        differences = []
        for feature_name in REVERSE_COMPARE_FEATS:
            column = column_of[feature_name]
            differences.append(features[candidate_index, column] - features[reverse_index, column])
        reverse_part[candidate_index, 1:] = differences

    rank_columns = []
    for feature_name in RANK_FEATS:
        rank_columns.append(rank_in_mesh(features[:, column_of[feature_name]]))
    rank_part = np.stack(rank_columns, 1)

    mesh_part = np.tile(np.array(mesh_state_row(mesh, candidate_count), np.float32), (candidate_count, 1))
    return np.concatenate([reverse_part, rank_part, mesh_part], 1).astype(np.float32)


def merge_context(mesh: EditableMesh, candidates) -> np.ndarray:
    """2단계 문맥 피처 (후보 수, len(MERGE_CTX_FEATS))."""
    candidate_count = len(candidates.pairs)
    if candidate_count == 0:
        return np.zeros((0, len(MERGE_CTX_FEATS)), np.float32)

    column_of = {}
    for column_index in range(len(MERGE_FEATS)):
        column_of[MERGE_FEATS[column_index]] = column_index
    features = candidates.feats

    rank_columns = []
    for feature_name in ["q_skew", "q_angle_min", "tri_angle_min"]:
        rank_columns.append(rank_in_mesh(features[:, column_of[feature_name]]))
    rank_part = np.stack(rank_columns, 1)
    mesh_part = np.tile(np.array(mesh_state_row(mesh, candidate_count), np.float32), (candidate_count, 1))
    return np.concatenate([mesh_part, rank_part], 1).astype(np.float32)


def with_context(mesh: EditableMesh, candidates, stage: int):
    """후보 피처 뒤에 문맥 피처를 붙인 **새** 후보 묶음 (캐시 객체는 건드리지 않는다)."""
    if stage == 1:
        context = collapse_context(mesh, candidates)
        combined = np.concatenate([candidates.feats, context], 1).astype(np.float32)
        return CollapseCands(candidates.pairs, combined)

    context = merge_context(mesh, candidates)
    combined = np.concatenate([candidates.feats, context], 1).astype(np.float32)
    return MergeCands(candidates.pairs, candidates.edges, combined)


# ──────────────────────────────────────────────────────────────────────────
# 2단계 — 삼각형 쌍 병합 후보
# ──────────────────────────────────────────────────────────────────────────
class MergeCands:
    """2단계 후보 묶음."""

    def __init__(self, pairs: list | None = None, edges: list | None = None, feats: np.ndarray | None = None):
        if pairs is None:
            pairs = []
        if edges is None:
            edges = []
        if feats is None:
            feats = np.zeros((0, len(MERGE_FEATS)))
        #: [(쉘 인덱스, 쉘 인덱스)] — mesh.shells 인덱스 = 그래프 노드 인덱스
        self.pairs = pairs
        #: [공유 변] (절점 번호) — 쉘 인덱스는 병합할 때마다 밀리므로 적용은 이걸로 한다
        self.edges = edges
        #: (후보 수, 피처 수) 배열
        self.feats = feats


def merge_cands(mesh: EditableMesh) -> MergeCands:
    """적용 가능한 삼각형 쌍 병합 후보와 피처."""
    rep = mesh.rep
    median_edge = rep.median_edge_length

    usable_candidates = []
    for candidate in merge_candidates(mesh, drop_lines=True):
        if candidate.ok:
            usable_candidates.append(candidate)

    # 삼각형 하나가 후보 몇 개에 걸려 있나 (경쟁 정도)
    candidate_count_of_shell = {}
    for candidate in usable_candidates:
        for shell_index in (candidate.si, candidate.sj):
            if shell_index not in candidate_count_of_shell:
                candidate_count_of_shell[shell_index] = 0
            candidate_count_of_shell[shell_index] = candidate_count_of_shell[shell_index] + 1

    result = MergeCands()
    feature_rows = []
    for candidate in usable_candidates:
        merged_quality = candidate.quality
        quality_a = mesh.quality(candidate.si)
        quality_b = mesh.quality(candidate.sj)
        all_nodes = set(mesh.shells[candidate.si][2]) | set(mesh.shells[candidate.sj][2])
        touches_ext = False
        for node_id in all_nodes:
            if mesh.ext.get(node_id, False):
                touches_ext = True
        bad_triangle_count = F.is_bad_shell(quality_a) + F.is_bad_shell(quality_b)
        feature_rows.append([
            candidate.dihedral, merged_quality.skew, merged_quality.aspect, merged_quality.warp,
            merged_quality.angle_min, merged_quality.angle_max, merged_quality.taper,
            merged_quality.area / (median_edge * median_edge + 1e-9), float(candidate.lines_on_edge),
            max(quality_a.skew, quality_b.skew), max(quality_a.aspect, quality_b.aspect),
            min(quality_a.angle_min, quality_b.angle_min),
            float(bad_triangle_count),
            float(touches_ext),
            max(quality_a.area, quality_b.area) / (min(quality_a.area, quality_b.area) + 1e-9),
            float(candidate_count_of_shell[candidate.si] + candidate_count_of_shell[candidate.sj] - 2),
        ])
        result.pairs.append((candidate.si, candidate.sj))
        result.edges.append(shared_edge_of(mesh, candidate.si, candidate.sj))

    if len(feature_rows) > 0:
        result.feats = np.array(feature_rows, dtype=np.float32)
    else:
        result.feats = np.zeros((0, len(MERGE_FEATS)), np.float32)
    return result

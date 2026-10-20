"""재생성 모델의 입력 — 메쉬 그래프 + 두 단계의 후보와 후보 피처.

두 단계 모두 "후보를 전부 늘어놓고 각 후보에 점수를 매기는" 구조다.

  1단계 후보 = **절점 병합** (remove → keep)
      변으로 이어진 이웃과 사각형의 대각 모서리, **양방향** 전부. 어느 자리가 남느냐에
      따라 결과가 완전히 달라서 (a→b) 와 (b→a) 는 다른 후보다.
  2단계 후보 = **삼각형 쌍 병합**
      1단계를 적용한 메쉬에서 변을 공유하는 삼각형 쌍.

후보는 **실제로 적용할 수 있는 것만** 남긴다 (적용해 보고 거부되면 뺀다). 사람이 한 병합
중 단독으로는 거부되고 묶음으로만 되는 것이 1% 남짓 있어 그만큼은 못 맞힌다.

피처에는 **미리보기**가 들어간다 — 후보를 사본에 실제로 적용해 보고 주변이 어떻게
바뀌는지(나쁜 요소가 사라지나, 삼각형이 줄어드나, 각이 펴지나) 잰다. 사람이 고르는
기준이 그것이라서다. 학습 없이 이 피처만으로도 꽤 맞힌다는 건 `node_baseline.py` 가
보였다 (고친 자리 상위1 82%).

GNN 쪽 입력은 부모 프로젝트의 **요소기반 그래프** 그대로다 (노드 = 쉘·1D 요소). 그래야
심판 분류기(`deploy_10_14`)의 인코더를 그대로 이어받을 수 있다. 절점 병합 후보는 절점
쌍이라, 절점마다 붙은 요소들의 임베딩을 모아(평균·최대) 절점 임베딩을 만든다
(``vertex_incidence``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

import gae_paths  # noqa: F401
import dataset as DS
import features as F
import graphs
from meshedit import EditableMesh, EditError, apply_merges, collapse_pairs, merge_candidates  # noqa: F401
from meshrep import Edge, MeshRep

#: 절점 하나를 설명하는 피처 (1단계 후보는 remove 쪽, keep 쪽에 하나씩 붙인다)
VERTEX_FEATS = ["ext", "deg_shell", "n_tri", "n_line", "n_free", "n_nonman",
                "angle_dev", "ang_min", "ang_max", "skew_max", "aspect_max", "n_bad"]

#: 주변(remove·keep 을 쓰는 요소들) 품질 요약 — 병합 전/후 둘 다 잰다
REGION_FEATS = ["ang_min", "ang_max", "skew_max", "aspect_max", "n_bad", "n_tri", "n_shell"]

COLLAPSE_FEATS = (
    ["is_diag", "len_rel", "lines_on_edge", "n_shared", "edge_free", "edge_nonman"]
    + [f"r_{v}" for v in VERTEX_FEATS] + [f"k_{v}" for v in VERTEX_FEATS]
    + [f"pre_{v}" for v in REGION_FEATS] + [f"post_{v}" for v in REGION_FEATS]
    + ["d_n_bad", "d_n_tri", "d_ang_min", "d_skew_max", "d_aspect_max",
       "n_removed", "n_lines_dropped", "area_rel"]
)

MERGE_FEATS = [
    "dihedral", "q_skew", "q_aspect", "q_warp", "q_angle_min", "q_angle_max", "q_taper",
    "q_area_rel", "lines_on_edge", "tri_skew_max", "tri_aspect_max", "tri_angle_min",
    "tri_bad", "any_ext", "area_ratio", "n_compete",
]


# ──────────────────────────────────────────────────────────────────────────
# 메쉬 그래프
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class MeshGraph:
    """GNN 입력 하나. 요소 노드 순서 = 쉘(m.shells 순서) 다음 1D(m.lines 순서)."""

    data: object                 # torch_geometric Data (정규화 끝)
    vertex_ids: list[int]        # 절점 번호 — 절점 인덱스의 기준
    #: (2, nnz) [절점 인덱스, 요소 노드 인덱스] — 절점마다 붙은 요소
    inc: torch.Tensor
    n_shell: int


def mesh_graph(m: EditableMesh, norm, gopts: dict) -> MeshGraph:
    rep = MeshRep(m.to_raw())
    ga = graphs.build(rep, "elem", **gopts)
    data = DS.to_pyg(ga, 0, m.name, norm)
    vids = sorted(m.coords)
    vix = {v: i for i, v in enumerate(vids)}
    n_shell = len(rep.shells)
    with_lines = gopts.get("include_lines", True)
    rows, cols = [], []
    for v in vids:
        for si in rep.node_shells.get(v, ()):
            rows.append(vix[v]); cols.append(si)
        if with_lines:
            for li in rep.node_lines.get(v, ()):
                rows.append(vix[v]); cols.append(n_shell + li)
    inc = torch.tensor([rows, cols], dtype=torch.long)
    return MeshGraph(data, vids, inc, n_shell)


# ──────────────────────────────────────────────────────────────────────────
# 1단계 — 절점 병합 후보
# ──────────────────────────────────────────────────────────────────────────
def _region(qs: list[F.ShellQuality]) -> list[float]:
    if not qs:
        return [90.0, 90.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return [min(q.angle_min for q in qs), max(q.angle_max for q in qs),
            max(q.skew for q in qs), max(q.aspect for q in qs),
            float(sum(F.is_bad_shell(q) for q in qs)),
            float(sum(not q.is_quad for q in qs)), float(len(qs))]


def _vertex_table(m: EditableMesh, rep: MeshRep) -> dict[int, list[float]]:
    out = {}
    for v in m.coords:
        sis = rep.node_shells.get(v, [])
        qs = [rep.shells[s].quality for s in sis]
        edges = [e for e in rep.edge_shells if v in e]
        reg = _region(qs)
        out[v] = [
            float(m.ext.get(v, False)), float(len(sis)),
            float(sum(not q.is_quad for q in qs)), float(len(rep.node_lines.get(v, []))),
            float(sum(1 for e in edges if rep.is_free_edge(e))),
            float(sum(1 for e in edges if rep.is_nonmanifold_edge(e))),
            abs(rep.node_angle_sum(v) - 360.0) if sis else 0.0,
            reg[0], reg[1], reg[2], reg[3], reg[4],
        ]
    return out


@dataclass
class CollapseCands:
    pairs: list[tuple[int, int]] = field(default_factory=list)
    feats: np.ndarray = field(default_factory=lambda: np.zeros((0, len(COLLAPSE_FEATS))))


def collapse_candidates(m: EditableMesh) -> CollapseCands:
    """적용 가능한 절점 병합 후보 전부와 미리보기 피처."""
    rep = m.rep
    med = rep.median_edge_length
    vt = _vertex_table(m, rep)
    area_of = lambda mesh, sis: sum(F.area_normal(mesh.corners(s))[0] for s in sis)
    pairs, rows = [], []
    for r, k, rel in collapse_pairs(m):
        t = m.clone()
        try:
            op = t.collapse_edge(r, k)
        except EditError:
            continue
        e = Edge((r, k))
        pre_s = sorted(set(rep.node_shells.get(r, [])) | set(rep.node_shells.get(k, [])))
        pre_q = [rep.shells[s].quality for s in pre_s]
        post_s = [i for i, s in enumerate(t.shells) if k in s[2]]
        post_q = [t.quality(i) for i in post_s]
        pre, post = _region(pre_q), _region(post_q)
        a0 = area_of(m, pre_s)
        a1 = area_of(t, post_s)
        rows.append(
            [float(rel == "diag"),
             float(np.linalg.norm(m.P(r) - m.P(k))) / (med + 1e-9),
             float(len(rep.edge_lines.get(e, ()))) if rel == "edge" else 0.0,
             float(sum(1 for s in rep.node_shells.get(r, []) if k in rep.shells[s].nodes)),
             float(rel == "edge" and rep.is_free_edge(e)),
             float(rel == "edge" and rep.is_nonmanifold_edge(e))]
            + vt[r] + vt[k] + pre + post
            + [post[4] - pre[4], post[5] - pre[5], post[0] - pre[0], post[2] - pre[2],
               post[3] - pre[3], float(len(op.info["removed_shells"])),
               float(len(op.info["dropped_lines"])), (a1 - a0) / (a0 + 1e-9)]
        )
        pairs.append((r, k))
    feats = np.array(rows, dtype=np.float32) if rows else np.zeros((0, len(COLLAPSE_FEATS)), np.float32)
    return CollapseCands(pairs, feats)


# ──────────────────────────────────────────────────────────────────────────
# 문맥 피처 — 후보 하나만 봐서는 모르는 것
# ──────────────────────────────────────────────────────────────────────────
# 후보 피처는 후보를 **혼자** 본다. 그런데 사람의 선택은 비교다:
#   - 이 변을 합친다면 어느 쪽 자리를 남기나 → 반대 방향 결과와 비교해야 안다
#   - 이 메쉬에서 제일 나쁜 곳이 여기인가 → 같은 메쉬의 다른 후보와 비교해야 안다
#   - 이 메쉬는 손댈 곳이 많은가 → 메쉬 전체 상태
# 이것들은 이미 계산한 후보 피처와 메쉬에서 싸게 나오므로, 비싼 1단계 캐시(미리보기)는
# 그대로 두고 학습/추론 때 덧붙인다.

_REV = ["post_ang_min", "post_skew_max", "post_aspect_max", "post_n_bad", "post_n_tri",
        "area_rel", "n_removed"]
_RANK = ["d_n_bad", "d_ang_min", "d_skew_max", "len_rel", "pre_ang_min", "post_ang_min"]
MESH_FEATS = ["m_n_shell", "m_n_tri", "m_n_bad", "m_ang_min", "m_n_cand"]

COLLAPSE_CTX_FEATS = (["rev_valid"] + [f"rev_d_{f}" for f in _REV]
                      + [f"rank_{f}" for f in _RANK] + MESH_FEATS)
MERGE_CTX_FEATS = MESH_FEATS + ["rank_q_skew", "rank_q_angle_min", "rank_tri_angle_min"]


def _mesh_row(m: EditableMesh, n_cand: int) -> list[float]:
    qs = [m.quality(i) for i in range(len(m.shells))]
    return [float(len(qs)), float(sum(not q.is_quad for q in qs)),
            float(sum(F.is_bad_shell(q) for q in qs)),
            min((q.angle_min for q in qs), default=90.0), float(n_cand)]


def _ranks(v: np.ndarray) -> np.ndarray:
    """메쉬 안에서의 순위를 0~1 로 (동률은 평균 순위)."""
    if len(v) <= 1:
        return np.zeros(len(v), np.float32)
    order = v.argsort(kind="stable")
    r = np.empty(len(v), float)
    r[order] = np.arange(len(v))
    for x in np.unique(v):      # 동률 평균
        mask = v == x
        if mask.sum() > 1:
            r[mask] = r[mask].mean()
    return (r / (len(v) - 1)).astype(np.float32)


def collapse_context(m: EditableMesh, c: CollapseCands) -> np.ndarray:
    """1단계 문맥 피처 ``(n, len(COLLAPSE_CTX_FEATS))``."""
    n = len(c.pairs)
    if n == 0:
        return np.zeros((0, len(COLLAPSE_CTX_FEATS)), np.float32)
    ix = {f: i for i, f in enumerate(COLLAPSE_FEATS)}
    X = c.feats
    pos = {p: i for i, p in enumerate(c.pairs)}
    rev = np.zeros((n, 1 + len(_REV)), np.float32)
    for i, (r, k) in enumerate(c.pairs):
        j = pos.get((k, r))
        if j is None:
            continue
        rev[i, 0] = 1.0
        rev[i, 1:] = [X[i, ix[f]] - X[j, ix[f]] for f in _REV]
    rank = np.stack([_ranks(X[:, ix[f]]) for f in _RANK], 1)
    mesh = np.tile(np.array(_mesh_row(m, n), np.float32), (n, 1))
    return np.concatenate([rev, rank, mesh], 1).astype(np.float32)


def merge_context(m: EditableMesh, c: "MergeCands") -> np.ndarray:
    n = len(c.pairs)
    if n == 0:
        return np.zeros((0, len(MERGE_CTX_FEATS)), np.float32)
    ix = {f: i for i, f in enumerate(MERGE_FEATS)}
    X = c.feats
    rank = np.stack([_ranks(X[:, ix[f]]) for f in ("q_skew", "q_angle_min", "tri_angle_min")], 1)
    mesh = np.tile(np.array(_mesh_row(m, n), np.float32), (n, 1))
    return np.concatenate([mesh, rank], 1).astype(np.float32)


def with_context(m: EditableMesh, c, stage: int):
    """후보 피처 뒤에 문맥 피처를 붙인 **새** 후보 묶음 (캐시 객체는 건드리지 않는다)."""
    from dataclasses import replace

    ctx = collapse_context(m, c) if stage == 1 else merge_context(m, c)
    return replace(c, feats=np.concatenate([c.feats, ctx], 1).astype(np.float32))


# ──────────────────────────────────────────────────────────────────────────
# 2단계 — 삼각형 쌍 병합 후보
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class MergeCands:
    #: (si, sj) — m.shells 인덱스 = 요소 그래프 노드 인덱스
    pairs: list[tuple[int, int]] = field(default_factory=list)
    #: 공유 변 (절점 번호) — 쉘 인덱스는 병합할 때마다 밀리므로 적용은 이걸로 한다
    edges: list[frozenset] = field(default_factory=list)
    feats: np.ndarray = field(default_factory=lambda: np.zeros((0, len(MERGE_FEATS))))


def merge_cands(m: EditableMesh) -> MergeCands:
    """적용 가능한 삼각형 쌍 병합 후보와 피처."""
    rep = m.rep
    med = rep.median_edge_length
    cs = [c for c in merge_candidates(m, drop_lines=True) if c.ok]
    use = {}
    for c in cs:
        use[c.si] = use.get(c.si, 0) + 1
        use[c.sj] = use.get(c.sj, 0) + 1
    out = MergeCands()
    rows = []
    for c in cs:
        q = c.quality
        qi, qj = m.quality(c.si), m.quality(c.sj)
        nodes = set(m.shells[c.si][2]) | set(m.shells[c.sj][2])
        rows.append([
            c.dihedral, q.skew, q.aspect, q.warp, q.angle_min, q.angle_max, q.taper,
            q.area / (med * med + 1e-9), float(c.lines_on_edge),
            max(qi.skew, qj.skew), max(qi.aspect, qj.aspect), min(qi.angle_min, qj.angle_min),
            float(F.is_bad_shell(qi) + F.is_bad_shell(qj)),
            float(any(m.ext.get(n, False) for n in nodes)),
            max(qi.area, qj.area) / (min(qi.area, qj.area) + 1e-9),
            float(use[c.si] + use[c.sj] - 2),
        ])
        out.pairs.append((c.si, c.sj))
        out.edges.append(Edge(tuple(set(m.shells[c.si][2]) & set(m.shells[c.sj][2]))))
    out.feats = np.array(rows, dtype=np.float32) if rows else np.zeros((0, len(MERGE_FEATS)), np.float32)
    return out

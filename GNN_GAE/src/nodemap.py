"""사라진 절점이 어느 절점에 흡수됐는지 찾는다 (절점 병합 판정).

좌표로 판정하면 안 된다. 사람은 두 절점을 합친 뒤 그 자리를 다시 잡기도 하고, 셋을 한
점으로 모으기도 한다. 실측 실패 사례:

  - `_1081`: 22196 이 22197 을 흡수했는데 최종 위치는 22197 자리에서 45mm 떨어짐
  - `_0021`: 15278 + 15279 + 23439 셋이 하나로. 변으로 이어진 건 15279 인데
    최종 위치는 23439 자리

그래서 **연결관계**로 푼다. "사라진 쉘이 새 쉘이 될 때 그 자리를 누가 이어받았나" 를
반복적으로 좁히고, 마지막에 **검산**한다 — 찾은 대응을 원본 쉘에 적용했을 때 실제로
수정본 쉘이 나오는지. 검산이 통과한 쌍만 믿는다.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np

import gae_paths  # noqa: F401
from bdf_io import RawMesh

#: 좌표가 같다고 볼 허용오차(mm)
XYZ_TOL = 1e-3


@dataclass
class NodeMap:
    """원본 절점 → 수정본 절점 대응."""

    #: 흡수된 절점 -> 그 자리를 이어받은 절점
    absorbed: dict[int, int] = field(default_factory=dict)
    #: 아무도 이어받지 않고 그냥 없어진 절점
    dropped: set[int] = field(default_factory=set)
    #: 흡수했지만 이어받은 쪽이 **새로 만든 절점** 인 경우 (병합이 아니라 교체)
    replaced_by_new: dict[int, int] = field(default_factory=dict)
    #: 판정 근거. "쉘교체" = 사라진 쉘이 새 쉘로 바뀌며 한 자리만 달라 직접 읽은 것(강).
    #: "퇴화추론" = 쉘이 찌그러져 없어져 같은 쉘의 다른 절점 중에서 골라낸 것(약).
    basis: dict[int, str] = field(default_factory=dict)
    #: 검산 — 병합된 절점을 쓰던 쉘 중 대응과 맞아떨어지는 비율. 1.0 이어야 정상.
    explained: float = 0.0
    n_removed_shell: int = 0

    @property
    def survivors(self) -> dict[int, list[int]]:
        """살아남은 절점 -> 그가 흡수한 절점들 (1:N)."""
        out: dict[int, list[int]] = {}
        for b, a in self.absorbed.items():
            out.setdefault(a, []).append(b)
        return {k: sorted(v) for k, v in out.items()}


def _sets(mesh: RawMesh) -> list[frozenset]:
    return [frozenset(n) for _, _, n in mesh.shells]


def _key(p) -> tuple:
    """좌표를 허용오차 단위로 끊어 사전 키로 쓴다 (같은 자리인지 비교용)."""
    return tuple(np.round(np.asarray(p, dtype=float) / XYZ_TOL).astype(np.int64))


def build_node_map(before: RawMesh, after: RawMesh) -> NodeMap:
    """연결관계로 절점 대응을 푼다.

    번호가 바뀐 절점은 이 데이터셋에 없다(실측 0건)는 사실을 쓴다 — 양쪽에 같은 번호로
    있으면 같은 절점이다. 그래서 풀어야 할 것은 **사라진 절점의 행방**뿐이다.
    """
    nm = NodeMap()
    gone = set(before.coords) - set(after.coords)
    new = set(after.coords) - set(before.coords)

    bset, aset = set(_sets(before)), set(_sets(after))
    removed = [list(n) for _, _, n in before.shells if frozenset(n) not in aset]
    created = [set(n) for _, _, n in after.shells if frozenset(n) not in bset]
    nm.n_removed_shell = len(removed)
    if not gone:
        nm.explained = 1.0 if not removed else _explained(removed, created, {})
        return nm

    # 원래 변으로 이어져 있던 이웃. 병합은 거의 전부 이웃끼리 일어나므로(실측 92%)
    # 후보를 고를 때 가장 강한 제약이 된다.
    nbr: dict[int, set[int]] = {}
    for _, _, nodes in before.shells:
        for i, u in enumerate(nodes):
            v = nodes[(i + 1) % len(nodes)]
            nbr.setdefault(u, set()).add(v)
            nbr.setdefault(v, set()).add(u)

    phi: dict[int, int] = {}

    # 0순위: 좌표 증거. 사라진 B 의 자리에 **정확히** 들어앉은 절점이 있으면 그가 흡수자다.
    # 이건 쉘 모양이 어떻게 바뀌든 흔들리지 않는다. 실측 `_0032` 에서 쉘 투표가
    # 엉뚱한 절점(15339)을 가리켰는데, 실제 흡수자(26865)는 26866 자리로 정확히
    # 옮겨와 있었다 — 그 경우를 이 규칙이 잡는다.
    landed: dict[tuple, int] = {}
    for n, c in after.coords.items():
        landed.setdefault(_key(c), n)
    for b in sorted(gone):
        cand = landed.get(_key(before.coords[b]))
        if cand is not None and cand != b and cand in nbr.get(b, ()):
            phi[b] = cand
            nm.basis[b] = "좌표+인접"

    for _ in range(8):  # 보통 2~3 회에 수렴한다
        # 1순위: 사라진 쉘이 **새 쉘로 바뀐** 경우. 한 자리만 다르면 그 자리가 곧 대응이다.
        strong: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        # 2순위: 사라진 쉘이 **퇴화해 없어진** 경우. 두 절점이 하나로 합쳐지면 삼각형이
        # 찌그러져 사라지므로 바뀐 쉘이 안 생긴다. 이때는 "같은 쉘 안의 다른 절점 중
        # 하나로 합쳐졌다" 까지만 알 수 있어서, 여러 쉘의 표를 모아 교집합으로 좁힌다.
        # (실측 `_0021`: 15278+15279+23439 가 한 점으로 모이며 삼각형 2개가 퇴화)
        weak: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        for nodes in removed:
            unknown = [n for n in nodes if n in gone and n not in phi]
            if len(unknown) != 1:
                continue  # 모르는 자리가 둘 이상이면 이번 회차에는 판단 보류
            u = unknown[0]
            known = {phi.get(n, n) for n in nodes if n != u}
            hit = False
            for t in created:
                if len(t) == len(nodes) and known <= t:
                    rest = t - known
                    if len(rest) == 1:
                        c = next(iter(rest))
                        # 이웃이 아닌 후보는 표를 반만 준다. 쉘이 통째로 다시 깔리면
                        # 사라진 쉘과 새 쉘이 **우연히** 한 자리만 다를 수 있는데,
                        # 그때 엉뚱한 절점이 흡수자로 뽑힌다 (실측 `_0032`).
                        strong[u][c] += 2 if c in nbr.get(u, ()) else 1
                        hit = True
            if not hit:
                for k in known:
                    weak[u][k] += 2 if k in nbr.get(u, ()) else 1

        picked: dict[int, int] = {}
        basis: dict[int, str] = {}
        for b, c in strong.items():
            best = c.most_common(1)[0][0]
            picked[b] = best
            basis[b] = "쉘교체" if best in nbr.get(b, ()) else "쉘교체(비인접)"
        for b, c in weak.items():
            if b in picked:
                continue
            best = max(c.values())
            tied = sorted(k for k, v in c.items() if v == best)
            if len(tied) == 1:
                picked[b] = tied[0]
                basis[b] = "퇴화추론"
            else:
                # 동점이면 좌표로만 가른다 — 흡수된 절점 자리에 최종적으로 놓인 쪽
                bp = np.asarray(before.coords[b], dtype=float)
                exact = [k for k in tied if k in after.coords
                         and float(np.linalg.norm(np.asarray(after.coords[k], dtype=float) - bp)) < XYZ_TOL]
                if len(exact) == 1:
                    picked[b] = exact[0]
                    basis[b] = "퇴화추론+좌표"
        if not picked:
            break
        phi.update(picked)
        nm.basis.update(basis)

    for b, a in phi.items():
        if a in new:
            nm.replaced_by_new[b] = a
        else:
            nm.absorbed[b] = a
    nm.dropped = gone - set(phi)
    nm.explained = _explained(removed, created, phi)
    return nm


def _explained(removed: list[list[int]], created: list[set], phi: dict[int, int]) -> float:
    """검산 — **병합된 절점을 쓰던** 쉘이 대응과 모순되지 않는가.

    대응이 맞다면 그런 쉘은 매핑 후 (a) 새 쉘이 되거나 (b) 절점이 겹쳐 퇴화해 없어진다.
    둘 다 아니면 대응이 틀린 것이다.

    병합된 절점을 **안 쓰던** 쉘은 세지 않는다 — 그건 사람이 그냥 다시 깐 것이라
    절점 대응으로 설명될 이유가 없다. 예전에는 그것까지 실패로 세어 중앙값이 50%로
    나왔고, 그 숫자로는 대응이 맞는지 알 수 없었다.
    """
    pool = {frozenset(t) for t in created}
    rel = [n for n in removed if any(x in phi for x in n)]
    if not rel:
        return 1.0
    ok = 0
    for nodes in rel:
        mapped = [phi.get(n, n) for n in nodes]
        if len(set(mapped)) < len(mapped) or frozenset(mapped) in pool:
            ok += 1
    return ok / len(rel)


# ──────────────────────────────────────────────────────────────────────────
def where_survivor_landed(before: RawMesh, after: RawMesh, nm: NodeMap) -> dict[int, str]:
    """살아남은 절점이 최종적으로 어디에 자리잡았나.

    흡수한 절점 **전부** 와 자기 원래 자리를 모두 비교한다. 하나만 보고 판정하면
    `_0021`(셋이 하나로 합쳐진 경우)처럼 오판한다.
    """
    out: dict[int, str] = {}
    for a, absorbed in nm.survivors.items():
        if a not in after.coords or a not in before.coords:
            continue
        p1 = np.asarray(after.coords[a], dtype=float)
        if float(np.linalg.norm(p1 - np.asarray(before.coords[a], dtype=float))) < XYZ_TOL:
            out[a] = "제자리"
            continue
        hit = next(
            (b for b in absorbed
             if float(np.linalg.norm(p1 - np.asarray(before.coords[b], dtype=float))) < XYZ_TOL),
            None,
        )
        if hit is not None:
            out[a] = "흡수한 절점 자리"
            continue
        pts = [before.coords[a]] + [before.coords[b] for b in absorbed]
        mid = np.mean(np.asarray(pts, dtype=float), axis=0)
        out[a] = ("관련 절점들의 중점" if float(np.linalg.norm(p1 - mid)) < XYZ_TOL
                  else "그 밖의 위치")
    return out

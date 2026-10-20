"""사람의 수정을 '우리 연산의 순서' 로 다시 쓴다 — 재생성 모델의 정답과 상한.

재생성은 메쉬를 통째로 새로 그리지 않는다. 사람이 한 일을 분해하면 거의 전부가 두
연산이다 (실측):

    1단계  절점 병합 (collapse_edge)   사라진 절점 481개 중 451개
    2단계  삼각형 쌍 병합 (merge_tris)  '연결만 변경' 196건의 97%, 병합 뒤 남은 차이의 대부분

그래서 모델이 할 일은 "어느 변을 어느 쪽으로 합칠지" 와 "그 뒤 어느 삼각형 쌍을 합칠지"
를 고르는 것이다. 이 모듈은 사람의 결과로부터 그 정답 순서(대본)를 뽑고, **실제 연산
코드로 다시 돌려서** 사람 결과가 정확히 재현되는지 검사한다. 재현되는 비율이 곧 이
연산 체계로 도달할 수 있는 상한이다.

정답 대본을 만드는 규칙:
  - 절점 병합은 **그룹** 단위로 본다. 살아남은 절점과 그가 흡수한 절점들이 한 그룹이고,
    최종 좌표에 원래 서 있던 그룹원이 '자리가 남는 절점(keep)' 이다. 셋이 하나로 모인
    `_0021` 처럼 남은 번호의 자리가 아닌 곳으로 모이는 경우가 있어서, 쌍 단위로 방향을
    정하면 틀린다.
  - 병합 순서: 지금 이웃인 것부터. 다른 병합이 먼저 일어나야 이웃이 되는 경우가 있다.
  - 삼각형 병합: 병합 뒤 메쉬와 사람 결과를 **좌표**로 비교해, 사람 결과에만 있는
    사각형을 이루는 삼각형 두 개를 찾는다.

    python src/regen_oracle.py              # 508쌍 전부 — 상한과 실패 사유
    python src/regen_oracle.py --name SN2592_FWD_Clean_0021 -v
"""

from __future__ import annotations

import argparse
import collections
import os
from dataclasses import dataclass, field

import numpy as np

import gae_paths as gp
from meshdiff import XYZ_TOL, pkey, signature
from meshedit import EditableMesh, EditError, collapse_greedy
from meshrep import Edge


@dataclass
class Script:
    """쌍 하나의 정답 대본과 재현 결과."""

    name: str
    #: (remove, keep) — 원본 절점 번호. 적용 순서대로.
    collapses: list[tuple[int, int]] = field(default_factory=list)
    #: 병합 뒤 메쉬에서 합칠 삼각형 쌍의 공유 변 (절점 번호). 적용 순서대로.
    merges: list[frozenset] = field(default_factory=list)
    #: 연산이 거부된 것 (사유 포함) — 사람이 한 일인데 우리 연산이 못 따라간 것
    failed: list[str] = field(default_factory=list)
    #: 이 연산 체계 밖의 수정 (절점 추가 / 순수 삭제 / 좌표 이동 / 신규 교체)
    outside: list[str] = field(default_factory=list)
    exact: bool = False
    exact_shells: bool = False
    #: 재현 뒤에도 남은 차이 (요소 수)
    res_rm: int = 0
    res_cr: int = 0
    #: 1D 가 안 맞는가 (요소는 다 맞았는데)
    line_mismatch: int = 0

    @property
    def reason(self) -> str:
        if self.exact:
            return "재현됨"
        if self.outside:
            return self.outside[0]
        if self.failed:
            return "연산 거부"
        if self.exact_shells:
            return "1D 불일치"
        return "재배치(연산 밖)"


# ──────────────────────────────────────────────────────────────────────────
def collapse_groups(pair) -> list[tuple[int, list[int]]]:
    """[(자리가 남는 절점, [합쳐질 절점들])] — 사람 결과의 좌표로 방향을 정한다."""
    b, a = pair.before, pair.after
    out = []
    for surv, absorbed in pair.nodes.survivors.items():
        if surv not in a.coords:
            continue
        group = [surv] + list(absorbed)
        p = np.asarray(a.coords[surv], dtype=float)
        dist = {g: float(np.linalg.norm(np.asarray(b.coords[g], float) - p)) for g in group}
        keep = min(group, key=lambda g: (dist[g], g))
        out.append((keep, [g for g in group if g != keep]))
    return out


def apply_collapses(m: EditableMesh, todo: list[tuple[int, int]],
                    failed: list[str] | None = None) -> list[tuple[int, int]]:
    """(remove, keep) 들을 한 묶음으로 적용한다. 적용한 것을 순서대로 돌려준다.

    원본에서 이미 이웃인 쌍을 앞에 둔다 — 두 칸 떨어진 쌍(7건)은 앞의 병합으로 이웃이
    된 뒤에야 합칠 수 있다. 묶음이 통째로 거부되면 ``collapse_greedy`` 가 되는 만큼만 남긴다.
    """
    todo = sorted(todo, key=lambda rk: m.collapse_relation(*rk) is None)
    done, why = collapse_greedy(m, todo)
    if failed is not None:
        for rk, msg in why.items():
            failed.append(f"병합 {rk[0]}→{rk[1]}: {msg}")
    return done


def shared_edge(m: EditableMesh, si: int, sj: int) -> frozenset:
    return Edge(tuple(set(m.shells[si][2]) & set(m.shells[sj][2])))


def tri_pairs(m: EditableMesh) -> list[tuple[int, int]]:
    """변을 공유하는 삼각형 쌍 (막힌 것 포함, 유효성은 보지 않는다)."""
    rep = m.rep
    out = []
    for e, v in rep.edge_shells.items():
        if len(v) == 2 and len(m.shells[v[0]][2]) == 3 and len(m.shells[v[1]][2]) == 3:
            out.append((v[0], v[1]))
    return out


def oracle_merges(m: EditableMesh, good, failed: list[str] | None = None) -> list[frozenset]:
    """사람 결과에만 있는 사각형을 이루는 삼각형 쌍을 찾아 합친다. 공유 변 목록을 돌려준다."""
    gkeys = {frozenset(pkey(good.coords[n]) for n in s[2]) for s in good.shells}
    done: list[frozenset] = []
    tried: set = set()
    while True:
        hit = None
        for si, sj in tri_pairs(m):
            u = frozenset(pkey(m.coords[n]) for n in set(m.shells[si][2]) | set(m.shells[sj][2]))
            if len(u) == 4 and u in gkeys:
                e = shared_edge(m, si, sj)
                if e in tried:
                    continue
                hit = (si, sj, e)
                break
        if hit is None:
            return done
        si, sj, e = hit
        tried.add(e)
        try:
            m.merge_tris(si, sj, drop_lines=True)
        except EditError as exc:
            if failed is not None:
                failed.append(f"삼각형 병합 {sorted(e)}: {exc}")
            continue
        done.append(e)


def build_script(pair) -> tuple[Script, EditableMesh, EditableMesh]:
    """(대본, 1단계 뒤 메쉬, 최종 메쉬). 메쉬들은 화면·학습에서 그대로 쓴다."""
    s = Script(pair.name)
    n = pair.nodes
    if n.added - set(n.replaced_by_new.values()):
        s.outside.append("절점 추가")
    if n.replaced_by_new:
        s.outside.append("신규 절점으로 교체")
    if n.deleted:
        s.outside.append("순수 삭제")
    if n.moved:
        s.outside.append("좌표 이동")

    m = EditableMesh(pair.before)
    todo = [(r, k) for k, rs in collapse_groups(pair) for r in rs]
    s.collapses = apply_collapses(m, todo, s.failed)
    m1 = m.clone()
    s.merges = oracle_merges(m, pair.after, s.failed)

    sp, sg = signature(m), signature(pair.after)
    s.exact = sp == sg
    s.exact_shells = sp.shells == sg.shells
    s.res_rm = len(sp.shells - sg.shells)
    s.res_cr = len(sg.shells - sp.shells)
    s.line_mismatch = len(sp.lines ^ sg.lines)
    return s, m1, m


def build_all(names=None, verbose: bool = False) -> list[Script]:
    from bdf_io import discover_pairs
    from pairs import all_pairs, discover_good_pairs, load_pair

    good = discover_good_pairs()
    orig, _, _ = discover_pairs(gp.FE_DIR)
    out = []
    for i, nm in enumerate(names or all_pairs(), 1):
        s, _, _ = build_script(load_pair(nm, good, orig))
        out.append(s)
        if verbose and i % 100 == 0:
            print(f"  ... {i}")
    return out


def to_frame(scripts: list[Script]):
    import pandas as pd

    return pd.DataFrame([dict(
        name=s.name, exact=s.exact, exact_shells=s.exact_shells, reason=s.reason,
        n_collapse=len(s.collapses), n_merge=len(s.merges), n_failed=len(s.failed),
        res_rm=s.res_rm, res_cr=s.res_cr, line_mismatch=s.line_mismatch,
        outside=";".join(s.outside), failed=" | ".join(s.failed),
        collapses=";".join(f"{r}>{k}" for r, k in s.collapses),
        merges=";".join("-".join(map(str, sorted(e))) for e in s.merges),
    ) for s in scripts])


def main() -> None:
    ap = argparse.ArgumentParser(description="사람 수정 → 연산 대본, 재현 상한")
    ap.add_argument("--name", help="한 쌍만")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.name:
        s = build_all([a.name])[0]
        print(f"{s.name}: {s.reason}")
        print(f"  절점 병합 {len(s.collapses)}: {s.collapses}")
        print(f"  삼각형 병합 {len(s.merges)}: {[sorted(e) for e in s.merges]}")
        for f in s.failed:
            print(f"  거부: {f}")
        print(f"  남은 차이: 요소 -{s.res_rm} +{s.res_cr}, 1D {s.line_mismatch}")
        return

    print("508쌍의 정답 대본을 만들고 재현하는 중...")
    S = build_all(verbose=True)
    d = to_frame(S)
    n = len(d)
    print(f"\n정확히 재현 (요소+1D+좌표): {int(d.exact.sum())}/{n} ({d.exact.mean():.1%})")
    print(f"요소만 재현:               {int(d.exact_shells.sum())}/{n} ({d.exact_shells.mean():.1%})")
    print("\n재현 못 한 사유:")
    for k, v in collections.Counter(d.reason).most_common():
        print(f"  {k:16s} {v}")
    print(f"\n연산 수 — 절점 병합 {int(d.n_collapse.sum())} / 삼각형 병합 {int(d.n_merge.sum())} / "
          f"거부 {int(d.n_failed.sum())}")
    fails = [f for s in S for f in s.failed]
    if fails:
        print("\n거부된 연산 (사람은 했는데 우리 연산이 막은 것):")
        cnt = collections.Counter(f.split(": ", 1)[1].split(" (")[0] for f in fails)
        for k, v in cnt.most_common(10):
            print(f"  {v:4d}  {k}")

    gp.ensure_dirs()
    out = os.path.join(gp.OUT_DIR, "regen_oracle.csv")
    d.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()

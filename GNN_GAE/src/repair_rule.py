"""Stage 0 — 규칙 기반 수리기. GNN 이 넘어야 할 선.

분류 프로젝트에서 tabular 기준선을 먼저 세웠던 것과 같은 규율이다. 학습 모델을 만들기
전에 "규칙만으로 어디까지 되는가" 를 확정해 둔다.

데이터가 말하는 수리의 정의:
  - 불량 요소(내각<30° 또는 >150°)의 **95%가 삼각형**이다. 사각형은 거의 안 망가진다.
  - 삼각형 비율 하나의 라벨 AUC 가 0.914 (최종 GNN 0.9605). 즉 수리의 본체는
    **삼각형을 사각형으로 되돌리는 것**이고, 절점 이동은 마무리 공정이다.

여기 구현한 세 가지 전략:
  1. ``greedy_merge``        — 합쳐서 멀쩡한 사각형이 되는 인접 삼각형 쌍을 좋은 것부터
  2. ``swap_assisted_merge`` — 바로는 못 합치지만 대각선을 한 번 뒤집으면 합쳐지는 쌍까지
                               (greedy 가 못 보는 2수 앞. 학습 정책이 이길 여지가 여기다)
  3. ``repair``              — 위 둘을 수렴할 때까지
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gae_paths as gp  # noqa: F401  (부모 src 를 sys.path 에 넣는다)
import features as F
from meshedit import EditableMesh, EditError, MergeCand, Op, check, merge_candidates


@dataclass
class RepairConfig:
    """병합을 허용할 조건. 전부 '합쳐도 되는가' 의 도메인 판단이다."""

    #: 두 삼각형 사이 이면각(도). 이보다 크면 서로 다른 패널이라 합치면 형상이 바뀐다.
    max_dihedral: float = 10.0
    #: 결과 사각형이 이보다 찌그러지면 합치지 않는다 (90°에서의 이탈)
    max_skew: float = 45.0
    #: 결과 사각형의 최장변/최단변 상한
    max_aspect: float = 4.0
    #: 결과 사각형이 불량각(내각<30° 또는 >150°)이면 합치지 않는다
    reject_bad_angle: bool = True
    #: swap 을 거친 뒤 합치는 2수 전략을 쓸지
    use_swap: bool = True
    #: 수렴 반복 상한
    max_rounds: int = 8


def acceptable(c: MergeCand, cfg: RepairConfig) -> bool:
    """이 병합을 해도 되는가."""
    if not c.ok or c.quality is None:
        return False
    if c.dihedral > cfg.max_dihedral:
        return False
    q = c.quality
    if q.skew > cfg.max_skew or q.aspect > cfg.max_aspect:
        return False
    if cfg.reject_bad_angle and F.is_bad_shell(q):
        return False
    return True


def merge_score(c: MergeCand) -> float:
    """작을수록 먼저 합친다. 결과 사각형이 직사각형에 가까울수록 좋다."""
    q = c.quality
    assert q is not None
    return q.skew + 10.0 * max(0.0, q.aspect - 1.0) + 2.0 * c.dihedral


def greedy_merge(m: EditableMesh, cfg: RepairConfig) -> list[Op]:
    """합칠 수 있는 삼각형 쌍을 점수 좋은 순으로, 겹치지 않게 합친다.

    한 라운드 안에서 쉘 인덱스가 밀리므로 **절점 집합으로** 대상을 기억했다가 매번
    다시 찾는다 — 인덱스를 캐시하면 조용히 엉뚱한 요소를 합치게 된다.
    """
    ops: list[Op] = []
    for _ in range(cfg.max_rounds):
        cands = [c for c in merge_candidates(m) if acceptable(c, cfg)]
        if not cands:
            break
        cands.sort(key=merge_score)
        # 이번 라운드에 합칠 쌍을 절점 집합으로 예약 (겹치는 삼각형은 한 번만)
        used: set[int] = set()
        picked: list[tuple[frozenset, frozenset]] = []
        for c in cands:
            if c.si in used or c.sj in used:
                continue
            used |= {c.si, c.sj}
            picked.append((frozenset(m.shells[c.si][2]), frozenset(m.shells[c.sj][2])))
        if not picked:
            break
        done = 0
        for ka, kb in picked:
            si = _find(m, ka)
            sj = _find(m, kb)
            if si is None or sj is None:
                continue
            try:
                ops.append(m.merge_tris(si, sj))
                done += 1
            except EditError:
                continue
        if done == 0:
            break
    return ops


def swap_assisted_merge(m: EditableMesh, cfg: RepairConfig) -> list[Op]:
    """바로는 못 합치는 삼각형 쌍을, 대각선을 한 번 뒤집어 합칠 수 있게 만든다.

    greedy 가 놓치는 2수 앞이다. 뒤집기 자체는 삼각형 수를 안 줄이므로, **뒤집은 뒤
    실제로 병합이 성사될 때만** 확정하고 아니면 되돌린다.
    """
    ops: list[Op] = []
    for _ in range(cfg.max_rounds):
        gained = False
        cands = merge_candidates(m)
        # 지금은 못 합치는 쌍만 본다
        for c in sorted(cands, key=lambda x: x.dihedral):
            if acceptable(c, cfg) or c.blocked is not None:
                continue
            if c.dihedral > cfg.max_dihedral:
                continue
            ka, kb = frozenset(m.shells[c.si][2]), frozenset(m.shells[c.sj][2])
            trial = m.clone()
            si, sj = _find(trial, ka), _find(trial, kb)
            if si is None or sj is None:
                continue
            try:
                trial.swap_edge(si, sj)
            except EditError:
                continue
            after = [x for x in merge_candidates(trial) if acceptable(x, cfg)]
            if not after:
                continue
            after.sort(key=merge_score)
            best = after[0]
            try:
                trial.merge_tris(best.si, best.sj)
            except EditError:
                continue
            if check(trial, m).ok:
                ops.extend(trial.log[len(m.log):])
                m.shells, m.coords, m.lines, m.log = (
                    trial.shells, trial.coords, trial.lines, trial.log)
                m._dirty()
                gained = True
                break
        if not gained:
            break
    return ops


def _find(m: EditableMesh, key: frozenset) -> int | None:
    """절점 집합으로 쉘 인덱스를 찾는다 (편집하면 인덱스가 밀리므로)."""
    for i, s in enumerate(m.shells):
        if frozenset(s[2]) == key:
            return i
    return None


def repair(m: EditableMesh, cfg: RepairConfig | None = None) -> tuple[EditableMesh, list[Op]]:
    """규칙 기반 수리 전체. 원본은 건드리지 않고 수리된 사본을 돌려준다."""
    cfg = cfg or RepairConfig()
    out = m.clone()
    ops = greedy_merge(out, cfg)
    if cfg.use_swap:
        ops += swap_assisted_merge(out, cfg)
        ops += greedy_merge(out, cfg)  # swap 이 열어준 새 후보를 다시 훑는다
    return out, ops


# ──────────────────────────────────────────────────────────────────────────
# 진단 — 규칙으로 어디까지 되는가
# ──────────────────────────────────────────────────────────────────────────
def diagnose(names: list[str], pairs: dict, cfg: RepairConfig) -> dict:
    """비정상 메쉬 묶음에 규칙 수리기를 돌리고 상한을 잰다."""
    import numpy as np

    tri_before = tri_after = n_shell = 0
    bad_before = bad_after = 0
    blocked_1d = blocked_other = 0
    n_pairs = n_far = 0
    viol = 0
    per_mesh = []

    for nm in names:
        b, j = pairs[nm]
        m = EditableMesh.load(b, j)
        cands = merge_candidates(m)
        n_pairs += len(cands)
        n_far += sum(1 for c in cands if c.dihedral > cfg.max_dihedral)
        for c in cands:
            if c.blocked is None:
                continue
            if "1D" in c.blocked:
                blocked_1d += 1
            else:
                blocked_other += 1

        t0 = len(m.tri_indices())
        b0 = sum(1 for i in range(len(m.shells)) if F.is_bad_shell(m.quality(i)))
        fixed, _ = repair(m, cfg)
        rep = check(fixed, m)
        if not rep.ok:
            viol += 1
        t1 = len(fixed.tri_indices())
        b1 = sum(1 for i in range(len(fixed.shells)) if F.is_bad_shell(fixed.quality(i)))

        tri_before += t0
        tri_after += t1
        bad_before += b0
        bad_after += b1
        n_shell += len(m.shells)
        per_mesh.append((t0 - t1) / max(1, t0))

    return dict(
        n_mesh=len(names),
        tri_before=tri_before,
        tri_after=tri_after,
        tri_removed_frac=(tri_before - tri_after) / max(1, tri_before),
        bad_before=bad_before,
        bad_after=bad_after,
        bad_removed_frac=(bad_before - bad_after) / max(1, bad_before),
        tri_frac_before=tri_before / max(1, n_shell),
        pair_total=n_pairs,
        pair_blocked_1d=blocked_1d,
        pair_blocked_other=blocked_other,
        pair_far=n_far,
        per_mesh_median=float(np.median(per_mesh)),
        violations=viol,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="규칙 기반 수리기 진단")
    ap.add_argument("--n", type=int, default=150, help="검사할 비정상 메쉬 수")
    ap.add_argument("--cls", type=int, default=1, help="1=비정상, 0=정상")
    ap.add_argument("--no-swap", action="store_true")
    ap.add_argument("--max-dihedral", type=float, default=10.0)
    args = ap.parse_args()

    from bdf_io import discover_pairs
    from labels import load_labels

    pairs, _, _ = discover_pairs(gp.FE_DIR)
    lab = load_labels(gp.LABEL_XLSX)
    names = [n for n in sorted(lab) if n in pairs and lab[n].cls == args.cls][: args.n]

    cfg = RepairConfig(use_swap=not args.no_swap, max_dihedral=args.max_dihedral)
    r = diagnose(names, pairs, cfg)

    tag = "비정상" if args.cls == 1 else "정상"
    print(f"\n{tag} 메쉬 {r['n_mesh']}개 — 규칙 기반 수리기 (swap {'사용' if cfg.use_swap else '미사용'})")
    print(f"  삼각형        {r['tri_before']} → {r['tri_after']}  "
          f"({r['tri_removed_frac']:.1%} 제거, 메쉬별 중앙값 {r['per_mesh_median']:.1%})")
    print(f"  불량각 요소   {r['bad_before']} → {r['bad_after']}  ({r['bad_removed_frac']:.1%} 제거)")
    print(f"  삼각형 비율   {r['tri_frac_before']:.3f} → "
          f"{(r['tri_frac_before'] * r['tri_after'] / max(1, r['tri_before'])):.3f}")
    print(f"  인접 삼각형 쌍 {r['pair_total']}개 중")
    print(f"    1D 요소가 막음      {r['pair_blocked_1d']} ({r['pair_blocked_1d']/max(1,r['pair_total']):.1%})")
    print(f"    기하적으로 불가     {r['pair_blocked_other']}")
    print(f"    다른 패널(이면각 큼) {r['pair_far']} ({r['pair_far']/max(1,r['pair_total']):.1%})")
    print(f"  불변식 위반 메쉬: {r['violations']}개")


if __name__ == "__main__":
    main()

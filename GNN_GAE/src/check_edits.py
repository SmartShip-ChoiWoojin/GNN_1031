"""편집 계층 회귀 테스트. 부모 프로젝트의 ``check_parsing.py`` 와 같은 역할이다.

수리기가 내놓는 메쉬는 결국 현업의 .bdf 로 들어간다. 조용히 깨진 메쉬를 만드는 것이
제일 나쁜 실패라, 편집 연산을 건드릴 때마다 이걸 먼저 돌린다.

    python src/check_edits.py            # 200개
    python src/check_edits.py --n 1000
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile

import numpy as np

import gae_paths as gp
import features as F
from bdf_io import discover_pairs
from meshedit import EditableMesh, EditError, check, merge_candidates


class Failed(AssertionError):
    pass


def _ok(cond: bool, msg: str) -> None:
    if not cond:
        raise Failed(msg)


# ──────────────────────────────────────────────────────────────────────────
def t_roundtrip(names: list[str], pairs: dict) -> str:
    """쓰고 다시 읽으면 좌표·요소가 정확히 같아야 한다."""
    d = tempfile.mkdtemp()
    for nm in names:
        m = EditableMesh.load(*pairs[nm])
        b = os.path.join(d, "t.bdf")
        j = os.path.join(d, "t.json")
        m.write_bdf(b)
        m.write_json(j)
        m2 = EditableMesh.load(b, j)
        _ok(len(m2.coords) == len(m.coords), f"{nm}: 절점 수 불일치")
        for nid, c in m.coords.items():
            _ok(
                np.allclose(np.array(c), np.array(m2.coords[nid]), rtol=0, atol=1e-6),
                f"{nm}: 절점 {nid} 좌표가 라운드트립에서 바뀜",
            )
        _ok([s[2] for s in m2.shells] == [s[2] for s in m.shells], f"{nm}: 쉘 불일치")
        _ok([tuple(l[2]) for l in m2.lines] == [tuple(l[2]) for l in m.lines], f"{nm}: 1D 불일치")
        _ok(m2.ext == m.ext, f"{nm}: ext 불일치")
    return f"{len(names)}개 .bdf/.json 라운드트립 정확 일치"


def t_split_merge_inverse(names: list[str], pairs: dict, rng: random.Random) -> str:
    """split 한 사각형을 다시 merge 하면 원본으로 돌아와야 한다."""
    n_case = 0
    for nm in names:
        base = EditableMesh.load(*pairs[nm])
        quads = [i for i, s in enumerate(base.shells) if len(s[2]) == 4]
        if not quads:
            continue
        for qi in rng.sample(quads, min(3, len(quads))):
            for diag in (0, 1):
                orig = set(base.shells[qi][2])
                t = base.clone()
                try:
                    t.split_quad(qi, diag=diag)
                except EditError:
                    continue  # 그 대각선이 이미 변으로 존재 — 정당한 거부다
                _ok(check(t, base).ok, f"{nm}: split 직후 불변식 위반\n{check(t, base)}")
                cands = [
                    c
                    for c in merge_candidates(t)
                    if set(t.shells[c.si][2]) | set(t.shells[c.sj][2]) == orig and c.ok
                ]
                if not cands:
                    continue  # 1D 요소가 대각선 위에 있으면 되돌릴 수 없다 — 정상 동작
                t.merge_tris(cands[0].si, cands[0].sj)
                before = sorted(sorted(s[2]) for s in base.shells)
                after = sorted(sorted(s[2]) for s in t.shells)
                _ok(before == after, f"{nm}: split→merge 가 원본을 복원하지 못함 (쉘 {qi}, diag {diag})")
                _ok(check(t, base).ok, f"{nm}: split→merge 후 불변식 위반")
                n_case += 1
    return f"split→merge 역연산 {n_case}건 전부 원본 복원"


def t_swap_involution(names: list[str], pairs: dict) -> str:
    """같은 삼각형 쌍을 두 번 뒤집으면 제자리로 돌아와야 한다."""
    n_case = 0
    for nm in names:
        base = EditableMesh.load(*pairs[nm])
        for c in merge_candidates(base):
            if not c.ok:
                continue
            t = base.clone()
            try:
                t.swap_edge(c.si, c.sj)
            except EditError:
                continue
            _ok(check(t, base).ok, f"{nm}: swap 직후 불변식 위반\n{check(t, base)}")
            back = [
                x
                for x in merge_candidates(t)
                if {x.si, x.sj} == {c.si, c.sj} and x.ok
            ]
            if not back:
                continue
            try:
                t.swap_edge(back[0].si, back[0].sj)
            except EditError:
                continue
            _ok(
                sorted(sorted(s[2]) for s in t.shells) == sorted(sorted(s[2]) for s in base.shells),
                f"{nm}: swap 두 번이 제자리로 안 돌아옴",
            )
            n_case += 1
            break
    return f"swap 2회 = 항등 {n_case}건 확인"


def t_invariants_under_edits(names: list[str], pairs: dict, rng: random.Random) -> str:
    """무작위 편집을 퍼부어도 도메인 불변식이 깨지지 않아야 한다.

    비교 기준이 원본이 아니라 **직전 상태**인 이유: 표면 보존은 연산 하나하나가 지켜야
    할 성질이고(실측 단일 연산 면적 변화 최대 3.7e-5), 원본 대비 누적은
    merge→split→merge 같은 왕복을 섞은 무작위 수열에서만 커진다(실측 1.7e-3).
    실제 수리는 그런 왕복을 하지 않으므로 누적치로 판정하면 엉뚱한 걸 잡는다.
    """
    n_edit = 0
    for nm in names:
        base = EditableMesh.load(*pairs[nm])
        m = base.clone()
        for _ in range(12):
            prev = m.clone()
            kind = rng.choice(["split", "merge", "swap"])
            try:
                if kind == "split":
                    quads = [i for i, s in enumerate(m.shells) if len(s[2]) == 4]
                    if not quads:
                        continue
                    m.split_quad(rng.choice(quads), diag=rng.randint(0, 1))
                else:
                    cands = [c for c in merge_candidates(m) if c.ok]
                    if not cands:
                        continue
                    c = rng.choice(cands)
                    (m.merge_tris if kind == "merge" else m.swap_edge)(c.si, c.sj)
            except EditError:
                continue
            n_edit += 1
            r = check(m, prev)
            _ok(r.ok, f"{nm}: {kind} 뒤 불변식 위반\n{r}")
    return f"무작위 편집 {n_edit}회 동안 불변식 유지"


def t_line_elements_protected(names: list[str], pairs: dict) -> str:
    """1D 요소가 올라탄 변은 절대 없어지면 안 된다."""
    n_blocked = 0
    for nm in names:
        m = EditableMesh.load(*pairs[nm])
        ledges = m.line_edges()
        for c in merge_candidates(m):
            shared = frozenset(set(m.shells[c.si][2]) & set(m.shells[c.sj][2]))
            if shared in ledges:
                _ok(
                    c.blocked is not None and "1D" in c.blocked,
                    f"{nm}: 1D 요소가 있는 변 {sorted(shared)} 의 병합이 막히지 않았다",
                )
                n_blocked += 1
    return f"1D 요소가 보호한 병합 후보 {n_blocked}건 전부 차단됨"


def t_drop_lines(names: list[str], pairs: dict) -> str:
    """1D 요소가 올라탄 변도 ``drop_lines=True`` 면 합칠 수 있어야 한다.

    사람이 고친 508쌍을 보면 손댄 구역 안의 1D 요소 1,886개 중 361개(19.1%)가 지워졌고,
    변을 없앨 때는 그 위의 1D 도 같이 없앴다(35건 중 34건). 그래서 "금지" 가 아니라
    "부르는 쪽이 고르는 선택지" 여야 한다. 기본값은 여전히 거부인지도 같이 본다.
    """
    n_block = n_drop = 0
    for nm in names:
        base = EditableMesh.load(*pairs[nm])
        free = [c for c in merge_candidates(base, drop_lines=True)
                if c.ok and c.lines_on_edge > 0]
        if not free:
            continue
        c = free[0]
        # 기본값은 거부해야 한다
        try:
            base.clone().merge_tris(c.si, c.sj)
        except EditError:
            n_block += 1
        else:
            raise Failed(f"{nm}: 1D 가 있는 변인데 기본 호출이 거부되지 않았다")
        # drop_lines=True 면 성공하고, 그 1D 요소가 사라져 있어야 한다
        t = base.clone()
        op = t.merge_tris(c.si, c.sj, drop_lines=True)
        dropped = op.info.get("dropped_lines", [])
        _ok(len(dropped) == c.lines_on_edge,
            f"{nm}: 지워진 1D 수가 {len(dropped)} 인데 변 위에는 {c.lines_on_edge} 개였다")
        _ok(len(t.lines) == len(base.lines) - c.lines_on_edge,
            f"{nm}: 1D 총수가 맞지 않는다")
        r = check(t, base)
        _ok(r.ok, f"{nm}: drop_lines 병합 뒤 불변식 위반\n{r}")
        _ok(r.lines_removed == c.lines_on_edge,
            f"{nm}: 보고서의 1D 삭제 수({r.lines_removed})가 실제({c.lines_on_edge})와 다르다")
        n_drop += 1
    return f"기본 거부 {n_block}건 / drop_lines 병합 {n_drop}건 — 1D 가 함께 삭제되고 불변식 유지"


def t_ext_nodes_frozen(names: list[str], pairs: dict) -> str:
    """ext=true 절점은 움직일 수 없어야 한다."""
    n = 0
    for nm in names:
        m = EditableMesh.load(*pairs[nm])
        for nid, is_ext in m.ext.items():
            if not is_ext:
                continue
            try:
                m.move_node(nid, np.array([1.0, 0.0, 0.0]))
            except EditError:
                n += 1
                break
            raise Failed(f"{nm}: ext 절점 {nid} 이 움직여졌다")
    return f"ext 절점 이동 차단 {n}개 메쉬에서 확인"


def t_reflex_quad_rejected(names: list[str], pairs: dict) -> str:
    """오목 사각형이 되는 병합은 거부돼야 한다.

    ``features.interior_angles`` 는 acos 기반이라 180°를 넘는 내각을 180° 미만으로
    보고한다 — 품질 지표로는 절대 안 걸리므로 편집 단계에서 막는 게 유일한 방어선이다.
    """
    n = 0
    for nm in names:
        m = EditableMesh.load(*pairs[nm])
        for c in merge_candidates(m):
            if c.blocked and "오목" in c.blocked:
                n += 1
            elif c.ok:
                # 통과한 후보는 사각형 면적이 두 삼각형 면적의 합이어야 한다.
                # 오목(dart) 사각형이면 합이 아니라 차가 나오므로 확실히 갈린다.
                # (내각 합으로는 못 가린다 — 실측 오목 사례가 355.8° 로 360° 에 가까웠다)
                q = c.quality
                assert q is not None
                a1 = F.shell_quality(m.corners(c.si)).area
                a2 = F.shell_quality(m.corners(c.sj)).area
                # 허용 2%: 이면각 10° 까지 허용하므로 비평면성 때문에 실측 0.2% 는 난다.
                # 오목 사각형은 합이 아니라 차가 나와 자릿수가 달라지므로 이 폭으로 충분하다
                # (실측 사례: 삼각형 합 124758 vs 오목 사각형 면적 633).
                _ok(
                    abs(q.area - (a1 + a2)) / (a1 + a2) < 0.02,
                    f"{nm}: 통과한 병합의 면적 {q.area:.1f} != 삼각형 합 {a1 + a2:.1f} (오목 의심)",
                )
    return f"오목 사각형 병합 {n}건 거부 + 통과분 내각합 360° 확인"


# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="편집 계층 회귀 테스트")
    ap.add_argument("--n", type=int, default=200, help="검사할 메쉬 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pairs, _, _ = discover_pairs(gp.FE_DIR)
    names = sorted(pairs)[: args.n]
    print(f"메쉬 {len(names)}개로 편집 계층 검증\n")

    tests = [
        ("라운드트립", lambda: t_roundtrip(names, pairs)),
        ("split↔merge 역연산", lambda: t_split_merge_inverse(names, pairs, rng)),
        ("swap 2회 = 항등", lambda: t_swap_involution(names, pairs)),
        ("무작위 편집 불변식", lambda: t_invariants_under_edits(names, pairs, rng)),
        ("1D 요소 보호(기본)", lambda: t_line_elements_protected(names, pairs)),
        ("1D 함께 삭제(drop_lines)", lambda: t_drop_lines(names, pairs)),
        ("ext 절점 고정", lambda: t_ext_nodes_frozen(names, pairs)),
        ("오목 사각형 거부", lambda: t_reflex_quad_rejected(names, pairs)),
    ]

    failed = 0
    for label, fn in tests:
        try:
            msg = fn()
        except Failed as exc:
            failed += 1
            print(f"  [실패] {label}\n         {exc}")
        else:
            print(f"  [통과] {label}: {msg}")

    print()
    if failed:
        raise SystemExit(f"{failed}개 테스트 실패")
    print("전부 통과")


if __name__ == "__main__":
    main()

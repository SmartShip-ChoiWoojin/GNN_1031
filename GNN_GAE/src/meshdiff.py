"""두 메쉬의 차이 — 채점과 화면 강조가 **같은 정의**를 쓰도록 한 곳에 모은다.

비교는 절점 **번호가 아니라 좌표**로 한다. 사람은 두 절점을 합칠 때 남길 번호를
일정하게 고르지 않는다 — 흡수한 쪽 번호를 흡수된 자리로 옮기기도 하고(99건), 제 번호를
제자리에 두기도 한다(343건). 결과 메쉬는 기하적으로 똑같은데 번호로 비교하면 "다르다" 가
나온다. 그래서

  - 요소/1D 는 꼭짓점 **좌표 집합**으로 식별한다 (번호 무관)
  - 절점은 **좌표**로 식별한다 — "어느 자리가 사라졌나 / 새로 생겼나"
  - 다만 화면의 "이동" 표시는 번호로 판정한다. 같은 번호가 다른 자리로 갔다는 사실이
    사람의 작업을 읽는 데는 더 직관적이라서다 (사람 결과에만 나온다 — 재생성은 좌표를
    하나도 안 옮긴다)

    sig_a == sig_b      ⇔ 두 메쉬가 기하·연결·1D 까지 같다
    diff(before, after) → 지운 요소 / 만든 요소 / 사라진 자리 / 새 자리 / 옮긴 절점
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: 같은 자리로 볼 허용오차(mm). pairs.XYZ_TOL 과 같다 — 좌표가 24만 mm 단위라 충분히 엄격.
XYZ_TOL = 1e-3


def pkey(c) -> tuple:
    """좌표를 허용오차 단위로 끊은 사전 키."""
    return tuple(np.round(np.asarray(c, dtype=float) / XYZ_TOL).astype(np.int64))


def _shell_key(mesh, nodes) -> frozenset:
    return frozenset(pkey(mesh.coords[n]) for n in nodes)


@dataclass(frozen=True)
class MeshSig:
    """번호와 무관한 메쉬 서명. 두 서명이 같으면 같은 메쉬다."""

    nodes: frozenset
    shells: frozenset
    lines: frozenset

    def __len__(self) -> int:
        return len(self.shells)


def signature(mesh) -> MeshSig:
    """``RawMesh`` / ``EditableMesh`` 어느 쪽이든 받는다 (coords/shells/lines 만 쓴다)."""
    return MeshSig(
        nodes=frozenset(pkey(c) for c in mesh.coords.values()),
        shells=frozenset(_shell_key(mesh, s[2]) for s in mesh.shells),
        lines=frozenset(_shell_key(mesh, l[2]) for l in mesh.lines),
    )


# ──────────────────────────────────────────────────────────────────────────
@dataclass
class MeshDiff:
    """before → after 에서 바뀐 것. 인덱스는 각 메쉬의 ``shells`` / ``lines`` 목록 위치."""

    removed_shells: list[int] = field(default_factory=list)   # before 기준
    created_shells: list[int] = field(default_factory=list)   # after 기준
    removed_lines: list[int] = field(default_factory=list)    # before 기준
    created_lines: list[int] = field(default_factory=list)    # after 기준
    #: before 절점 id -> 설명. 그 **자리**가 after 에 없는 절점
    deleted_nodes: dict[int, str] = field(default_factory=dict)
    #: after 절점 id -> 설명. before 에 없던 **자리**
    added_nodes: dict[int, str] = field(default_factory=dict)
    #: 같은 번호가 자리를 옮긴 절점 -> (출발 좌표, 도착 좌표, 설명)
    moved_nodes: dict[int, tuple] = field(default_factory=dict)

    @property
    def n_changes(self) -> int:
        return (len(self.removed_shells) + len(self.created_shells)
                + len(self.removed_lines) + len(self.created_lines)
                + len(self.deleted_nodes) + len(self.added_nodes) + len(self.moved_nodes))


def diff(before, after, absorbed: dict[int, int] | None = None) -> MeshDiff:
    """before → after 의 차이.

    Args:
        absorbed: {사라진 절점: 이어받은 절점} — 설명 문구에만 쓴다. 재생성 결과면 연산
            기록(``Op.key``)에서, 사람 결과면 ``pairs.NodeMatch.absorbed`` 에서 온다.
    """
    absorbed = absorbed or {}
    d = MeshDiff()

    bk = [_shell_key(before, s[2]) for s in before.shells]
    ak = [_shell_key(after, s[2]) for s in after.shells]
    bset, aset = set(bk), set(ak)
    d.removed_shells = [i for i, k in enumerate(bk) if k not in aset]
    d.created_shells = [i for i, k in enumerate(ak) if k not in bset]

    bl = [_shell_key(before, l[2]) for l in before.lines]
    al = [_shell_key(after, l[2]) for l in after.lines]
    bls, als = set(bl), set(al)
    d.removed_lines = [i for i, k in enumerate(bl) if k not in als]
    d.created_lines = [i for i, k in enumerate(al) if k not in bls]

    bpos = {pkey(c): n for n, c in before.coords.items()}
    apos = {pkey(c): n for n, c in after.coords.items()}
    for n, c in before.coords.items():
        if pkey(c) in apos:
            continue
        if n in after.coords:
            continue    # 번호가 남아 다른 자리로 갔다 — 아래 '이동' 에서 다룬다
        if n in absorbed:
            d.deleted_nodes[n] = f"절점 {n} — {absorbed[n]} 에 합쳐져 사라짐"
        else:
            d.deleted_nodes[n] = f"절점 {n} — 삭제됨 (이어받은 절점 없음)"
    for n, c in after.coords.items():
        if n in before.coords:
            p0, p1 = np.asarray(before.coords[n], float), np.asarray(c, float)
            if float(np.linalg.norm(p1 - p0)) <= XYZ_TOL:
                continue
            took = bpos.get(pkey(c))
            if took is not None and took != n:
                why = f"절점 {n} — {took} 자리로 옮겨가며 {took} 를 흡수"
            else:
                why = f"절점 {n} — 이동 ({float(np.linalg.norm(p1 - p0)):.1f} mm)"
            d.moved_nodes[n] = (p0, p1, why)
        elif pkey(c) not in bpos:
            d.added_nodes[n] = f"절점 {n} — 새로 생김"
    return d


# ──────────────────────────────────────────────────────────────────────────
# 채점 — 재생성 결과를 사람 결과와 비교
# ──────────────────────────────────────────────────────────────────────────
def _prf(pred: set, true: set) -> tuple[float, float]:
    """(정밀도, 재현율). 둘 다 비면 1.0 — '아무것도 안 바꿔야 하는데 안 바꿨다' 는 맞힌 것."""
    if not pred and not true:
        return 1.0, 1.0
    tp = len(pred & true)
    return (tp / len(pred) if pred else 0.0), (tp / len(true) if true else 0.0)


def compare(bad, pred, good) -> dict:
    """재생성 결과(pred)를 사람 결과(good)에 대어 본다. 셋 다 같은 원본(bad)에서 출발.

    정확 일치(exact)가 주 지표다. 나머지는 "어디까지 맞았나" 를 보는 부분 점수:
      - 지운 요소: bad 의 요소 중 결과에서 사라진 것 (좌표 집합 기준)
      - 만든 요소: 결과에 새로 생긴 요소
      - 사라진 자리: bad 의 절점 자리 중 결과에 없는 것
    """
    sb, sp, sg = signature(bad), signature(pred), signature(good)
    rm_p, rm_g = sb.shells - sp.shells, sb.shells - sg.shells
    cr_p, cr_g = sp.shells - sb.shells, sg.shells - sb.shells
    dn_p, dn_g = sb.nodes - sp.nodes, sb.nodes - sg.nodes
    out = {"exact": sp == sg,
           "exact_shells": sp.shells == sg.shells,
           "n_rm_pred": len(rm_p), "n_rm_true": len(rm_g),
           "n_cr_pred": len(cr_p), "n_cr_true": len(cr_g),
           "n_del_pred": len(dn_p), "n_del_true": len(dn_g)}
    for nm, (p, t) in (("rm", (rm_p, rm_g)), ("cr", (cr_p, cr_g)), ("del", (dn_p, dn_g))):
        out[f"{nm}_prec"], out[f"{nm}_rec"] = _prf(p, t)
    # 바뀐 요소 전체(지운 것 + 만든 것)의 IoU — 한 숫자로 "얼마나 겹치나"
    ch_p, ch_g = rm_p | cr_p, rm_g | cr_g
    out["change_iou"] = (len(ch_p & ch_g) / len(ch_p | ch_g)) if (ch_p | ch_g) else 1.0
    out["did_nothing"] = sp == sb
    return out

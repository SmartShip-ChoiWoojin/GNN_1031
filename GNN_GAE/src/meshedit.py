"""편집 가능한 메쉬 + 국소 편집 연산 + 도메인 불변식 검사.

수리의 토대. ``MeshRep`` 은 읽기 전용 파생 구조라 편집에 못 쓴다 — 대신 ``RawMesh`` 를
그대로 들고 편집하고, 평가할 때만 ``MeshRep`` 을 다시 만든다. 패치가 절점 55개 규모라
매번 재구축해도 싸다.

연산은 전부 **국소**다. 메쉬 전체를 다시 생성하지 않는다 — 그래프 생성 모델은 유효한
.bdf(1D 요소가 쉘 변 위에 있고, 비매니폴드가 안 늘어나는)를 보장할 수 없기 때문이다.

도메인 불변식 (``check`` 가 검사한다):
  1. ext=true 절점은 좌표 불변 + 삭제 금지 — 이웃 패치와 공유하는 절단면이다
     (절점 병합이 섞인 재생성 결과는 ``regen=True`` 로 검사한다 — 사람도 ext 절점을
     이웃 ext 에 합친다. 그때 삭제는 경고, 이동은 여전히 위반)
  2. 모든 1D 요소(CBAR/CBEAM)는 쉘 변 위에 있어야 한다 — 원본 4000개에서 100% 성립
  3. 비매니폴드 변 수가 원본보다 늘지 않는다 (0 이 아니라 '악화 없음' 이 기준 —
     선체 모델은 패널 교차 때문에 원래 비매니폴드 변을 중앙값 8~13개 갖는다)
  4. 쉘 법선이 뒤집히지 않는다
  5. 총 면적 보존 — merge/swap 은 면적을 바꾸지 않는 연산이다 (절점 병합은 경계에서
     면적을 바꾼다 — ``regen=True`` 면 ``REGEN_AREA_TOL`` 까지 경고)

연산 (전부 국소, 절점 번호는 새로 만들지 않는다):
  merge_tris   삼각형 2개 → 사각형          swap_edge   대각선 뒤집기
  split_quad   사각형 → 삼각형 2개          move_node   절점 이동
  collapse_edge 절점 하나를 이웃에 합침 (사람 수정에서 사라진 절점의 94%)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import gae_paths  # noqa: F401  (부모 src 를 sys.path 에 넣는다)
import features as F
from bdf_io import RawMesh, load_raw_mesh
from meshrep import Edge, MeshRep

EPS = 1e-12

#: 두 삼각형을 합치거나 대각선을 뒤집어도 되는 이면각 상한(도).
#: 이보다 크면 서로 다른 패널로 보고 막는다.
#:
#: 10.0 이었는데 사람이 고친 275쌍을 재서 20.0 으로 올렸다. 실측 분포는 1° 이하가
#: 96.7%, 10° 이하가 99.3%, 최대 16.7° 다 — 10° 가드가 사람 수정의 0.7%를 막고 있었다.
#: 이면각이 큰 쌍도 **합친 결과 사각형의 warp 는 최대 2.4°** 로 거의 평면이다. 즉 그건
#: 실제로 꺾인 게 아니라 슬라이버 삼각형이라 법선이 불안정해서 크게 나온 값이다.
#: 표면이 실제로 바뀌는 것은 아래 면적 보존 검사가 막으므로, 이 값은 느슨해도 된다.
MAX_DIHEDRAL_DEG = 20.0

#: `check()` 가 보는 **메쉬 전체** 면적의 허용 변화. 요소를 하나 잃으면 ~2% 가 바뀌므로
#: 1e-3 이면 구조적 실수를 확실히 잡는다.
AREA_TOL = 1e-3

#: 병합/뒤집기 연산이 보는 **그 두 요소** 면적의 허용 변화.
#: 전체 기준과 분모가 다르므로 값도 달라야 한다 — 쌍의 면적이 0.8% 바뀌어도 메쉬
#: 전체로는 0.8% x (2/47) = 0.03% 라 AREA_TOL 과 충돌하지 않는다.
#: 1e-3 이었는데 사람이 고친 191개를 재서 1e-2 로 올렸다 — 4건이 0.14~0.80% 를
#: 바꾸며 표면을 평탄화했고(삼각형은 전부 멀쩡했다: 최소내각 20~49°), 그게 정답이다.
MERGE_AREA_TOL = 1e-2


# ──────────────────────────────────────────────────────────────────────────
# 편집 연산 기록
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Op:
    """적용된(또는 후보) 편집 연산 하나.

    ``kind`` 는 "merge" | "swap" | "split" | "move".
    ``key`` 는 연산을 **절점 id 로** 식별한다 — 쉘 인덱스는 편집하면 바뀌지만 절점 id 는
    안 바뀌므로, 정답 시퀀스를 저장했다가 나중에 재현할 때 안전하다.
    """

    kind: str
    key: tuple
    info: dict = field(default_factory=dict, compare=False)

    def __repr__(self) -> str:
        return f"Op({self.kind}, {self.key})"


class EditError(ValueError):
    """연산을 적용할 수 없을 때."""


def _compact(ns: list[int]) -> list[int] | None:
    """붕괴 뒤 요소의 절점 목록을 정리한다. 요소가 없어져야 하면 None.

    이웃한 중복(감김 순서상 붙은 두 절점이 같아진 것)은 하나로 합친다 — 사각형이
    삼각형이 된다. 그러고도 중복이 남으면(마주 보는 모서리가 겹친 사각형) 면적이 0 인
    '나비넥타이' 라 지운다. 3개 미만이면 찌그러져 없어진 것이다.
    """
    out: list[int] = []
    for x in ns:
        if not out or out[-1] != x:
            out.append(x)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    if len(out) < 3 or len(set(out)) != len(out):
        return None
    return out


def _relation(shells, remove: int, keep: int) -> str | None:
    """``shells`` (``[_, _, 절점목록 또는 None, ...]``) 안에서 두 절점의 관계. "edge"|"diag"|None."""
    diag = False
    for w in shells:
        ns = w[2]
        if ns is None or remove not in ns or keep not in ns:
            continue
        k = len(ns)
        if (ns.index(remove) - ns.index(keep)) % k in (1, k - 1):
            return "edge"
        diag = True
    return "diag" if diag else None


def _edge_count(shells) -> dict:
    """변 -> 그 변을 쓰는 쉘 수."""
    c: dict = {}
    for s in shells:
        ns = s[2]
        for e in zip(ns, ns[1:] + ns[:1]):
            k = Edge(e)
            c[k] = c.get(k, 0) + 1
    return c


def _convex_quad(P: np.ndarray) -> bool:
    """사각형이 볼록한가 — 두 대각선 어느 쪽으로 잘라도 두 삼각형의 법선이 같은 쪽."""
    for i in (0, 1):
        a, b, c, d = P[i], P[(i + 1) % 4], P[(i + 2) % 4], P[(i + 3) % 4]
        if float(np.dot(np.cross(b - a, c - a), np.cross(c - a, d - a))) <= 0.0:
            return False
    return True


def _num(v: float) -> str:
    """원본 .bdf 와 같은 십진 표기 (243680.0, -2840.0 처럼).

    정수값만 ``.1f`` 로 쓰고 나머지는 ``repr`` 을 쓴다 — ``repr`` 은 double 을 정확히
    되읽을 수 있는 가장 짧은 표기다. ``%.6g`` 를 쓰면 좌표가 24만 단위라 유효숫자가
    모자라 조용히 잘린다 (실측: 1127.4305 → 1127.43, 0.5mm 오차).
    """
    if abs(v - round(v)) < 1e-9:
        return f"{v:.1f}"
    return repr(v)


# ──────────────────────────────────────────────────────────────────────────
# 편집 가능한 메쉬
# ──────────────────────────────────────────────────────────────────────────
class EditableMesh:
    """``RawMesh`` 의 편집 가능한 사본.

    상태는 ``RawMesh`` 와 같은 모양이라 언제든 되돌려 쓸 수 있다 (``to_raw``,
    ``write_bdf``). ``rep`` 은 캐시되고 편집하면 자동으로 무효화된다.
    """

    def __init__(self, raw: RawMesh):
        self.name = raw.name
        self.coords: dict[int, tuple[float, float, float]] = dict(raw.coords)
        self.shells: list[list] = [[eid, t, list(n)] for eid, t, n in raw.shells]
        self.lines: list[list] = [[eid, t, tuple(n)] for eid, t, n in raw.lines]
        self.ext: dict[int, bool] = dict(raw.ext)
        self.log: list[Op] = []
        self._rep: MeshRep | None = None

    # ── 로딩 / 저장 ───────────────────────────────────────────────────
    @classmethod
    def load(cls, bdf_path: str, json_path: str | None = None) -> "EditableMesh":
        return cls(load_raw_mesh(bdf_path, json_path))

    def clone(self) -> "EditableMesh":
        m = EditableMesh.__new__(EditableMesh)
        m.name = self.name
        m.coords = dict(self.coords)
        m.shells = [[s[0], s[1], list(s[2])] for s in self.shells]
        m.lines = [[l[0], l[1], tuple(l[2])] for l in self.lines]
        m.ext = dict(self.ext)
        m.log = list(self.log)
        # MeshRep 은 읽기 전용 파생 구조라 사본이 고쳐지기 전까지는 같이 써도 된다
        # (고치는 순간 _dirty() 가 끊는다). 후보 수백 개를 사본으로 미리 볼 때 재구축을 아낀다.
        m._rep = self._rep
        return m

    def to_raw(self) -> RawMesh:
        return RawMesh(
            name=self.name,
            coords=dict(self.coords),
            shells=[(e, t, list(n)) for e, t, n in self.shells],
            lines=[(e, t, tuple(n)) for e, t, n in self.lines],
            ext=dict(self.ext),
        )

    def write_bdf(self, path: str) -> None:
        """원본과 같은 자유형식 쉼표 .bdf 로 쓴다 (PID 는 데이터셋 전체가 1)."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.bdf_text())

    def bdf_text(self) -> str:
        """``write_bdf`` 가 쓰는 내용 그대로 (화면에서 내려받기용)."""
        out = ["BEGIN BULK"]
        for nid in sorted(self.coords):
            x, y, z = self.coords[nid]
            out.append(f"GRID,{nid},,{_num(x)},{_num(y)},{_num(z)}")
        for eid, t, n in self.shells:
            out.append(f"{t},{eid},1," + ",".join(str(i) for i in n))
        for eid, t, n in self.lines:
            out.append(f"{t},{eid},1,{n[0]},{n[1]}")
        out.append("ENDDATA")
        return "\n".join(out) + "\n"

    def write_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.json_text())

    def json_text(self) -> str:
        import json

        return json.dumps({str(k): bool(v) for k, v in sorted(self.ext.items())}, indent=2)

    # ── 파생 구조 ─────────────────────────────────────────────────────
    @property
    def rep(self) -> MeshRep:
        if self._rep is None:
            self._rep = MeshRep(self.to_raw())
        return self._rep

    def _dirty(self) -> None:
        self._rep = None

    def P(self, nid: int) -> np.ndarray:
        return np.array(self.coords[nid], dtype=float)

    def corners(self, si: int) -> np.ndarray:
        return np.array([self.coords[n] for n in self.shells[si][2]], dtype=float)

    def quality(self, si: int) -> F.ShellQuality:
        return F.shell_quality(self.corners(si))

    def tri_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.shells) if len(s[2]) == 3]

    def _next_eid(self) -> int:
        ids = [s[0] for s in self.shells] + [l[0] for l in self.lines]
        return (max(ids) + 1) if ids else 1

    def line_edges(self) -> set:
        """1D 요소가 올라타 있는 변 집합.

        이 변을 없애려면 그 위의 1D 요소도 같이 지워야 한다. 예전에는 "없애면 안 되는
        변" 으로 보고 무조건 거부했는데, 사람이 고친 508쌍을 재보니 **손댄 구역 안의
        1D 요소 1,886개 중 361개(19.1%)를 실제로 지웠다.** 금지 규칙이 아니라
        "같이 처리해야 하는 대상" 으로 다룬다 (``drop_lines`` 참고).
        """
        return {Edge(tuple(l[2])) for l in self.lines}

    def lines_on(self, e) -> list[int]:
        """변 ``e`` 위에 올라타 있는 1D 요소의 인덱스."""
        return [i for i, l in enumerate(self.lines) if Edge(tuple(l[2])) == Edge(e)]

    # ── 연산 1: 인접 삼각형 2개 → 사각형 ───────────────────────────────
    def merge_tris(
        self, si: int, sj: int, max_dihedral: float = MAX_DIHEDRAL_DEG,
        drop_lines: bool = False,
    ) -> Op:
        """공유 변을 대각선으로 흡수해 CQUAD4 하나로 만든다.

        공유 변 위에 1D 요소(보강재)가 있으면 그 변을 없앨 때 1D 도 같이 없애야 한다.
        ``drop_lines=False`` 면 거부하고, True 면 1D 요소를 함께 지운다. 기본이 False 인
        이유는 1D 가 실제 구조 부재라 지우는 것이 결과가 큰 결정이기 때문 — 부르는 쪽이
        명시적으로 골라야 한다. 지운 요소는 ``Op.info["dropped_lines"]`` 에 남는다.
        """
        shared = self._shared_edge(si, sj)
        quad = self._merged_quad_nodes(si, sj, max_dihedral, drop_lines)
        dropped = self._drop_lines_on(shared) if drop_lines else []
        keep, drop = (si, sj) if si < sj else (sj, si)
        self.shells[keep] = [self.shells[keep][0], "CQUAD4", quad]
        del self.shells[drop]
        self._dirty()
        op = Op("merge", tuple(quad), {"dropped_lines": dropped})
        self.log.append(op)
        return op

    def _shared_edge(self, si: int, sj: int):
        shared = [n for n in self.shells[si][2] if n in self.shells[sj][2]]
        return Edge(tuple(shared)) if len(shared) == 2 else None

    def _drop_lines_on(self, e) -> list[tuple]:
        """변 위의 1D 요소를 지우고 (eid, type, nodes) 목록을 돌려준다."""
        if e is None:
            return []
        idx = self.lines_on(e)
        out = [(self.lines[i][0], self.lines[i][1], tuple(self.lines[i][2])) for i in idx]
        for i in sorted(idx, reverse=True):
            del self.lines[i]
        return out

    def _merged_quad_nodes(
        self, si: int, sj: int, max_dihedral: float = MAX_DIHEDRAL_DEG,
        drop_lines: bool = False,
    ) -> list[int]:
        """병합 결과 사각형의 절점 순서를 구한다. 불가능하면 ``EditError``."""
        a, b = self.shells[si], self.shells[sj]
        if len(a[2]) != 3 or len(b[2]) != 3:
            raise EditError("merge 는 삼각형 2개에만 적용된다")
        shared = [n for n in a[2] if n in b[2]]
        if len(shared) != 2:
            raise EditError(f"공유 절점이 2개가 아님: {shared}")
        if not drop_lines and Edge(tuple(shared)) in self.line_edges():
            raise EditError("공유 변 위에 1D 요소가 있다 (drop_lines 로 함께 지울 수 있다)")

        # 두 삼각형이 서로 다른 패널에 있으면 합치거나 뒤집는 순간 선체 표면이 바뀐다.
        # 실측으로 이면각 90°(직교 패널) 쌍을 swap 해서 총 면적이 0.02% 변하는 사례가 나왔다.
        qa = F.shell_quality(np.array([self.coords[n] for n in a[2]], dtype=float))
        qb = F.shell_quality(np.array([self.coords[n] for n in b[2]], dtype=float))
        dih = F.dihedral_deg(qa.normal, qb.normal)
        if dih > max_dihedral:
            raise EditError(f"두 삼각형이 다른 패널에 있다 (이면각 {dih:.1f}°)")

        p = next(n for n in a[2] if n not in shared)
        q = next(n for n in b[2] if n not in shared)
        # a 의 감김 방향을 유지하도록 p 를 맨 앞으로 회전 → (p, x, y), {x,y} == shared
        k = a[2].index(p)
        _, x, y = a[2][k:] + a[2][:k]
        quad = [p, x, q, y]
        if len(set(quad)) != 4:
            raise EditError("병합 결과에 중복 절점")

        Pq = np.array([self.coords[n] for n in quad], dtype=float)
        # 오목(reflex) 사각형 방지. p 에서 부채꼴로 자른 두 삼각형의 법선이 서로
        # 반대면 대각선 p-q 가 도형 밖으로 나간 것 — 내각 하나가 180°를 넘는다.
        # features.interior_angles 는 acos 라 이런 각을 180° 미만으로 보고하므로
        # 품질 지표로는 걸러지지 않는다. 여기서 막아야 한다.
        # (실측: 얇은 슬라이버 삼각형 2개가 만나 내각 [11°, 182°, 11°, 156°] 같은
        #  화살촉 사각형이 되는 경우가 인접 삼각형 쌍의 29%)
        n1 = np.cross(Pq[1] - Pq[0], Pq[2] - Pq[0])
        n2 = np.cross(Pq[2] - Pq[0], Pq[3] - Pq[0])
        if float(np.dot(n1, n2)) <= 0:
            raise EditError("병합하면 오목(reflex) 사각형이 된다")

        # 표면 보존 — 이 연산의 진짜 유효성 조건이다.
        # 사각형 면적은 p 에서 부채꼴로 자른 (p,x,q)+(p,q,y) 의 합이므로, 이 값은
        # merge 결과의 면적이면서 동시에 **swap 결과 두 삼각형의 면적 합**이다.
        # 원래 두 삼각형의 합과 같아야 표면이 안 변한 것이다. 이 한 줄이
        #   - 다른 대각선으로 크게 접힌 4점 (이면각은 0.14°인데 warp 90°인 실측 사례)
        #   - 퇴화에 가까워 법선이 불안정한 쌍
        #   - 오목(dart) 사각형 — 합이 아니라 차가 나온다
        # 을 전부 한꺼번에 막는다. 이면각 검사만으로는 첫 두 개를 못 잡는다.
        a_sum = qa.area + qb.area
        a_new = F.area_normal(Pq)[0]
        if a_sum > 0 and abs(a_new - a_sum) / a_sum > MERGE_AREA_TOL:
            raise EditError(
                f"표면 면적이 바뀐다 ({a_sum:.1f} → {a_new:.1f}, "
                f"{100 * (a_new - a_sum) / a_sum:+.2f}%)"
            )
        return quad

    def merged_quality(
        self, si: int, sj: int, max_dihedral: float = MAX_DIHEDRAL_DEG,
        drop_lines: bool = False,
    ) -> F.ShellQuality:
        """실제로 합치지 않고 결과 사각형의 품질만 본다."""
        quad = self._merged_quad_nodes(si, sj, max_dihedral, drop_lines)
        return F.shell_quality(np.array([self.coords[n] for n in quad], dtype=float))

    # ── 연산 2: 대각선 뒤집기 ─────────────────────────────────────────
    def swap_edge(
        self, si: int, sj: int, max_dihedral: float = MAX_DIHEDRAL_DEG,
        drop_lines: bool = False,
    ) -> Op:
        """공유 변을 가진 삼각형 2개의 대각선을 반대쪽으로 뒤집는다.

        뒤집으면 원래 대각선이 사라지므로, 그 위에 1D 요소가 있으면 ``merge_tris`` 와
        같은 선택이 필요하다 — 거부하거나(기본) 1D 를 같이 지우거나.
        """
        shared = self._shared_edge(si, sj)
        quad = self._merged_quad_nodes(si, sj, max_dihedral, drop_lines)  # 같은 유효성 조건
        p, x, q, y = quad
        a, b = self.shells[si], self.shells[sj]
        # 새 대각선 p-q 가 이미 다른 곳에 있는 변이면 중복 변이 생긴다
        if Edge((p, q)) in self.rep.edge_shells:
            raise EditError(f"새 대각선 {p}-{q} 가 이미 존재하는 변이다")
        dropped = self._drop_lines_on(shared) if drop_lines else []
        self.shells[si] = [a[0], "CTRIA3", [p, x, q]]
        self.shells[sj] = [b[0], "CTRIA3", [p, q, y]]
        self._dirty()
        op = Op("swap", (p, x, q, y), {"dropped_lines": dropped})
        self.log.append(op)
        return op

    # ── 연산 3: 사각형 → 삼각형 2개 (손상 생성의 주력) ──────────────────
    def split_quad(self, si: int, diag: int = 0) -> Op:
        """CQUAD4 를 대각선으로 갈라 CTRIA3 2개로 만든다. ``merge_tris`` 의 역연산."""
        s = self.shells[si]
        if len(s[2]) != 4:
            raise EditError("split 은 사각형에만 적용된다")
        a, b, c, d = s[2]
        # 새로 생기는 대각선이 이미 메쉬에 있는 변이면 그 변의 인접 쉘이 3개가 되어
        # 비매니폴드가 늘어난다. swap 과 같은 조건인데 split 에 빠져 있었다
        # (실측: 1000개 중 SN2592_FWD_Clean_0497 에서 비매니폴드 변 6 → 7).
        newdiag = Edge((a, c)) if diag == 0 else Edge((b, d))
        if newdiag in self.rep.edge_shells:
            raise EditError(f"새 대각선 {sorted(newdiag)} 가 이미 존재하는 변이다")
        t1, t2 = ([a, b, c], [a, c, d]) if diag == 0 else ([b, c, d], [b, d, a])
        self.shells[si] = [s[0], "CTRIA3", t1]
        self.shells.append([self._next_eid(), "CTRIA3", t2])
        self._dirty()
        op = Op("split", (a, b, c, d), {"diag": diag})
        self.log.append(op)
        return op

    # ── 연산 4: 절점 이동 ─────────────────────────────────────────────
    def move_node(self, nid: int, delta: np.ndarray) -> Op:
        """절점을 옮긴다. ext 절점은 거부한다 (이웃 패치와 공유하는 절단면)."""
        if self.ext.get(nid, False):
            raise EditError(f"절점 {nid} 은 ext=true — 움직일 수 없다")
        x, y, z = self.coords[nid]
        d = np.asarray(delta, dtype=float)
        self.coords[nid] = (x + float(d[0]), y + float(d[1]), z + float(d[2]))
        self._dirty()
        op = Op("move", (nid,), {"delta": d.tolist()})
        self.log.append(op)
        return op

    # ── 연산 5: 절점 병합 (변 붕괴) ─────────────────────────────────────
    def collapse_relation(self, remove: int, keep: int) -> str | None:
        """두 절점이 합칠 수 있는 관계인가. "edge" | "diag" | None.

        사람이 합친 451쌍을 원본에서 재보면 변으로 이어진 이웃 432 / 사각형의 대각
        모서리 12 / 두 칸 떨어짐 7 이다. 앞의 둘(98.4%)만 받는다 — 두 칸 떨어진 것은
        다른 병합이 먼저 일어나 이웃이 된 뒤에야 합쳐진 것이다.
        """
        return _relation([[None, None, ns] for _, _, ns in self.shells], remove, keep)

    def collapse_edge(self, remove: int, keep: int) -> Op:
        """절점 ``remove`` 를 ``keep`` 에 합친다. ``collapse_many([(remove, keep)])`` 와 같다."""
        return self.collapse_many([(remove, keep)])

    def collapse_many(self, pairs: list[tuple[int, int]]) -> Op:
        """절점 병합 여러 개를 **한꺼번에** 적용한다. ``(remove, keep)`` — remove 의 자리가 사라진다.

        사람이 고친 508쌍에서 사라진 절점 481개 중 451개가 이 연산이다(nodemap 의 '병합').
        사람은 남는 절점 **번호**를 일정하게 고르지 않는다 — 흡수한 쪽 번호에 흡수된
        자리(99건)이기도 하고, 제 번호에 제자리(343건)이기도 하다. 결과 메쉬는 기하적으로
        같으므로 여기서는 "자리가 남는 절점의 번호가 남는다" 로 고정한다. 그래서 이 연산은
        좌표를 하나도 바꾸지 않는다.

        왜 한꺼번에인가: 병합을 하나씩 검사하면 **중간 상태**에서 거부되는 경우가 있다.
        실측 `_0274` — 19904→19905 만 하면 사각형 41399 가 내각 177°로 오목해지고,
        22947→19905 만 하면 세 점이 일직선이 된다. 둘 다 하면 그 사각형은 깨끗한
        삼각형이 된다. 그래서 순서대로 적용하되 모양 검사는 **마지막 상태 하나**에 한다.
        이웃 여부만은 순서대로 본다 — 앞의 병합으로 이웃이 된 쌍(두 칸 떨어진 7건)을 받기 위해.

        ``remove`` 를 쓰던 요소는:
          - 삼각형이 찌그러지면(절점이 겹치면) 지운다
          - 사각형의 이웃한 두 모서리가 겹치면 삼각형이 된다 (이웃 변 붕괴)
          - 사각형의 마주 보는 두 모서리가 겹치면(대각 붕괴) 면적이 0 이 되므로 지운다
          - 이미 있는 요소와 절점 집합이 같아지면 겹친 요소라 지운다

        **거부**하는 경우 (``EditError``, 메쉬는 그대로):
          - 이미 사라진 절점을 합치려 한다 / 이웃(변 또는 사각형 대각)이 아니다
          - 남는 요소가 접히거나(법선 반전) 면적이 0 에 가까워진다
          - 남는 사각형이 오목해진다
          - 비매니폴드 변이 늘어난다
          - 1D 요소가 쉘 변 위를 벗어난다

        ext 절점의 자리를 없애는 것은 **거부하지 않는다** — 사람도 508쌍 중 48쌍에서 ext
        자리 60개를 없앴다(대부분 절단면을 따라 이웃한 ext 끼리). 대신 ``Op.info`` 에
        ``ext_removed`` 로 남겨, 검사(``check``)와 화면이 경고로 보여주게 한다.
        """
        if not pairs:
            raise EditError("합칠 절점이 없다")
        coords = self.coords
        # 작업본: [eid, type, 지금 절점, 원래 절점]. 원래 절점은 접힘 검사의 기준이다.
        work = [[eid, t, list(ns), list(ns)] for eid, t, ns in self.shells]
        gone: set[int] = set()
        rels: list[str] = []
        for remove, keep in pairs:
            if remove == keep:
                raise EditError("같은 절점끼리는 합칠 수 없다")
            for n in (remove, keep):
                if n not in coords or n in gone:
                    raise EditError(f"없는(또는 이미 합쳐진) 절점: {n}")
            rel = _relation(work, remove, keep)
            if rel is None:
                raise EditError(f"{remove} 와 {keep} 는 이웃(변 또는 사각형 대각)이 아니다")
            rels.append(rel)
            for w in work:
                if w[2] is not None and remove in w[2]:
                    w[2] = _compact([keep if x == remove else x for x in w[2]])
            gone.add(remove)

        removed: list[tuple] = []
        alive: list[list] = []
        for eid, t, now, orig in work:
            if now is None:
                removed.append((eid, t, tuple(orig)))
            else:
                alive.append([eid, t, now, orig])

        # 겹친 요소 — 모양이 바뀐 요소가 다른 요소와 절점 집합이 같아진 경우 바뀐 쪽을 지운다
        seen = {frozenset(w[2]) for w in alive if w[2] == w[3]}
        kept: list[list] = []
        for w in alive:
            if w[2] != w[3]:
                k = frozenset(w[2])
                if k in seen:
                    removed.append((w[0], w[1], tuple(w[3])))
                    continue
                seen.add(k)
            kept.append(w)

        # 남는 요소의 모양 검사 — 사라진 자리를 쓰던 요소는 남는 자리로 늘어나며 접힐 수 있다
        P = lambda ns: np.array([coords[n] for n in ns], dtype=float)
        for eid, t, now, orig in kept:
            if now == orig:
                continue
            a0, n0 = F.area_normal(P(orig))
            a1, n1 = F.area_normal(P(now))
            if a1 <= 1e-9 * max(a0, 1.0):
                raise EditError(f"요소 {eid} 의 면적이 0 이 된다")
            if float(np.dot(n0, n1)) <= 0.0:
                raise EditError(f"요소 {eid} 이 접힌다 (법선 반전)")
            if len(now) == 4 and not _convex_quad(P(now)):
                raise EditError(f"요소 {eid} 이 오목한 사각형이 된다")
        # 총 면적 — 경계 절점을 합치면 면적이 바뀐다. 사람은 최대 2.6% 였다. 그 이상은
        # 표면을 크게 잘라먹는 병합 묶음이라 거부한다 (학습이 덜 된 모델이 병합 35개를
        # 한꺼번에 골라 면적 36%를 날린 것을 실제로 봤다).
        a_tot = float(sum(s.quality.area for s in self.rep.shells))
        d_area = -sum(F.area_normal(P(r[2]))[0] for r in removed)
        for eid, t, now, orig in kept:
            if now != orig:
                d_area += F.area_normal(P(now))[0] - F.area_normal(P(orig))[0]
        if a_tot > 0 and abs(d_area) / a_tot > REGEN_AREA_TOL:
            raise EditError(f"총 면적이 {100 * d_area / a_tot:+.1f}% 바뀐다")
        new_shells = [[eid, "CQUAD4" if len(now) == 4 else "CTRIA3", now]
                      for eid, t, now, orig in kept]

        # 1D 요소 — 번호를 옮기고, 길이가 0 이 되거나 겹친 것은 지운다
        fwd: dict[int, int] = {}
        for remove, keep in pairs:
            fwd[remove] = keep
        def res(n: int) -> int:
            while n in fwd:
                n = fwd[n]
            return n
        new_lines: list[list] = []
        dropped: list[tuple] = []
        lseen: set = set()
        for eid, t, (a, b) in self.lines:
            a2, b2 = res(a), res(b)
            k = Edge((a2, b2))
            if a2 == b2 or k in lseen:
                dropped.append((eid, t, (a, b)))
                continue
            lseen.add(k)
            new_lines.append([eid, t, (a2, b2)])

        # 위상 검사 — 비매니폴드 변 수와 1D 위치
        before_c, after_c = _edge_count(self.shells), _edge_count(new_shells)
        nm0 = sum(1 for v in before_c.values() if v > 2)
        nm1 = sum(1 for v in after_c.values() if v > 2)
        if nm1 > nm0:
            raise EditError(f"비매니폴드 변이 {nm0} → {nm1} 로 늘어난다")
        was_on = {l[0] for l in self.lines if Edge(tuple(l[2])) in before_c}
        for eid, t, (a, b) in new_lines:
            if eid in was_on and Edge((a, b)) not in after_c:
                raise EditError(f"1D 요소 {eid} 가 쉘 변 위를 벗어난다")

        ext_removed = [r for r, _ in pairs if self.ext.get(r, False)]
        self.shells = new_shells
        self.lines = new_lines
        for remove, keep in pairs:
            # ext 자리를 합쳤다면 남는 절점이 이웃 패치와의 연결을 이어받는다
            k = res(keep)
            self.ext[k] = bool(self.ext.get(k, False)) or bool(self.ext.get(remove, False))
        for remove, _ in pairs:
            del self.coords[remove]
            self.ext.pop(remove, None)
        self._dirty()
        op = Op("collapse", tuple(tuple(p) for p in pairs), {
            "relations": rels, "removed_shells": removed, "dropped_lines": dropped,
            "ext_removed": ext_removed,
        })
        self.log.append(op)
        return op


# ──────────────────────────────────────────────────────────────────────────
# 후보 열거
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MergeCand:
    """삼각형 병합 후보 하나 + 적용했을 때의 결과."""

    si: int
    sj: int
    #: 두 삼각형 사이 이면각(도). 크면 서로 다른 패널이라 합치면 안 된다.
    dihedral: float
    #: 병합 결과 사각형의 품질. 막힌 후보면 None.
    quality: F.ShellQuality | None
    #: 막혔다면 이유, 아니면 None
    blocked: str | None
    #: 공유 변 위에 올라타 있는 1D 요소 수. 0 이 아니면 합칠 때 같이 지워야 한다
    #: (금지가 아니다 — 사람도 손댄 구역 1D 의 19%를 지운다).
    lines_on_edge: int = 0

    @property
    def ok(self) -> bool:
        return self.blocked is None


def merge_candidates(m: EditableMesh, drop_lines: bool = False) -> list[MergeCand]:
    """변을 공유하는 삼각형 쌍 전부를, 막힌 것까지 이유와 함께 돌려준다.

    막힌 이유를 버리지 않는 건 진단 때문이다 — '1D 요소 때문에 못 합치는 쌍이 몇 %인가'
    가 수리 상한을 직접 좌우한다.
    """
    rep = m.rep
    tri = set(m.tri_indices())
    out: list[MergeCand] = []
    for e, v in rep.edge_shells.items():
        if len(v) != 2 or v[0] not in tri or v[1] not in tri:
            continue
        si, sj = v
        dih = F.dihedral_deg(rep.shells[si].quality.normal, rep.shells[sj].quality.normal)
        n_line = len(m.lines_on(e))
        try:
            q = m.merged_quality(si, sj, drop_lines=drop_lines)
        except EditError as exc:
            out.append(MergeCand(si, sj, dih, None, str(exc), n_line))
        else:
            out.append(MergeCand(si, sj, dih, q, None, n_line))
    return out


def collapse_greedy(m: EditableMesh, pairs: list[tuple[int, int]]
                    ) -> tuple[list[tuple[int, int]], dict[tuple[int, int], str]]:
    """병합 목록을 가능한 만큼 적용한다. ``(적용한 것, {거부한 것: 사유})``.

    전부 한꺼번에 되면 그대로 끝. 안 되면 **주어진 순서대로** 하나씩 보태 보며 묶음이
    여전히 유효한 것만 남긴다 (순서 = 우선순위 — 모델이 쓸 때는 점수 순). 앞의 병합이
    있어야 이웃이 되는 쌍이 있으므로 거부된 것은 진전이 없을 때까지 다시 훑는다.
    메쉬에는 남긴 것 전부를 ``collapse_many`` 한 번으로 적용한다.
    """
    pairs = [tuple(p) for p in pairs]
    if not pairs:
        return [], {}
    try:
        m.clone().collapse_many(pairs)
    except EditError:
        pass
    else:
        m.collapse_many(pairs)
        return pairs, {}

    acc: list[tuple[int, int]] = []
    why: dict[tuple[int, int], str] = {}
    pending = list(pairs)
    while pending:
        progress = False
        for p in list(pending):
            # 이미 쓰인 절점과 엉키는 것 — 사라진 절점을 또 합치거나, 사라진 자리에 합치기
            if any(p[0] == q[0] or p[1] == q[0] for q in acc):
                why[p] = "다른 병합과 겹친다"
                pending.remove(p)
                continue
            try:
                m.clone().collapse_many(acc + [p])
            except EditError as exc:
                why[p] = str(exc)
                continue
            acc.append(p)
            pending.remove(p)
            why.pop(p, None)
            progress = True
        if not progress:
            break
    if acc:
        m.collapse_many(acc)
    return acc, {p: why[p] for p in pairs if p in why}


def apply_collapse_seq(bad: EditableMesh, pairs: list[tuple[int, int]]) -> EditableMesh:
    """병합을 주어진 순서로 **하나씩** 적용한 사본 (반복형 재생성이 고른 순서 그대로)."""
    m = bad.clone()
    m.log.clear()
    for p in pairs:
        m.collapse_edge(*p)
    return m


def apply_merges(m: EditableMesh, edges: list) -> tuple[list, list[str]]:
    """공유 변 목록대로 삼각형 쌍을 합친다 (제자리). ``(합친 변, 못 합친 사유)``.

    병합할 때마다 쉘 인덱스가 밀리므로 매번 변으로 다시 찾는다.
    """
    done, notes = [], []
    for e in edges:
        hit = None
        for c in merge_candidates(m, drop_lines=True):
            if Edge(tuple(set(m.shells[c.si][2]) & set(m.shells[c.sj][2]))) == e:
                hit = c
                break
        if hit is None or not hit.ok:
            notes.append(f"{sorted(e)}: " + (hit.blocked if hit else "더 이상 삼각형 쌍이 아님"))
            continue
        try:
            m.merge_tris(hit.si, hit.sj, drop_lines=True)
        except EditError as exc:
            notes.append(f"{sorted(e)}: {exc}")
            continue
        done.append(e)
    return done, notes


def replay_ops(bad: EditableMesh, collapses: list[tuple[int, int]], merges: list) -> EditableMesh:
    """기록된 재생성 연산을 원본에 다시 적용한다 — 모델 없이 결과를 재현한다 (화면이 쓴다).

    반복형은 병합을 하나씩, oneshot 은 묶음으로 적용했다. 하나씩 먼저 해 보고 중간에
    거부되면 묶음으로 한다 — 둘 다 된다면 결과가 같다(check_edits '묶음 = 하나씩').
    """
    try:
        m = apply_collapse_seq(bad, collapses)
    except EditError:
        m = bad.clone()
        m.log.clear()
        if collapses:
            m.collapse_many(list(collapses))
    apply_merges(m, merges)
    return m


def collapse_pairs(m: EditableMesh) -> list[tuple[int, int, str]]:
    """합칠 수 있는 관계의 절점 쌍을 **양방향으로** 전부 돌려준다. ``(remove, keep, relation)``.

    방향이 결과를 가른다 — 어느 자리가 남느냐에 따라 주변 요소의 모양이 완전히 다르다.
    그래서 (a→b) 와 (b→a) 를 별개의 후보로 둔다. 실제로 합칠 수 있는지(접힘 등)는
    여기서 보지 않는다 — ``collapse_edge`` 를 시도해 봐야 안다.
    """
    seen: dict[tuple[int, int], str] = {}
    for _, _, ns in m.shells:
        k = len(ns)
        for i in range(k):
            a, b = ns[i], ns[(i + 1) % k]
            seen[(min(a, b), max(a, b))] = "edge"
        if k == 4:
            for a, b in ((ns[0], ns[2]), (ns[1], ns[3])):
                seen.setdefault((min(a, b), max(a, b)), "diag")
    out = []
    for (a, b), rel in sorted(seen.items()):
        out.append((a, b, rel))
        out.append((b, a, rel))
    return out


# ──────────────────────────────────────────────────────────────────────────
# 불변식 검사
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class CheckReport:
    violations: list[str] = field(default_factory=list)
    #: 위반은 아니지만 눈에 띄게 알려야 할 것 (``check(..., regen=True)`` 에서만 채운다)
    warnings: list[str] = field(default_factory=list)
    n_shell: int = 0
    n_tri: int = 0
    n_nonmanifold: int = 0
    area_total: float = 0.0
    max_node_move: float = 0.0
    #: 원본 대비 사라진 1D 요소 수. 위반은 아니지만(사람도 손댄 구역 1D 의 19%를 지운다)
    #: 보강재가 조용히 없어지는 일이라 항상 눈에 띄게 적어 둔다.
    lines_removed: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def __str__(self) -> str:
        head = "OK" if self.ok else f"위반 {len(self.violations)}건"
        body = "".join(f"\n  - {v}" for v in self.violations)
        body += "".join(f"\n  (경고) {w}" for w in self.warnings)
        drop = f" / 1D 삭제 {self.lines_removed}개" if self.lines_removed else ""
        return (
            f"[{head}] 쉘 {self.n_shell}개 (삼각 {self.n_tri}) / "
            f"비매니폴드변 {self.n_nonmanifold}{drop}{body}"
        )


#: ``check(..., regen=True)`` 에서 총 면적 변화를 위반으로 보는 선.
#: 절점 병합은 경계 절점을 합치면 면적을 바꾼다 — 사람도 508쌍 중 107쌍에서 0.1% 넘게,
#: 최대 2.6% 바꿨다. 그 위로 넉넉히 둔다. 0.1% 와 이 값 사이는 경고로 남긴다.
REGEN_AREA_TOL = 5e-2


def check(m: EditableMesh, base: EditableMesh | None = None, regen: bool = False) -> CheckReport:
    """도메인 불변식을 검사한다. ``base`` 를 주면 '원본 대비 악화' 까지 본다.

    ``regen=True`` 는 재생성 결과(절점 병합이 섞인 결과)를 검사할 때 쓴다. 사람의 수정에서
    실제로 일어나는 두 가지를 위반이 아니라 **경고**로 돌린다:
      - ext 절점이 다른 절점에 합쳐져 사라짐 (508쌍 중 48쌍)
      - 총 면적이 0.1% 넘게 바뀜 (107쌍, 최대 2.6%) — ``REGEN_AREA_TOL`` 까지는 경고
    ext 절점이 **움직이는** 것은 여전히 위반이다.
    """
    r = CheckReport()
    rep = m.rep
    r.n_shell = len(m.shells)
    r.n_tri = sum(1 for s in m.shells if len(s[2]) == 3)

    # 1. 요소 형식
    seen: set[int] = set()
    for eid, t, n in m.shells:
        if eid in seen:
            r.violations.append(f"요소 id 중복: {eid}")
        seen.add(eid)
        want = 4 if t == "CQUAD4" else 3
        if len(n) != want or len(set(n)) != want:
            r.violations.append(f"{t} {eid} 절점 구성 이상: {n}")
        for x in n:
            if x not in m.coords:
                r.violations.append(f"{t} {eid} 이 미정의 절점 {x} 참조")

    # 2. 1D 요소는 쉘 변 위에 있어야 한다
    shell_edges = set(rep.edge_shells)
    for eid, t, n in m.lines:
        if Edge(tuple(n)) not in shell_edges:
            r.violations.append(f"{t} {eid} 이 쉘 변 위에 없다: {n}")

    r.n_nonmanifold = sum(1 for v in rep.edge_shells.values() if len(v) > 2)
    r.area_total = float(sum(s.quality.area for s in rep.shells))

    if base is None:
        return r

    # 1D 요소가 몇 개 없어졌나 — 위반이 아니라 기록이다
    r.lines_removed = max(0, len(base.lines) - len(m.lines))

    brep = base.rep

    # 3. 비매니폴드 악화 금지
    b_nonman = sum(1 for v in brep.edge_shells.values() if len(v) > 2)
    if r.n_nonmanifold > b_nonman:
        r.violations.append(f"비매니폴드 변이 {b_nonman} → {r.n_nonmanifold} 로 늘었다")

    # 4. ext 절점 불변 + 삭제 금지
    for nid, is_ext in base.ext.items():
        if not is_ext:
            continue
        if nid not in m.coords:
            (r.warnings if regen else r.violations).append(
                f"ext 절점 {nid} 이 " + ("다른 절점에 합쳐졌다" if regen else "삭제됐다"))
        elif float(np.linalg.norm(base.P(nid) - m.P(nid))) > 1e-6:
            r.violations.append(f"ext 절점 {nid} 이 움직였다")
    r.max_node_move = max(
        (float(np.linalg.norm(base.P(n) - m.P(n))) for n in m.coords if n in base.coords),
        default=0.0,
    )

    # 5. 표면 보존 — 총 면적으로 본다.
    # 정확히 0 이 아닌 이유: 완전 평면이 아닌 요소를 합치거나 쪼개면 부채꼴 삼각분할
    # 면적이 미세하게 달라진다. 이면각 10° 상한(MAX_DIHEDRAL_DEG) 아래에서는 실측
    # 2e-4 이내라 1e-3 이면 충분히 조인다 — 요소를 하나 잃는 구조적 실수는 ~2% 라
    # 이 정도 허용오차로도 확실히 걸린다.
    b_area = float(sum(s.quality.area for s in brep.shells))
    rel_area = abs(r.area_total - b_area) / b_area if b_area > 0 else 0.0
    if rel_area > 1e-3:
        msg = (f"총 면적이 {b_area:.1f} → {r.area_total:.1f} 로 바뀌었다 "
               f"({100 * (r.area_total - b_area) / b_area:+.3f}%)")
        if regen and rel_area <= REGEN_AREA_TOL:
            r.warnings.append(msg)
        else:
            r.violations.append(msg)

    # 6. 손 안 댄 요소가 조용히 뒤집히지 않았는가.
    # 대응은 **요소 id + 절점 집합**으로 잡는다. 절점 집합만으로 잡으면 안 되는데,
    # 이 데이터셋은 쉘 감김 방향이 요소마다 제멋대로라(features.unsigned_normal 참고)
    # 편집으로 새로 생긴 요소가 우연히 다른 원본 요소와 같은 절점 집합을 가지면
    # 방향이 반대라는 이유만으로 거짓 위반이 뜬다.
    bmap = {
        s[0]: (frozenset(s[2]), brep.shells[i].quality.normal)
        for i, s in enumerate(base.shells)
    }
    for i, s in enumerate(m.shells):
        prev = bmap.get(s[0])
        if prev is None or prev[0] != frozenset(s[2]):
            continue  # 새로 생겼거나 절점이 바뀐 요소 — 비교 대상이 아니다
        if float(np.dot(prev[1], rep.shells[i].quality.normal)) < 0:
            r.violations.append(f"요소 {s[0]} 의 법선이 뒤집혔다 (절점은 그대로인데)")
    return r

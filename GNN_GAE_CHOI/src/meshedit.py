"""편집 가능한 메쉬 + 국소 편집 연산 + 도메인 불변식 검사.

수리의 토대. ``MeshRep`` 은 읽기 전용 파생 구조라 편집에 못 쓴다 — 대신 ``RawMesh`` 와
같은 모양(coords / shells / lines / ext)을 그대로 들고 편집하고, 평가할 때만 ``MeshRep`` 을
다시 만든다. 패치가 절점 55개 규모라 매번 다시 만들어도 싸다.

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
  merge_tris     삼각형 2개 → 사각형          swap_edge   대각선 뒤집기
  split_quad     사각형 → 삼각형 2개          move_node   절점 이동
  collapse_edge  절점 하나를 이웃에 합침 (사람 수정에서 사라진 절점의 94%)
  collapse_many  절점 병합 여러 개를 한꺼번에

메쉬 상태의 모양 (RawMesh 와 같다):
  coords  {절점 번호: (x, y, z)}
  shells  [[요소 번호, "CQUAD4"/"CTRIA3", [절점 번호들]], ...]
  lines   [[요소 번호, "CBAR"/"CBEAM", (절점 번호, 절점 번호)], ...]
  ext     {절점 번호: True/False}  — True 면 이웃 패치와 공유하는 절단면 절점
"""

from __future__ import annotations

import json

import numpy as np

import gae_paths  # noqa: F401  (부모 src 를 sys.path 에 넣는다)
import features as F
from bdf_io import RawMesh, load_raw_mesh
from meshrep import Edge, MeshRep

#: 두 삼각형을 합치거나 대각선을 뒤집어도 되는 이면각 상한(도).
#: 이보다 크면 서로 다른 패널로 보고 막는다.
#:
#: 10.0 이었는데 사람이 고친 275쌍을 재서 20.0 으로 올렸다. 실측 분포는 1° 이하가
#: 96.7%, 10° 이하가 99.3%, 최대 16.7° 다 — 10° 가드가 사람 수정의 0.7%를 막고 있었다.
#: 이면각이 큰 쌍도 **합친 결과 사각형의 warp 는 최대 2.4°** 로 거의 평면이다. 즉 그건
#: 실제로 꺾인 게 아니라 슬라이버 삼각형이라 법선이 불안정해서 크게 나온 값이다.
#: 표면이 실제로 바뀌는 것은 아래 면적 보존 검사가 막으므로, 이 값은 느슨해도 된다.
MAX_DIHEDRAL_DEG = 20.0

#: ``check()`` 가 보는 **메쉬 전체** 면적의 허용 변화. 요소를 하나 잃으면 ~2% 가 바뀌므로
#: 1e-3 이면 구조적 실수를 확실히 잡는다.
AREA_TOL = 1e-3

#: 병합/뒤집기 연산이 보는 **그 두 요소** 면적의 허용 변화.
#: 전체 기준과 분모가 다르므로 값도 달라야 한다 — 쌍의 면적이 0.8% 바뀌어도 메쉬
#: 전체로는 0.8% x (2/47) = 0.03% 라 AREA_TOL 과 충돌하지 않는다.
#: 1e-3 이었는데 사람이 고친 191개를 재서 1e-2 로 올렸다 — 4건이 0.14~0.80% 를
#: 바꾸며 표면을 평탄화했고(삼각형은 전부 멀쩡했다: 최소내각 20~49°), 그게 정답이다.
MERGE_AREA_TOL = 1e-2

#: ``check(..., regen=True)`` 와 절점 병합에서 총 면적 변화를 위반으로 보는 선.
#: 절점 병합은 경계 절점을 합치면 면적을 바꾼다 — 사람도 508쌍 중 107쌍에서 0.1% 넘게,
#: 최대 2.6% 바꿨다. 그 위로 넉넉히 둔다. 0.1% 와 이 값 사이는 경고로 남긴다.
REGEN_AREA_TOL = 5e-2


# ──────────────────────────────────────────────────────────────────────────
# 편집 연산 기록
# ──────────────────────────────────────────────────────────────────────────
class Op:
    """적용된(또는 후보) 편집 연산 하나.

    kind  "merge" | "swap" | "split" | "move" | "collapse"
    key   연산을 **절점 번호로** 식별한다 — 쉘 인덱스는 편집하면 바뀌지만 절점 번호는
          안 바뀌므로, 정답 순서를 저장했다가 나중에 다시 적용할 때 안전하다.
    info  덧붙인 정보 (지운 1D 요소, 지운 쉘 등)
    """

    def __init__(self, kind: str, key: tuple, info: dict | None = None):
        self.kind = kind
        self.key = key
        if info is None:
            info = {}
        self.info = info

    def __repr__(self) -> str:
        return f"Op({self.kind}, {self.key})"


class EditError(ValueError):
    """연산을 적용할 수 없을 때."""


def compact_nodes(node_ids: list):
    """절점 병합 뒤 요소의 절점 목록을 정리한다. 요소가 없어져야 하면 None.

    이웃한 중복(감김 순서상 붙은 두 절점이 같아진 것)은 하나로 합친다 — 사각형이
    삼각형이 된다. 그러고도 중복이 남으면(마주 보는 모서리가 겹친 사각형) 면적이 0 인
    '나비넥타이' 라 지운다. 3개 미만이면 찌그러져 없어진 것이다.
    """
    result = []
    for node_id in node_ids:
        if len(result) == 0:
            result.append(node_id)
        elif result[-1] != node_id:
            result.append(node_id)

    # 처음과 끝이 같으면 (감김이 한 바퀴 돌아 붙은 것) 끝을 뺀다
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()

    if len(result) < 3:
        return None
    if len(set(result)) != len(result):
        return None
    return result


def node_relation(shells, remove_node: int, keep_node: int):
    """``shells`` 안에서 두 절점의 관계. "edge"(변으로 이어짐) | "diag"(사각형 대각) | None.

    shells 의 각 항목은 ``[_, _, 절점 목록 또는 None, ...]`` 모양이다 (None 은 이미 없어진 요소).
    """
    found_as_diagonal = False
    for shell in shells:
        node_ids = shell[2]
        if node_ids is None:
            continue
        if remove_node not in node_ids:
            continue
        if keep_node not in node_ids:
            continue
        corner_count = len(node_ids)
        position_gap = (node_ids.index(remove_node) - node_ids.index(keep_node)) % corner_count
        if position_gap == 1 or position_gap == corner_count - 1:
            return "edge"
        found_as_diagonal = True

    if found_as_diagonal:
        return "diag"
    return None


def count_edge_use(shells) -> dict:
    """{변: 그 변을 쓰는 쉘 수}."""
    use_count = {}
    for shell in shells:
        node_ids = shell[2]
        corner_count = len(node_ids)
        for corner_index in range(corner_count):
            edge = Edge((node_ids[corner_index], node_ids[(corner_index + 1) % corner_count]))
            if edge not in use_count:
                use_count[edge] = 0
            use_count[edge] = use_count[edge] + 1
    return use_count


def count_nonmanifold_edges(shells) -> int:
    """세 개 이상의 쉘이 쓰는 변(비매니폴드 변)의 수."""
    nonmanifold_count = 0
    for use in count_edge_use(shells).values():
        if use > 2:
            nonmanifold_count = nonmanifold_count + 1
    return nonmanifold_count


def is_convex_quad(points: np.ndarray) -> bool:
    """사각형이 볼록한가 — 두 대각선 어느 쪽으로 잘라도 두 삼각형의 법선이 같은 쪽."""
    for start in (0, 1):
        point_a = points[start]
        point_b = points[(start + 1) % 4]
        point_c = points[(start + 2) % 4]
        point_d = points[(start + 3) % 4]
        normal_1 = np.cross(point_b - point_a, point_c - point_a)
        normal_2 = np.cross(point_c - point_a, point_d - point_a)
        if float(np.dot(normal_1, normal_2)) <= 0.0:
            return False
    return True


def number_text(value: float) -> str:
    """원본 .bdf 와 같은 십진 표기 (243680.0, -2840.0 처럼).

    정수값만 ``.1f`` 로 쓰고 나머지는 ``repr`` 을 쓴다 — ``repr`` 은 double 을 정확히
    되읽을 수 있는 가장 짧은 표기다. ``%.6g`` 를 쓰면 좌표가 24만 단위라 유효숫자가
    모자라 조용히 잘린다 (실측: 1127.4305 → 1127.43, 0.5mm 오차).
    """
    if abs(value - round(value)) < 1e-9:
        return f"{value:.1f}"
    return repr(value)


def shared_edge_of(mesh, shell_index_a: int, shell_index_b: int) -> frozenset:
    """두 쉘이 공유하는 절점들의 Edge (두 쉘의 절점 집합의 교집합)."""
    common_nodes = set(mesh.shells[shell_index_a][2]) & set(mesh.shells[shell_index_b][2])
    return Edge(tuple(common_nodes))


# ──────────────────────────────────────────────────────────────────────────
# 편집 가능한 메쉬
# ──────────────────────────────────────────────────────────────────────────
class EditableMesh:
    """``RawMesh`` 의 편집 가능한 사본.

    상태는 ``RawMesh`` 와 같은 모양이라 언제든 되돌려 쓸 수 있다 (``to_raw``,
    ``write_bdf``). ``rep`` 은 저장해 두고, 편집하면 자동으로 버린다.
    """

    def __init__(self, raw: RawMesh):
        self.name = raw.name
        self.coords = dict(raw.coords)
        self.shells = []
        for element_id, element_type, node_ids in raw.shells:
            self.shells.append([element_id, element_type, list(node_ids)])
        self.lines = []
        for element_id, element_type, node_ids in raw.lines:
            self.lines.append([element_id, element_type, tuple(node_ids)])
        self.ext = dict(raw.ext)
        #: 적용한 연산 기록 [Op, ...]
        self.log = []
        #: MeshRep 저장본 (편집하면 None 으로 버린다)
        self._saved_rep = None

    # ── 읽기 / 쓰기 ───────────────────────────────────────────────────
    @classmethod
    def load(cls, bdf_path: str, json_path: str | None = None) -> "EditableMesh":
        return cls(load_raw_mesh(bdf_path, json_path))

    def clone(self) -> "EditableMesh":
        """깊은 사본. 원본을 건드리지 않고 연산을 미리 해 볼 때 쓴다."""
        copy = EditableMesh.__new__(EditableMesh)
        copy.name = self.name
        copy.coords = dict(self.coords)
        copy.shells = []
        for shell in self.shells:
            copy.shells.append([shell[0], shell[1], list(shell[2])])
        copy.lines = []
        for line in self.lines:
            copy.lines.append([line[0], line[1], tuple(line[2])])
        copy.ext = dict(self.ext)
        copy.log = list(self.log)
        # MeshRep 은 읽기 전용이라 사본이 고쳐지기 전까지는 같이 써도 된다
        # (고치는 순간 _mark_changed() 가 끊는다). 후보 수백 개를 사본으로 미리 볼 때 다시 만들기를 아낀다.
        copy._saved_rep = self._saved_rep
        return copy

    def to_raw(self) -> RawMesh:
        shells = []
        for element_id, element_type, node_ids in self.shells:
            shells.append((element_id, element_type, list(node_ids)))
        lines = []
        for element_id, element_type, node_ids in self.lines:
            lines.append((element_id, element_type, tuple(node_ids)))
        return RawMesh(name=self.name, coords=dict(self.coords), shells=shells, lines=lines,
                       ext=dict(self.ext))

    def write_bdf(self, path: str) -> None:
        """원본과 같은 자유형식 쉼표 .bdf 로 쓴다 (PID 는 데이터셋 전체가 1)."""
        with open(path, "w", encoding="utf-8") as file_handle:
            file_handle.write(self.bdf_text())

    def bdf_text(self) -> str:
        """``write_bdf`` 가 쓰는 내용 그대로 (화면에서 내려받기용)."""
        output_lines = ["BEGIN BULK"]
        for node_id in sorted(self.coords):
            x, y, z = self.coords[node_id]
            output_lines.append(f"GRID,{node_id},,{number_text(x)},{number_text(y)},{number_text(z)}")
        for element_id, element_type, node_ids in self.shells:
            node_texts = []
            for node_id in node_ids:
                node_texts.append(str(node_id))
            output_lines.append(f"{element_type},{element_id},1," + ",".join(node_texts))
        for element_id, element_type, node_ids in self.lines:
            output_lines.append(f"{element_type},{element_id},1,{node_ids[0]},{node_ids[1]}")
        output_lines.append("ENDDATA")
        return "\n".join(output_lines) + "\n"

    def write_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file_handle:
            file_handle.write(self.json_text())

    def json_text(self) -> str:
        """{"절점 번호": true/false} 형식의 ext 정보."""
        ext_by_text_key = {}
        for node_id, is_ext in sorted(self.ext.items()):
            ext_by_text_key[str(node_id)] = bool(is_ext)
        return json.dumps(ext_by_text_key, indent=2)

    # ── 파생 구조 ─────────────────────────────────────────────────────
    @property
    def rep(self) -> MeshRep:
        """지금 상태의 MeshRep (위상·품질). 처음 부를 때 만들고 편집하기 전까지 다시 쓴다."""
        if self._saved_rep is None:
            self._saved_rep = MeshRep(self.to_raw())
        return self._saved_rep

    def _mark_changed(self) -> None:
        """편집했으니 저장해 둔 MeshRep 을 버린다."""
        self._saved_rep = None

    def point(self, node_id: int) -> np.ndarray:
        """절점 좌표 (numpy 배열)."""
        return np.array(self.coords[node_id], dtype=float)

    def points_of(self, node_ids) -> np.ndarray:
        """절점들의 좌표 (k, 3) 배열."""
        point_list = []
        for node_id in node_ids:
            point_list.append(self.coords[node_id])
        return np.array(point_list, dtype=float)

    def corners(self, shell_index: int) -> np.ndarray:
        """쉘 하나의 꼭짓점 좌표 (3 또는 4, 3) 배열."""
        return self.points_of(self.shells[shell_index][2])

    def quality(self, shell_index: int):
        """쉘 하나의 품질 (features.ShellQuality)."""
        return F.shell_quality(self.corners(shell_index))

    def tri_indices(self) -> list:
        """삼각형 쉘의 인덱스 목록."""
        triangle_indices = []
        for shell_index in range(len(self.shells)):
            if len(self.shells[shell_index][2]) == 3:
                triangle_indices.append(shell_index)
        return triangle_indices

    def next_element_id(self) -> int:
        """새 요소에 줄 번호 — 지금 쓰는 요소 번호 중 가장 큰 것 + 1."""
        all_ids = []
        for shell in self.shells:
            all_ids.append(shell[0])
        for line in self.lines:
            all_ids.append(line[0])
        if len(all_ids) == 0:
            return 1
        return max(all_ids) + 1

    def line_edges(self) -> set:
        """1D 요소가 올라타 있는 변 집합.

        이 변을 없애려면 그 위의 1D 요소도 같이 지워야 한다. 예전에는 "없애면 안 되는
        변" 으로 보고 무조건 거부했는데, 사람이 고친 508쌍을 재보니 **손댄 구역 안의
        1D 요소 1,886개 중 361개(19.1%)를 실제로 지웠다.** 금지 규칙이 아니라
        "같이 처리해야 하는 대상" 으로 다룬다 (``drop_lines`` 참고).
        """
        edges = set()
        for line in self.lines:
            edges.add(Edge(tuple(line[2])))
        return edges

    def lines_on(self, edge) -> list:
        """변 ``edge`` 위에 올라타 있는 1D 요소의 인덱스 목록."""
        target_edge = Edge(edge)
        line_indices = []
        for line_index in range(len(self.lines)):
            if Edge(tuple(self.lines[line_index][2])) == target_edge:
                line_indices.append(line_index)
        return line_indices

    # ── 연산 1: 인접 삼각형 2개 → 사각형 ───────────────────────────────
    def merge_tris(self, shell_index_a: int, shell_index_b: int,
                   max_dihedral: float = MAX_DIHEDRAL_DEG, drop_lines: bool = False) -> Op:
        """공유 변을 대각선으로 흡수해 CQUAD4 하나로 만든다.

        공유 변 위에 1D 요소(보강재)가 있으면 그 변을 없앨 때 1D 도 같이 없애야 한다.
        ``drop_lines=False`` 면 거부하고, True 면 1D 요소를 함께 지운다. 기본이 False 인
        이유는 1D 가 실제 구조 부재라 지우는 것이 결과가 큰 결정이기 때문 — 부르는 쪽이
        명시적으로 골라야 한다. 지운 요소는 ``Op.info["dropped_lines"]`` 에 남는다.
        """
        shared_edge = self._shared_edge(shell_index_a, shell_index_b)
        quad_nodes = self._merged_quad_nodes(shell_index_a, shell_index_b, max_dihedral, drop_lines)

        if drop_lines:
            dropped_lines = self._drop_lines_on(shared_edge)
        else:
            dropped_lines = []

        # 번호가 작은 쪽 자리에 사각형을 두고 큰 쪽을 지운다
        if shell_index_a < shell_index_b:
            keep_index = shell_index_a
            delete_index = shell_index_b
        else:
            keep_index = shell_index_b
            delete_index = shell_index_a
        self.shells[keep_index] = [self.shells[keep_index][0], "CQUAD4", quad_nodes]
        del self.shells[delete_index]
        self._mark_changed()

        operation = Op("merge", tuple(quad_nodes), {"dropped_lines": dropped_lines})
        self.log.append(operation)
        return operation

    def _shared_edge(self, shell_index_a: int, shell_index_b: int):
        """두 쉘이 공유하는 변. 공유 절점이 정확히 2개가 아니면 None."""
        shared_nodes = []
        for node_id in self.shells[shell_index_a][2]:
            if node_id in self.shells[shell_index_b][2]:
                shared_nodes.append(node_id)
        if len(shared_nodes) == 2:
            return Edge(tuple(shared_nodes))
        return None

    def _drop_lines_on(self, edge) -> list:
        """변 위의 1D 요소를 지우고 [(요소 번호, 종류, 절점)] 목록을 돌려준다."""
        if edge is None:
            return []
        line_indices = self.lines_on(edge)
        dropped = []
        for line_index in line_indices:
            line = self.lines[line_index]
            dropped.append((line[0], line[1], tuple(line[2])))
        # 뒤에서부터 지워야 앞 인덱스가 안 밀린다
        for line_index in sorted(line_indices, reverse=True):
            del self.lines[line_index]
        return dropped

    def _merged_quad_nodes(self, shell_index_a: int, shell_index_b: int,
                           max_dihedral: float = MAX_DIHEDRAL_DEG, drop_lines: bool = False) -> list:
        """두 삼각형을 합친 사각형의 절점 순서를 구한다. 합칠 수 없으면 ``EditError``."""
        shell_a = self.shells[shell_index_a]
        shell_b = self.shells[shell_index_b]
        if len(shell_a[2]) != 3 or len(shell_b[2]) != 3:
            raise EditError("merge 는 삼각형 2개에만 적용된다")

        shared_nodes = []
        for node_id in shell_a[2]:
            if node_id in shell_b[2]:
                shared_nodes.append(node_id)
        if len(shared_nodes) != 2:
            raise EditError(f"공유 절점이 2개가 아님: {shared_nodes}")
        if not drop_lines:
            if Edge(tuple(shared_nodes)) in self.line_edges():
                raise EditError("공유 변 위에 1D 요소가 있다 (drop_lines 로 함께 지울 수 있다)")

        # 두 삼각형이 서로 다른 패널에 있으면 합치거나 뒤집는 순간 선체 표면이 바뀐다.
        # 실측으로 이면각 90°(직교 패널) 쌍을 swap 해서 총 면적이 0.02% 변하는 사례가 나왔다.
        quality_a = F.shell_quality(self.points_of(shell_a[2]))
        quality_b = F.shell_quality(self.points_of(shell_b[2]))
        dihedral = F.dihedral_deg(quality_a.normal, quality_b.normal)
        if dihedral > max_dihedral:
            raise EditError(f"두 삼각형이 다른 패널에 있다 (이면각 {dihedral:.1f}°)")

        # 공유 변 밖의 꼭짓점: a 쪽 p, b 쪽 q
        apex_a = None
        for node_id in shell_a[2]:
            if node_id not in shared_nodes:
                apex_a = node_id
                break
        apex_b = None
        for node_id in shell_b[2]:
            if node_id not in shared_nodes:
                apex_b = node_id
                break

        # a 의 감김 방향을 유지하도록 p 를 맨 앞으로 돌린다 → (p, x, y), {x, y} == 공유 변
        apex_position = shell_a[2].index(apex_a)
        rotated = shell_a[2][apex_position:] + shell_a[2][:apex_position]
        shared_first = rotated[1]
        shared_second = rotated[2]
        quad_nodes = [apex_a, shared_first, apex_b, shared_second]
        if len(set(quad_nodes)) != 4:
            raise EditError("병합 결과에 중복 절점")

        quad_points = self.points_of(quad_nodes)
        # 오목(reflex) 사각형 방지. p 에서 부채꼴로 자른 두 삼각형의 법선이 서로
        # 반대면 대각선 p-q 가 도형 밖으로 나간 것 — 내각 하나가 180°를 넘는다.
        # features.interior_angles 는 acos 라 이런 각을 180° 미만으로 보고하므로
        # 품질 지표로는 걸러지지 않는다. 여기서 막아야 한다.
        # (실측: 얇은 슬라이버 삼각형 2개가 만나 내각 [11°, 182°, 11°, 156°] 같은
        #  화살촉 사각형이 되는 경우가 인접 삼각형 쌍의 29%)
        fan_normal_1 = np.cross(quad_points[1] - quad_points[0], quad_points[2] - quad_points[0])
        fan_normal_2 = np.cross(quad_points[2] - quad_points[0], quad_points[3] - quad_points[0])
        if float(np.dot(fan_normal_1, fan_normal_2)) <= 0:
            raise EditError("병합하면 오목(reflex) 사각형이 된다")

        # 표면 보존 — 이 연산의 진짜 유효성 조건이다.
        # 사각형 면적은 p 에서 부채꼴로 자른 (p,x,q)+(p,q,y) 의 합이므로, 이 값은
        # merge 결과의 면적이면서 동시에 **swap 결과 두 삼각형의 면적 합**이다.
        # 원래 두 삼각형의 합과 같아야 표면이 안 변한 것이다. 이 검사가
        #   - 다른 대각선으로 크게 접힌 4점 (이면각은 0.14°인데 warp 90°인 실측 사례)
        #   - 퇴화에 가까워 법선이 불안정한 쌍
        #   - 오목(dart) 사각형 — 합이 아니라 차가 나온다
        # 을 전부 한꺼번에 막는다. 이면각 검사만으로는 첫 두 개를 못 잡는다.
        triangles_area = quality_a.area + quality_b.area
        quad_area = F.area_normal(quad_points)[0]
        if triangles_area > 0:
            relative_change = abs(quad_area - triangles_area) / triangles_area
            if relative_change > MERGE_AREA_TOL:
                percent = 100 * (quad_area - triangles_area) / triangles_area
                raise EditError(
                    f"표면 면적이 바뀐다 ({triangles_area:.1f} → {quad_area:.1f}, {percent:+.2f}%)"
                )
        return quad_nodes

    def merged_quality(self, shell_index_a: int, shell_index_b: int,
                       max_dihedral: float = MAX_DIHEDRAL_DEG, drop_lines: bool = False):
        """실제로 합치지 않고 결과 사각형의 품질만 본다."""
        quad_nodes = self._merged_quad_nodes(shell_index_a, shell_index_b, max_dihedral, drop_lines)
        return F.shell_quality(self.points_of(quad_nodes))

    # ── 연산 2: 대각선 뒤집기 ─────────────────────────────────────────
    def swap_edge(self, shell_index_a: int, shell_index_b: int,
                  max_dihedral: float = MAX_DIHEDRAL_DEG, drop_lines: bool = False) -> Op:
        """공유 변을 가진 삼각형 2개의 대각선을 반대쪽으로 뒤집는다.

        뒤집으면 원래 대각선이 사라지므로, 그 위에 1D 요소가 있으면 ``merge_tris`` 와
        같은 선택이 필요하다 — 거부하거나(기본) 1D 를 같이 지우거나.
        """
        shared_edge = self._shared_edge(shell_index_a, shell_index_b)
        # 유효성 조건은 merge 와 같다
        quad_nodes = self._merged_quad_nodes(shell_index_a, shell_index_b, max_dihedral, drop_lines)
        apex_a, shared_first, apex_b, shared_second = quad_nodes
        shell_a = self.shells[shell_index_a]
        shell_b = self.shells[shell_index_b]

        # 새 대각선 p-q 가 이미 다른 곳에 있는 변이면 중복 변이 생긴다
        if Edge((apex_a, apex_b)) in self.rep.edge_shells:
            raise EditError(f"새 대각선 {apex_a}-{apex_b} 가 이미 존재하는 변이다")

        if drop_lines:
            dropped_lines = self._drop_lines_on(shared_edge)
        else:
            dropped_lines = []
        self.shells[shell_index_a] = [shell_a[0], "CTRIA3", [apex_a, shared_first, apex_b]]
        self.shells[shell_index_b] = [shell_b[0], "CTRIA3", [apex_a, apex_b, shared_second]]
        self._mark_changed()

        operation = Op("swap", (apex_a, shared_first, apex_b, shared_second), {"dropped_lines": dropped_lines})
        self.log.append(operation)
        return operation

    # ── 연산 3: 사각형 → 삼각형 2개 ─────────────────────────────────────
    def split_quad(self, shell_index: int, diag: int = 0) -> Op:
        """CQUAD4 를 대각선으로 갈라 CTRIA3 2개로 만든다. ``merge_tris`` 의 역연산.

        diag = 0 이면 a-c 대각선, 1 이면 b-d 대각선으로 자른다.
        """
        shell = self.shells[shell_index]
        if len(shell[2]) != 4:
            raise EditError("split 은 사각형에만 적용된다")
        node_a, node_b, node_c, node_d = shell[2]

        # 새로 생기는 대각선이 이미 메쉬에 있는 변이면 그 변의 인접 쉘이 3개가 되어
        # 비매니폴드가 늘어난다 (실측: SN2592_FWD_Clean_0497 에서 비매니폴드 변 6 → 7).
        if diag == 0:
            new_diagonal = Edge((node_a, node_c))
        else:
            new_diagonal = Edge((node_b, node_d))
        if new_diagonal in self.rep.edge_shells:
            raise EditError(f"새 대각선 {sorted(new_diagonal)} 가 이미 존재하는 변이다")

        if diag == 0:
            triangle_1 = [node_a, node_b, node_c]
            triangle_2 = [node_a, node_c, node_d]
        else:
            triangle_1 = [node_b, node_c, node_d]
            triangle_2 = [node_b, node_d, node_a]
        self.shells[shell_index] = [shell[0], "CTRIA3", triangle_1]
        self.shells.append([self.next_element_id(), "CTRIA3", triangle_2])
        self._mark_changed()

        operation = Op("split", (node_a, node_b, node_c, node_d), {"diag": diag})
        self.log.append(operation)
        return operation

    # ── 연산 4: 절점 이동 ─────────────────────────────────────────────
    def move_node(self, node_id: int, delta) -> Op:
        """절점을 옮긴다. ext 절점은 거부한다 (이웃 패치와 공유하는 절단면)."""
        if self.ext.get(node_id, False):
            raise EditError(f"절점 {node_id} 은 ext=true — 움직일 수 없다")
        x, y, z = self.coords[node_id]
        delta_array = np.asarray(delta, dtype=float)
        self.coords[node_id] = (x + float(delta_array[0]), y + float(delta_array[1]), z + float(delta_array[2]))
        self._mark_changed()

        operation = Op("move", (node_id,), {"delta": delta_array.tolist()})
        self.log.append(operation)
        return operation

    # ── 연산 5: 절점 병합 (edge collapse) ──────────────────────────────
    def collapse_relation(self, remove_node: int, keep_node: int):
        """두 절점이 합칠 수 있는 관계인가. "edge" | "diag" | None.

        사람이 합친 451쌍을 원본에서 재보면 변으로 이어진 이웃 432 / 사각형의 대각
        모서리 12 / 두 칸 떨어짐 7 이다. 앞의 둘(98.4%)만 받는다 — 두 칸 떨어진 것은
        다른 병합이 먼저 일어나 이웃이 된 뒤에야 합쳐진 것이다.
        """
        return node_relation(self.shells, remove_node, keep_node)

    def collapse_edge(self, remove_node: int, keep_node: int) -> Op:
        """절점 ``remove_node`` 를 ``keep_node`` 에 합친다. ``collapse_many([(remove, keep)])`` 와 같다."""
        return self.collapse_many([(remove_node, keep_node)])

    def collapse_many(self, pairs: list) -> Op:
        """절점 병합 여러 개를 **한꺼번에** 적용한다. ``pairs`` = [(remove, keep), ...]

        remove 의 자리가 사라지고, remove 를 쓰던 요소는 keep 을 쓰게 된다.

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

        remove 를 쓰던 요소는:
          - 삼각형이 찌그러지면(절점이 겹치면) 지운다
          - 사각형의 이웃한 두 모서리가 겹치면 삼각형이 된다 (이웃 변 붕괴)
          - 사각형의 마주 보는 두 모서리가 겹치면(대각 붕괴) 면적이 0 이 되므로 지운다
          - 이미 있는 요소와 절점 집합이 같아지면 겹친 요소라 지운다

        **거부**하는 경우 (``EditError``, 메쉬는 그대로):
          - 이미 사라진 절점을 합치려 한다 / 이웃(변 또는 사각형 대각)이 아니다
          - 남는 요소가 접히거나(법선 반전) 면적이 0 에 가까워진다
          - 남는 사각형이 오목해진다
          - 총 면적이 REGEN_AREA_TOL 넘게 바뀐다
          - 비매니폴드 변이 늘어난다
          - 1D 요소가 쉘 변 위를 벗어난다

        ext 절점의 자리를 없애는 것은 **거부하지 않는다** — 사람도 508쌍 중 48쌍에서 ext
        자리 60개를 없앴다(대부분 절단면을 따라 이웃한 ext 끼리). 대신 ``Op.info`` 에
        ``ext_removed`` 로 남겨, 검사(``check``)와 화면이 경고로 보여주게 한다.
        """
        if len(pairs) == 0:
            raise EditError("합칠 절점이 없다")

        # ── 1) 작업본에서 절점 바꾸기 ────────────────────────────────
        # 작업본 한 줄 = [요소 번호, 종류, 지금 절점 목록(없어지면 None), 원래 절점 목록]
        # 원래 절점 목록은 접힘 검사의 기준이다.
        work_shells = []
        for element_id, element_type, node_ids in self.shells:
            work_shells.append([element_id, element_type, list(node_ids), list(node_ids)])

        gone_nodes = set()
        relations = []
        for remove_node, keep_node in pairs:
            if remove_node == keep_node:
                raise EditError("같은 절점끼리는 합칠 수 없다")
            for node_id in (remove_node, keep_node):
                if node_id not in self.coords or node_id in gone_nodes:
                    raise EditError(f"없는(또는 이미 합쳐진) 절점: {node_id}")

            relation = node_relation(work_shells, remove_node, keep_node)
            if relation is None:
                raise EditError(f"{remove_node} 와 {keep_node} 는 이웃(변 또는 사각형 대각)이 아니다")
            relations.append(relation)

            for work_shell in work_shells:
                current_nodes = work_shell[2]
                if current_nodes is None:
                    continue
                if remove_node not in current_nodes:
                    continue
                replaced_nodes = []
                for node_id in current_nodes:
                    if node_id == remove_node:
                        replaced_nodes.append(keep_node)
                    else:
                        replaced_nodes.append(node_id)
                work_shell[2] = compact_nodes(replaced_nodes)
            gone_nodes.add(remove_node)

        # ── 2) 없어진 요소 / 남은 요소 나누기 ─────────────────────────
        removed_shells = []      # [(요소 번호, 종류, 원래 절점)]
        alive_shells = []        # [[요소 번호, 종류, 지금 절점, 원래 절점]]
        for element_id, element_type, current_nodes, original_nodes in work_shells:
            if current_nodes is None:
                removed_shells.append((element_id, element_type, tuple(original_nodes)))
            else:
                alive_shells.append([element_id, element_type, current_nodes, original_nodes])

        # 겹친 요소 — 모양이 바뀐 요소가 다른 요소와 절점 집합이 같아진 경우 바뀐 쪽을 지운다
        seen_node_sets = set()
        for alive_shell in alive_shells:
            if alive_shell[2] == alive_shell[3]:
                seen_node_sets.add(frozenset(alive_shell[2]))
        kept_shells = []
        for alive_shell in alive_shells:
            changed = alive_shell[2] != alive_shell[3]
            if changed:
                node_set = frozenset(alive_shell[2])
                if node_set in seen_node_sets:
                    removed_shells.append((alive_shell[0], alive_shell[1], tuple(alive_shell[3])))
                    continue
                seen_node_sets.add(node_set)
            kept_shells.append(alive_shell)

        # ── 3) 남는 요소의 모양 검사 ─────────────────────────────────
        # 사라진 자리를 쓰던 요소는 남는 자리로 늘어나며 접힐 수 있다
        for element_id, element_type, current_nodes, original_nodes in kept_shells:
            if current_nodes == original_nodes:
                continue
            original_area, original_normal = F.area_normal(self.points_of(original_nodes))
            current_area, current_normal = F.area_normal(self.points_of(current_nodes))
            if current_area <= 1e-9 * max(original_area, 1.0):
                raise EditError(f"요소 {element_id} 의 면적이 0 이 된다")
            if float(np.dot(original_normal, current_normal)) <= 0.0:
                raise EditError(f"요소 {element_id} 이 접힌다 (법선 반전)")
            if len(current_nodes) == 4:
                if not is_convex_quad(self.points_of(current_nodes)):
                    raise EditError(f"요소 {element_id} 이 오목한 사각형이 된다")

        # ── 4) 총 면적 검사 ─────────────────────────────────────────
        # 경계 절점을 합치면 면적이 바뀐다. 사람은 최대 2.6% 였다. 그 이상은
        # 표면을 크게 잘라먹는 병합 묶음이라 거부한다 (학습이 덜 된 모델이 병합 35개를
        # 한꺼번에 골라 면적 36%를 날린 것을 실제로 봤다).
        # 실수 합은 반드시 내장 sum() 으로 한다 — Python 3.12 부터 sum() 은 오차를 보정하며 더해서
        # for 문으로 하나씩 더한 값과 마지막 자리가 다를 수 있다 (원래 코드와 결과를 맞추기 위해)
        total_area = float(total_shell_area(self.rep))

        removed_areas = []
        for removed_shell in removed_shells:
            removed_areas.append(F.area_normal(self.points_of(removed_shell[2]))[0])
        area_change = -sum(removed_areas)
        for element_id, element_type, current_nodes, original_nodes in kept_shells:
            if current_nodes != original_nodes:
                area_change += (F.area_normal(self.points_of(current_nodes))[0]
                                - F.area_normal(self.points_of(original_nodes))[0])
        if total_area > 0:
            if abs(area_change) / total_area > REGEN_AREA_TOL:
                raise EditError(f"총 면적이 {100 * area_change / total_area:+.1f}% 바뀐다")

        new_shells = []
        for element_id, element_type, current_nodes, original_nodes in kept_shells:
            if len(current_nodes) == 4:
                new_type = "CQUAD4"
            else:
                new_type = "CTRIA3"
            new_shells.append([element_id, new_type, current_nodes])

        # ── 5) 1D 요소 — 번호를 옮기고, 길이가 0 이 되거나 겹친 것은 지운다 ──
        forward = {}
        for remove_node, keep_node in pairs:
            forward[remove_node] = keep_node

        new_lines = []
        dropped_lines = []
        seen_line_edges = set()
        for element_id, element_type, (node_a, node_b) in self.lines:
            final_a = follow_forward(forward, node_a)
            final_b = follow_forward(forward, node_b)
            line_edge = Edge((final_a, final_b))
            if final_a == final_b or line_edge in seen_line_edges:
                dropped_lines.append((element_id, element_type, (node_a, node_b)))
                continue
            seen_line_edges.add(line_edge)
            new_lines.append([element_id, element_type, (final_a, final_b)])

        # ── 6) 위상 검사 — 비매니폴드 변 수와 1D 위치 ─────────────────
        edge_use_before = count_edge_use(self.shells)
        edge_use_after = count_edge_use(new_shells)
        nonmanifold_before = 0
        for use in edge_use_before.values():
            if use > 2:
                nonmanifold_before = nonmanifold_before + 1
        nonmanifold_after = 0
        for use in edge_use_after.values():
            if use > 2:
                nonmanifold_after = nonmanifold_after + 1
        if nonmanifold_after > nonmanifold_before:
            raise EditError(f"비매니폴드 변이 {nonmanifold_before} → {nonmanifold_after} 로 늘어난다")

        lines_that_were_on_shell_edges = set()
        for line in self.lines:
            if Edge(tuple(line[2])) in edge_use_before:
                lines_that_were_on_shell_edges.add(line[0])
        for element_id, element_type, (node_a, node_b) in new_lines:
            if element_id in lines_that_were_on_shell_edges:
                if Edge((node_a, node_b)) not in edge_use_after:
                    raise EditError(f"1D 요소 {element_id} 가 쉘 변 위를 벗어난다")

        # ── 7) 적용 ─────────────────────────────────────────────────
        ext_removed = []
        for remove_node, keep_node in pairs:
            if self.ext.get(remove_node, False):
                ext_removed.append(remove_node)

        self.shells = new_shells
        self.lines = new_lines
        for remove_node, keep_node in pairs:
            # ext 자리를 합쳤다면 남는 절점이 이웃 패치와의 연결을 이어받는다
            final_keep = follow_forward(forward, keep_node)
            keep_is_ext = bool(self.ext.get(final_keep, False))
            remove_is_ext = bool(self.ext.get(remove_node, False))
            self.ext[final_keep] = keep_is_ext or remove_is_ext
        for remove_node, keep_node in pairs:
            del self.coords[remove_node]
            self.ext.pop(remove_node, None)
        self._mark_changed()

        pair_tuples = []
        for pair in pairs:
            pair_tuples.append(tuple(pair))
        operation = Op("collapse", tuple(pair_tuples), {
            "relations": relations, "removed_shells": removed_shells, "dropped_lines": dropped_lines,
            "ext_removed": ext_removed,
        })
        self.log.append(operation)
        return operation


def follow_forward(forward: dict, node_id: int) -> int:
    """병합 사슬을 따라간다. a→b, b→c 면 a 의 최종 자리는 c."""
    while node_id in forward:
        node_id = forward[node_id]
    return node_id


# ──────────────────────────────────────────────────────────────────────────
# 후보 열거
# ──────────────────────────────────────────────────────────────────────────
class MergeCand:
    """삼각형 병합 후보 하나 + 적용했을 때의 결과."""

    def __init__(self, si: int, sj: int, dihedral: float, quality, blocked, lines_on_edge: int = 0):
        #: 두 삼각형의 쉘 인덱스
        self.si = si
        self.sj = sj
        #: 두 삼각형 사이 이면각(도). 크면 서로 다른 패널이라 합치면 안 된다.
        self.dihedral = dihedral
        #: 병합 결과 사각형의 품질. 막힌 후보면 None.
        self.quality = quality
        #: 막혔다면 이유 문자열, 아니면 None
        self.blocked = blocked
        #: 공유 변 위에 올라타 있는 1D 요소 수. 0 이 아니면 합칠 때 같이 지워야 한다
        #: (금지가 아니다 — 사람도 손댄 구역 1D 의 19%를 지운다).
        self.lines_on_edge = lines_on_edge

    @property
    def ok(self) -> bool:
        """막히지 않은 후보인가."""
        return self.blocked is None


def merge_candidates(mesh: EditableMesh, drop_lines: bool = False) -> list:
    """변을 공유하는 삼각형 쌍 전부를, 막힌 것까지 이유와 함께 돌려준다.

    막힌 이유를 버리지 않는 건 진단 때문이다 — '1D 요소 때문에 못 합치는 쌍이 몇 %인가'
    가 수리 상한을 직접 좌우한다.
    """
    rep = mesh.rep
    triangle_indices = set(mesh.tri_indices())
    candidates = []
    for edge, shell_indices in rep.edge_shells.items():
        if len(shell_indices) != 2:
            continue
        if shell_indices[0] not in triangle_indices:
            continue
        if shell_indices[1] not in triangle_indices:
            continue
        shell_index_a = shell_indices[0]
        shell_index_b = shell_indices[1]
        dihedral = F.dihedral_deg(rep.shells[shell_index_a].quality.normal,
                                  rep.shells[shell_index_b].quality.normal)
        line_count = len(mesh.lines_on(edge))
        try:
            merged = mesh.merged_quality(shell_index_a, shell_index_b, drop_lines=drop_lines)
        except EditError as error:
            candidates.append(MergeCand(shell_index_a, shell_index_b, dihedral, None, str(error), line_count))
        else:
            candidates.append(MergeCand(shell_index_a, shell_index_b, dihedral, merged, None, line_count))
    return candidates


def collapse_greedy(mesh: EditableMesh, pairs: list):
    """병합 목록을 가능한 만큼 적용한다. ``(적용한 것, {거부한 것: 사유})``.

    전부 한꺼번에 되면 그대로 끝. 안 되면 **주어진 순서대로** 하나씩 보태 보며 묶음이
    여전히 유효한 것만 남긴다 (순서 = 우선순위 — 모델이 쓸 때는 점수 순). 앞의 병합이
    있어야 이웃이 되는 쌍이 있으므로 거부된 것은 진전이 없을 때까지 다시 훑는다.
    메쉬에는 남긴 것 전부를 ``collapse_many`` 한 번으로 적용한다.
    """
    pair_list = []
    for pair in pairs:
        pair_list.append(tuple(pair))
    if len(pair_list) == 0:
        return [], {}

    # 전부 한꺼번에 되는지 사본에 먼저 해 본다
    all_at_once_ok = True
    try:
        mesh.clone().collapse_many(pair_list)
    except EditError:
        all_at_once_ok = False
    if all_at_once_ok:
        mesh.collapse_many(pair_list)
        return pair_list, {}

    accepted = []
    reasons = {}
    pending = list(pair_list)
    while len(pending) > 0:
        made_progress = False
        for pair in list(pending):
            # 이미 쓰인 절점과 엉키는 것 — 사라진 절점을 또 합치거나, 사라진 자리에 합치기
            tangled = False
            for accepted_pair in accepted:
                if pair[0] == accepted_pair[0] or pair[1] == accepted_pair[0]:
                    tangled = True
            if tangled:
                reasons[pair] = "다른 병합과 겹친다"
                pending.remove(pair)
                continue

            try:
                mesh.clone().collapse_many(accepted + [pair])
            except EditError as error:
                reasons[pair] = str(error)
                continue
            accepted.append(pair)
            pending.remove(pair)
            reasons.pop(pair, None)
            made_progress = True
        if not made_progress:
            break

    if len(accepted) > 0:
        mesh.collapse_many(accepted)

    rejected = {}
    for pair in pair_list:
        if pair in reasons:
            rejected[pair] = reasons[pair]
    return accepted, rejected


def apply_collapse_seq(bad: EditableMesh, pairs: list) -> EditableMesh:
    """병합을 주어진 순서로 **하나씩** 적용한 사본 (반복형 재생성이 고른 순서 그대로)."""
    mesh = bad.clone()
    mesh.log.clear()
    for remove_node, keep_node in pairs:
        mesh.collapse_edge(remove_node, keep_node)
    return mesh


def apply_merges(mesh: EditableMesh, edges: list):
    """공유 변 목록대로 삼각형 쌍을 합친다 (제자리). ``(합친 변, 못 합친 사유)``.

    병합할 때마다 쉘 인덱스가 밀리므로 매번 변으로 다시 찾는다.
    """
    merged_edges = []
    notes = []
    for edge in edges:
        found = None
        for candidate in merge_candidates(mesh, drop_lines=True):
            if shared_edge_of(mesh, candidate.si, candidate.sj) == edge:
                found = candidate
                break

        if found is None:
            notes.append(f"{sorted(edge)}: 더 이상 삼각형 쌍이 아님")
            continue
        if not found.ok:
            notes.append(f"{sorted(edge)}: {found.blocked}")
            continue
        try:
            mesh.merge_tris(found.si, found.sj, drop_lines=True)
        except EditError as error:
            notes.append(f"{sorted(edge)}: {error}")
            continue
        merged_edges.append(edge)
    return merged_edges, notes


def replay_ops(bad: EditableMesh, collapses: list, merges: list) -> EditableMesh:
    """기록된 재생성 연산을 원본에 다시 적용한다 — 모델 없이 결과를 재현한다 (화면이 쓴다).

    반복형은 병합을 하나씩, oneshot 은 묶음으로 적용했다. 하나씩 먼저 해 보고 중간에
    거부되면 묶음으로 한다 — 둘 다 된다면 결과가 같다(check_edits '묶음 = 하나씩').
    """
    try:
        mesh = apply_collapse_seq(bad, collapses)
    except EditError:
        mesh = bad.clone()
        mesh.log.clear()
        if len(collapses) > 0:
            mesh.collapse_many(list(collapses))
    apply_merges(mesh, merges)
    return mesh


def collapse_pairs(mesh: EditableMesh) -> list:
    """합칠 수 있는 관계의 절점 쌍을 **양방향으로** 전부 돌려준다. [(remove, keep, relation)].

    방향이 결과를 가른다 — 어느 자리가 남느냐에 따라 주변 요소의 모양이 완전히 다르다.
    그래서 (a→b) 와 (b→a) 를 별개의 후보로 둔다. 실제로 합칠 수 있는지(접힘 등)는
    여기서 보지 않는다 — ``collapse_edge`` 를 시도해 봐야 안다.
    """
    relation_of_pair = {}
    for element_id, element_type, node_ids in mesh.shells:
        corner_count = len(node_ids)
        # 변으로 이어진 쌍은 언제나 "edge" (사각형 대각보다 우선)
        for corner_index in range(corner_count):
            node_a = node_ids[corner_index]
            node_b = node_ids[(corner_index + 1) % corner_count]
            pair_key = (min(node_a, node_b), max(node_a, node_b))
            relation_of_pair[pair_key] = "edge"
        # 사각형의 대각 쌍은 아직 없을 때만 "diag"
        if corner_count == 4:
            for node_a, node_b in ((node_ids[0], node_ids[2]), (node_ids[1], node_ids[3])):
                pair_key = (min(node_a, node_b), max(node_a, node_b))
                if pair_key not in relation_of_pair:
                    relation_of_pair[pair_key] = "diag"

    result = []
    for (node_a, node_b), relation in sorted(relation_of_pair.items()):
        result.append((node_a, node_b, relation))
        result.append((node_b, node_a, relation))
    return result


# ──────────────────────────────────────────────────────────────────────────
# 불변식 검사
# ──────────────────────────────────────────────────────────────────────────
class CheckReport:
    """``check`` 의 결과."""

    def __init__(self):
        #: 위반 문구 목록 (비어 있어야 OK)
        self.violations = []
        #: 위반은 아니지만 눈에 띄게 알려야 할 것 (``check(..., regen=True)`` 에서만 채운다)
        self.warnings = []
        self.n_shell = 0
        self.n_tri = 0
        self.n_nonmanifold = 0
        self.area_total = 0.0
        self.max_node_move = 0.0
        #: 원본 대비 사라진 1D 요소 수. 위반은 아니지만(사람도 손댄 구역 1D 의 19%를 지운다)
        #: 보강재가 조용히 없어지는 일이라 항상 눈에 띄게 적어 둔다.
        self.lines_removed = 0

    @property
    def ok(self) -> bool:
        return len(self.violations) == 0

    def __str__(self) -> str:
        if self.ok:
            head = "OK"
        else:
            head = f"위반 {len(self.violations)}건"
        body = ""
        for violation in self.violations:
            body = body + f"\n  - {violation}"
        for warning in self.warnings:
            body = body + f"\n  (경고) {warning}"
        if self.lines_removed:
            dropped_text = f" / 1D 삭제 {self.lines_removed}개"
        else:
            dropped_text = ""
        return (f"[{head}] 쉘 {self.n_shell}개 (삼각 {self.n_tri}) / "
                f"비매니폴드변 {self.n_nonmanifold}{dropped_text}{body}")


def total_shell_area(rep: MeshRep) -> float:
    """MeshRep 의 쉘 면적 합 (내장 sum() — 오차 보정 덧셈)."""
    areas = []
    for shell_entry in rep.shells:
        areas.append(shell_entry.quality.area)
    return float(sum(areas))


def check(mesh: EditableMesh, base: EditableMesh | None = None, regen: bool = False) -> CheckReport:
    """도메인 불변식을 검사한다. ``base`` 를 주면 '원본 대비 악화' 까지 본다.

    ``regen=True`` 는 재생성 결과(절점 병합이 섞인 결과)를 검사할 때 쓴다. 사람의 수정에서
    실제로 일어나는 두 가지를 위반이 아니라 **경고**로 돌린다:
      - ext 절점이 다른 절점에 합쳐져 사라짐 (508쌍 중 48쌍)
      - 총 면적이 0.1% 넘게 바뀜 (107쌍, 최대 2.6%) — ``REGEN_AREA_TOL`` 까지는 경고
    ext 절점이 **움직이는** 것은 여전히 위반이다.
    """
    report = CheckReport()
    rep = mesh.rep
    report.n_shell = len(mesh.shells)
    for shell in mesh.shells:
        if len(shell[2]) == 3:
            report.n_tri = report.n_tri + 1

    # 1. 요소 형식
    seen_element_ids = set()
    for element_id, element_type, node_ids in mesh.shells:
        if element_id in seen_element_ids:
            report.violations.append(f"요소 id 중복: {element_id}")
        seen_element_ids.add(element_id)
        if element_type == "CQUAD4":
            expected_corner_count = 4
        else:
            expected_corner_count = 3
        if len(node_ids) != expected_corner_count or len(set(node_ids)) != expected_corner_count:
            report.violations.append(f"{element_type} {element_id} 절점 구성 이상: {node_ids}")
        for node_id in node_ids:
            if node_id not in mesh.coords:
                report.violations.append(f"{element_type} {element_id} 이 미정의 절점 {node_id} 참조")

    # 2. 1D 요소는 쉘 변 위에 있어야 한다
    shell_edges = set(rep.edge_shells)
    for element_id, element_type, node_ids in mesh.lines:
        if Edge(tuple(node_ids)) not in shell_edges:
            report.violations.append(f"{element_type} {element_id} 이 쉘 변 위에 없다: {node_ids}")

    for shell_indices in rep.edge_shells.values():
        if len(shell_indices) > 2:
            report.n_nonmanifold = report.n_nonmanifold + 1
    report.area_total = total_shell_area(rep)

    if base is None:
        return report

    # 1D 요소가 몇 개 없어졌나 — 위반이 아니라 기록이다
    report.lines_removed = max(0, len(base.lines) - len(mesh.lines))

    base_rep = base.rep

    # 3. 비매니폴드 악화 금지
    base_nonmanifold = 0
    for shell_indices in base_rep.edge_shells.values():
        if len(shell_indices) > 2:
            base_nonmanifold = base_nonmanifold + 1
    if report.n_nonmanifold > base_nonmanifold:
        report.violations.append(f"비매니폴드 변이 {base_nonmanifold} → {report.n_nonmanifold} 로 늘었다")

    # 4. ext 절점 불변 + 삭제 금지
    for node_id, is_ext in base.ext.items():
        if not is_ext:
            continue
        if node_id not in mesh.coords:
            if regen:
                report.warnings.append(f"ext 절점 {node_id} 이 다른 절점에 합쳐졌다")
            else:
                report.violations.append(f"ext 절점 {node_id} 이 삭제됐다")
        elif float(np.linalg.norm(base.point(node_id) - mesh.point(node_id))) > 1e-6:
            report.violations.append(f"ext 절점 {node_id} 이 움직였다")

    largest_move = 0.0
    found_any = False
    for node_id in mesh.coords:
        if node_id not in base.coords:
            continue
        move = float(np.linalg.norm(base.point(node_id) - mesh.point(node_id)))
        if not found_any or move > largest_move:
            largest_move = move
            found_any = True
    report.max_node_move = largest_move

    # 5. 표면 보존 — 총 면적으로 본다.
    # 정확히 0 이 아닌 이유: 완전 평면이 아닌 요소를 합치거나 쪼개면 부채꼴 삼각분할
    # 면적이 미세하게 달라진다. 실측 2e-4 이내라 1e-3 이면 충분히 조인다 — 요소를 하나
    # 잃는 구조적 실수는 ~2% 라 이 정도 허용오차로도 확실히 걸린다.
    base_area = total_shell_area(base_rep)
    if base_area > 0:
        relative_area_change = abs(report.area_total - base_area) / base_area
    else:
        relative_area_change = 0.0
    if relative_area_change > AREA_TOL:
        percent = 100 * (report.area_total - base_area) / base_area
        message = (f"총 면적이 {base_area:.1f} → {report.area_total:.1f} 로 바뀌었다 "
                   f"({percent:+.3f}%)")
        if regen and relative_area_change <= REGEN_AREA_TOL:
            report.warnings.append(message)
        else:
            report.violations.append(message)

    # 6. 손 안 댄 요소가 조용히 뒤집히지 않았는가.
    # 대응은 **요소 번호 + 절점 집합**으로 잡는다. 절점 집합만으로 잡으면 안 되는데,
    # 이 데이터셋은 쉘 감김 방향이 요소마다 제멋대로라(features.unsigned_normal 참고)
    # 편집으로 새로 생긴 요소가 우연히 다른 원본 요소와 같은 절점 집합을 가지면
    # 방향이 반대라는 이유만으로 거짓 위반이 뜬다.
    base_shell_by_id = {}
    for shell_index in range(len(base.shells)):
        shell = base.shells[shell_index]
        base_shell_by_id[shell[0]] = (frozenset(shell[2]), base_rep.shells[shell_index].quality.normal)
    for shell_index in range(len(mesh.shells)):
        shell = mesh.shells[shell_index]
        if shell[0] not in base_shell_by_id:
            continue
        base_node_set, base_normal = base_shell_by_id[shell[0]]
        if base_node_set != frozenset(shell[2]):
            # 절점이 바뀐 요소 — 비교 대상이 아니다
            continue
        if float(np.dot(base_normal, rep.shells[shell_index].quality.normal)) < 0:
            report.violations.append(f"요소 {shell[0]} 의 법선이 뒤집혔다 (절점은 그대로인데)")
    return report

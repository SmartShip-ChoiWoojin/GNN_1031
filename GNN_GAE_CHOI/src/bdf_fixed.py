"""고정폭(fixed / large field) Nastran .bdf 리더.

사람이 고친 메쉬(`*_good.bdf`)는 FEGate 가 내보낸 **고정폭 8칸** 포맷이다. 원본
(`SN2592_FWD_Clean/*.bdf`)의 쉼표 구분 포맷과 달라서 부모 프로젝트의
``bdf_io.read_bdf`` 로는 못 읽는다. 읽은 결과는 같은 :class:`bdf_io.RawMesh` 로 돌려주므로
그 뒤 계층(meshrep / features / graphs)은 원본과 수정본을 구분하지 않는다.

공백으로 쪼개면 안 된다. 실측 예::

    GRID    15723           254080. -4765.433100.

``-4765.43`` 과 ``3100.`` 사이에 공백이 없다 — 칸 위치로만 갈린다.

large field(``GRID*``)도 받는다. 카드 이름 뒤 16칸씩이고 다음 줄 ``*`` 로 이어진다::

    GRID*   502                             242780.         10054.3
    *       2327.720208207

이 형식으로 적힌 절점은 전부 좌표가 길다(``2327.720208207``) — 8칸에 안 들어가서
large field 가 된 것이고, 곧 **사람이 수정하며 새로 만든 절점**이라는 표시다.
"""

from __future__ import annotations

import os
import re

import gae_paths  # noqa: F401  (부모 src 를 sys.path 에 넣는다)
from bdf_io import BdfFormatError, RawMesh, read_bdf, read_ext_json

#: 쉘 요소 종류 → 꼭짓점 수
SHELL_CORNER_COUNT = {"CQUAD4": 4, "CTRIA3": 3}

#: 1D 요소 종류
LINE_ELEMENT_TYPES = {"CBAR", "CBEAM"}

#: 읽되 무시하는 카드 — 해석 설정이라 형상과 무관하다
IGNORED_CARDS = {"PARAM", "BEGIN", "ENDDATA"}

#: Nastran 지수 축약형. "1.5+3" = 1.5E3, "2.0-4" = 2.0E-4
SHORT_EXPONENT_PATTERN = re.compile(r"([+-]?[\d.]+)([+-]\d+)")


def read_fixed_field(line: str, field_index: int, field_width: int = 8, start_column: int = 0) -> str:
    """고정폭 줄 ``line`` 에서 ``field_index`` 번째 칸의 글자를 공백을 떼고 돌려준다."""
    first_column = start_column + field_width * field_index
    last_column = first_column + field_width
    return line[first_column:last_column].strip()


def parse_number(text: str) -> float:
    """Nastran 실수 글자를 float 로. ``1.5+3`` 같은 지수 축약형(= 1.5e3)도 받는다."""
    text = text.strip()
    if text == "":
        return 0.0

    match = SHORT_EXPONENT_PATTERN.fullmatch(text)
    if match is None:
        return float(text)

    mantissa = match.group(1)
    exponent = match.group(2)
    return float(mantissa + "E" + exponent)


def is_free_field(path: str) -> bool:
    """쉼표 구분(free field) 파일인가.

    수정본 508개 중 4개(0131, 0284, 3067, 3076)는 고정폭이 아니라 원본과 같은 쉼표
    구분으로 저장돼 있다. 내보낸 도구가 달랐던 것으로 보인다 — 형식을 파일마다 본다.
    첫 번째 '내용이 있는 줄' 에 쉼표가 있으면 쉼표 구분 파일이다.
    """
    with open(path, encoding="utf-8", errors="replace") as file_handle:
        for line in file_handle:
            if line.strip() == "":
                continue
            if line.startswith("$"):
                continue
            if line.startswith("BEGIN") or line.startswith("ENDDATA"):
                continue
            if "," in line:
                return True
            else:
                return False
    return False


def read_bdf_fixed(path: str):
    """.bdf 하나를 (coords, shells, lines) 로 읽는다. 고정폭/쉼표 구분을 자동 판별한다.

    돌려주는 값:
        coords  {절점 번호: (x, y, z)}
        shells  [(요소 번호, "CQUAD4" 또는 "CTRIA3", [절점 번호들])]
        lines   [(요소 번호, "CBAR" 또는 "CBEAM", (절점 번호, 절점 번호))]
    """
    if is_free_field(path):
        return read_bdf(path)

    coords = {}
    shells = []
    lines = []

    with open(path, encoding="utf-8", errors="replace") as file_handle:
        all_lines = []
        for raw_line in file_handle:
            all_lines.append(raw_line.rstrip("\n"))

    line_number = 0
    while line_number < len(all_lines):
        line = all_lines[line_number]
        line_number = line_number + 1

        if line.strip() == "":
            continue
        if line.startswith("$"):
            continue

        card_name = read_fixed_field(line, 0)
        if card_name in IGNORED_CARDS:
            continue
        if card_name.startswith("BEGIN"):
            continue
        if line.startswith("*"):
            # 앞 카드(GRID*)에서 이미 읽은 연속행이다
            continue

        if card_name == "GRID":
            node_id = int(read_fixed_field(line, 1))
            if node_id in coords:
                raise BdfFormatError(f"{path}: GRID id 중복: {node_id}")
            x = parse_number(read_fixed_field(line, 3))
            y = parse_number(read_fixed_field(line, 4))
            z = parse_number(read_fixed_field(line, 5))
            coords[node_id] = (x, y, z)

        elif card_name == "GRID*":
            # 16칸 필드 2개(번호, 좌표계) + 좌표 2개. 나머지 좌표 1개는 다음 줄에 있다.
            node_id = int(read_fixed_field(line, 0, 16, start_column=8))
            x = parse_number(read_fixed_field(line, 2, 16, start_column=8))
            y = parse_number(read_fixed_field(line, 3, 16, start_column=8))

            has_next_line = line_number < len(all_lines)
            if not has_next_line:
                raise BdfFormatError(f"{path}: GRID* {node_id} 의 연속행이 없음")
            next_line = all_lines[line_number]
            if not next_line.startswith("*"):
                raise BdfFormatError(f"{path}: GRID* {node_id} 의 연속행이 없음")

            z = parse_number(read_fixed_field(next_line, 0, 16, start_column=8))
            line_number = line_number + 1
            if node_id in coords:
                raise BdfFormatError(f"{path}: GRID id 중복: {node_id}")
            coords[node_id] = (x, y, z)

        elif card_name in SHELL_CORNER_COUNT:
            corner_count = SHELL_CORNER_COUNT[card_name]
            element_id = int(read_fixed_field(line, 1))
            node_ids = []
            for corner_index in range(corner_count):
                node_ids.append(int(read_fixed_field(line, 3 + corner_index)))
            shells.append((element_id, card_name, node_ids))

        elif card_name in LINE_ELEMENT_TYPES:
            element_id = int(read_fixed_field(line, 1))
            first_node = int(read_fixed_field(line, 3))
            second_node = int(read_fixed_field(line, 4))
            lines.append((element_id, card_name, (first_node, second_node)))

        else:
            raise BdfFormatError(f"{path}: 예상치 못한 카드 {card_name!r}")

    return coords, shells, lines


def load_fixed_mesh(bdf_path: str, json_path: str | None = None) -> RawMesh:
    """고정폭 .bdf + 짝 .json 을 :class:`RawMesh` 로 만든다.

    부모 ``bdf_io.load_raw_mesh`` 와 같은 검증을 한다 — 미정의 절점 참조, 중복 절점,
    1D 요소의 양 끝이 같은 절점, json 의 절점 집합이 bdf 와 다른 경우.
    """
    mesh_name = os.path.splitext(os.path.basename(bdf_path))[0]
    if json_path is None:
        json_path = os.path.splitext(bdf_path)[0] + ".json"

    coords, shells, lines = read_bdf_fixed(bdf_path)
    if len(coords) == 0:
        raise BdfFormatError(f"{bdf_path} 에 GRID 가 없음")
    if len(shells) == 0:
        raise BdfFormatError(f"{bdf_path} 에 쉘 요소가 없음")

    # 쉘 요소 검사
    for element_id, element_type, node_ids in shells:
        missing_nodes = []
        for node_id in node_ids:
            if node_id not in coords:
                missing_nodes.append(node_id)
        if len(missing_nodes) > 0:
            raise BdfFormatError(f"{bdf_path} {element_type} {element_id} 이 미정의 절점 참조: {missing_nodes}")
        if len(set(node_ids)) != len(node_ids):
            raise BdfFormatError(f"{bdf_path} {element_type} {element_id} 에 중복 절점: {node_ids}")

    # 1D 요소 검사
    for element_id, element_type, (first_node, second_node) in lines:
        missing_nodes = []
        for node_id in (first_node, second_node):
            if node_id not in coords:
                missing_nodes.append(node_id)
        if len(missing_nodes) > 0:
            raise BdfFormatError(f"{bdf_path} {element_type} {element_id} 이 미정의 절점 참조: {missing_nodes}")
        if first_node == second_node:
            raise BdfFormatError(f"{bdf_path} {element_type} {element_id} 의 두 절점이 동일: {first_node}")

    # json 의 절점 집합이 bdf 와 같아야 한다
    ext = read_ext_json(json_path)
    bdf_node_set = set(coords)
    json_node_set = set(ext)
    if bdf_node_set != json_node_set:
        only_in_bdf = sorted(bdf_node_set - json_node_set)
        only_in_json = sorted(json_node_set - bdf_node_set)
        raise BdfFormatError(
            f"{mesh_name}: json 절점 집합이 bdf GRID 와 불일치 "
            f"(bdf에만 {len(only_in_bdf)}개 예:{only_in_bdf[:5]}, "
            f"json에만 {len(only_in_json)}개 예:{only_in_json[:5]})"
        )
    return RawMesh(name=mesh_name, coords=coords, shells=shells, lines=lines, ext=ext)

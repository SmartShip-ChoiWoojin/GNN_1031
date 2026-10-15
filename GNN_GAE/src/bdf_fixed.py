"""고정폭(fixed/large field) Nastran .bdf 리더.

수정본(`*_good.bdf`)은 FEGate 가 내보낸 **고정폭 8칸** 포맷이다. 원본
(`SN2592_FWD_Clean/*.bdf`)의 쉼표 구분 포맷과 달라서 부모 프로젝트의
``bdf_io.read_bdf`` 로는 못 읽는다. 결과는 같은 :class:`bdf_io.RawMesh` 로 돌려주므로
그 뒤 계층(meshrep / features / graph_*)은 원본과 수정본을 구분하지 않는다.

공백으로 쪼개면 안 된다. 실측 예::

    GRID    15723           254080. -4765.433100.

``-4765.43`` 과 ``3100.`` 사이에 공백이 없다 — 칸 위치로만 갈린다.

large field(``GRID*``)도 받는다. 카드명 뒤 16칸씩이고 다음 줄 ``*`` 로 이어진다::

    GRID*   502                             242780.         10054.3
    *       2327.720208207

이 형식으로 적힌 절점은 전부 좌표가 길다(``2327.720208207``) — 8칸에 안 들어가서
large field 가 된 것이고, 곧 **사람이 수정하며 새로 만든 절점**이라는 표시다.
"""

from __future__ import annotations

import os
import re

from bdf_io import BdfFormatError, RawMesh, read_ext_json

SHELL_CORNERS = {"CQUAD4": 4, "CTRIA3": 3}
LINE_TYPES = {"CBAR", "CBEAM"}
#: 읽되 무시하는 카드 — 해석 설정이라 형상과 무관하다
IGNORED = {"PARAM", "BEGIN", "ENDDATA"}

_EXP = re.compile(r"([+-]?[\d.]+)([+-]\d+)")


def _field(line: str, i: int, width: int = 8, start: int = 0) -> str:
    """``line`` 의 i 번째 고정폭 필드."""
    a = start + width * i
    return line[a: a + width].strip()


def parse_number(s: str) -> float:
    """Nastran 실수. ``1.5+3`` 같은 지수 축약형(= 1.5e3)도 받는다."""
    s = s.strip()
    if not s:
        return 0.0
    m = _EXP.fullmatch(s)
    return float(f"{m.group(1)}E{m.group(2)}") if m else float(s)


def is_free_field(path: str) -> bool:
    """쉼표 구분(free field) 파일인가.

    수정본 508개 중 4개(0131, 0284, 3067, 3076)는 고정폭이 아니라 원본과 같은 쉼표
    구분으로 저장돼 있다. 내보낸 도구가 달랐던 것으로 보인다 — 형식을 파일마다 본다.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith("$"):
                continue
            if ln.startswith(("BEGIN", "ENDDATA")):
                continue
            return "," in ln
    return False


def read_bdf_fixed(path: str) -> tuple[
    dict[int, tuple[float, float, float]],
    list[tuple[int, str, list[int]]],
    list[tuple[int, str, tuple[int, int]]],
]:
    """.bdf 하나를 (coords, shells, lines) 로 파싱한다. 고정폭/쉼표 구분을 자동 판별한다."""
    if is_free_field(path):
        from bdf_io import read_bdf

        return read_bdf(path)
    coords: dict[int, tuple[float, float, float]] = {}
    shells: list[tuple[int, str, list[int]]] = []
    lines: list[tuple[int, str, tuple[int, int]]] = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = [ln.rstrip("\n") for ln in fh]

    i = 0
    while i < len(raw):
        line = raw[i]
        i += 1
        if not line.strip() or line.startswith("$"):
            continue
        card = _field(line, 0)
        if card in IGNORED or card.startswith("BEGIN"):
            continue
        if line.startswith("*"):
            continue  # 앞 카드에서 이미 소비한 연속행

        if card == "GRID":
            nid = int(_field(line, 1))
            if nid in coords:
                raise BdfFormatError(f"{path}: GRID id 중복: {nid}")
            coords[nid] = tuple(parse_number(_field(line, 3 + k)) for k in range(3))

        elif card == "GRID*":
            # 16칸 필드 2개(id, cp) + 좌표 2개, 나머지 1개는 다음 줄
            nid = int(_field(line, 0, 16, start=8))
            x1 = parse_number(_field(line, 2, 16, start=8))
            x2 = parse_number(_field(line, 3, 16, start=8))
            if i >= len(raw) or not raw[i].startswith("*"):
                raise BdfFormatError(f"{path}: GRID* {nid} 의 연속행이 없음")
            x3 = parse_number(_field(raw[i], 0, 16, start=8))
            i += 1
            if nid in coords:
                raise BdfFormatError(f"{path}: GRID id 중복: {nid}")
            coords[nid] = (x1, x2, x3)

        elif card in SHELL_CORNERS:
            k = SHELL_CORNERS[card]
            shells.append((int(_field(line, 1)), card,
                           [int(_field(line, 3 + j)) for j in range(k)]))

        elif card in LINE_TYPES:
            lines.append((int(_field(line, 1)), card,
                          (int(_field(line, 3)), int(_field(line, 4)))))
        else:
            raise BdfFormatError(f"{path}: 예상치 못한 카드 {card!r}")

    return coords, shells, lines


def load_fixed_mesh(bdf_path: str, json_path: str | None = None) -> RawMesh:
    """고정폭 .bdf + 짝 .json 을 :class:`RawMesh` 로 만든다.

    부모 ``bdf_io.load_raw_mesh`` 와 같은 검증을 건다 — 미정의 절점 참조, 1D 요소의
    양 끝이 같은 절점, json 키 집합 불일치.
    """
    name = os.path.splitext(os.path.basename(bdf_path))[0]
    if json_path is None:
        json_path = os.path.splitext(bdf_path)[0] + ".json"

    coords, shells, lines = read_bdf_fixed(bdf_path)
    if not coords:
        raise BdfFormatError(f"{bdf_path} 에 GRID 가 없음")
    if not shells:
        raise BdfFormatError(f"{bdf_path} 에 쉘 요소가 없음")

    for eid, etype, nodes in shells:
        missing = [n for n in nodes if n not in coords]
        if missing:
            raise BdfFormatError(f"{bdf_path} {etype} {eid} 이 미정의 절점 참조: {missing}")
        if len(set(nodes)) != len(nodes):
            raise BdfFormatError(f"{bdf_path} {etype} {eid} 에 중복 절점: {nodes}")
    for eid, etype, (a, b) in lines:
        missing = [n for n in (a, b) if n not in coords]
        if missing:
            raise BdfFormatError(f"{bdf_path} {etype} {eid} 이 미정의 절점 참조: {missing}")
        if a == b:
            raise BdfFormatError(f"{bdf_path} {etype} {eid} 의 두 절점이 동일: {a}")

    ext = read_ext_json(json_path)
    if set(ext) != set(coords):
        only_bdf = sorted(set(coords) - set(ext))[:5]
        only_json = sorted(set(ext) - set(coords))[:5]
        raise BdfFormatError(
            f"{name}: json 절점 집합이 bdf GRID 와 불일치 "
            f"(bdf에만 {len(set(coords) - set(ext))}개 예:{only_bdf}, "
            f"json에만 {len(set(ext) - set(coords))}개 예:{only_json})"
        )
    return RawMesh(name=name, coords=coords, shells=shells, lines=lines, ext=ext)

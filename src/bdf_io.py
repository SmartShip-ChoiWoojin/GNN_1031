"""Nastran .bdf 벌크 데이터 + 절점 외부연결 .json 파서.

이 데이터셋(SN2592_FWD_Clean_*)은 형식이 매우 균일하다:
  - 카드는 GRID / CQUAD4 / CTRIA3 / CBAR / CBEAM 뿐
  - 전부 콤마 구분(free field), 한 줄에 하나, 연속 줄(continuation) 없음
  - GRID 는 6필드 (GRID, id, CP=빈칸, x, y, z), PID 는 항상 1

기존 SHI/Ver1/bdf.py 는 CBAR/CBEAM 을 인식하지 않아 재사용하지 않는다.
파서는 위 가정을 assert 로 검증하고, 어긋나면 조용히 넘기지 않고 실패한다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

SHELL_TYPES = ("CQUAD4", "CTRIA3")
LINE_TYPES = ("CBAR", "CBEAM")
ELEM_TYPES = SHELL_TYPES + LINE_TYPES

#: 요소 타입 -> one-hot 인덱스 (피처/그래프 빌더 전체가 이 순서를 공유)
TYPE_INDEX = {t: i for i, t in enumerate(ELEM_TYPES)}


class BdfFormatError(ValueError):
    """.bdf 가 이 데이터셋의 가정을 벗어날 때."""


@dataclass
class RawMesh:
    """.bdf + .json 을 읽은 그대로 담는 컨테이너. 파생 위상 계산은 meshrep.py."""

    name: str
    #: 절점 id -> (x, y, z)
    coords: dict[int, tuple[float, float, float]]
    #: (eid, type, [node ids]) — CQUAD4/CTRIA3, 파일 등장 순서
    shells: list[tuple[int, str, list[int]]]
    #: (eid, type, (n1, n2)) — CBAR/CBEAM, 파일 등장 순서
    lines: list[tuple[int, str, tuple[int, int]]]
    #: 절점 id -> 모델 외부 요소와 연결되어 있는지 (True 면 외부 연결)
    ext: dict[int, bool]

    @property
    def node_ids(self) -> list[int]:
        """정렬된 절점 id 목록. 그래프 노드 순서의 기준."""
        return sorted(self.coords)

    @property
    def points(self) -> np.ndarray:
        """(n_node, 3) 좌표 배열. 행 순서는 ``node_ids``."""
        return np.array([self.coords[n] for n in self.node_ids], dtype=float)


def _fields(line: str) -> list[str]:
    return [f.strip() for f in line.split(",")]


def read_bdf(path: str) -> tuple[
    dict[int, tuple[float, float, float]],
    list[tuple[int, str, list[int]]],
    list[tuple[int, str, tuple[int, int]]],
]:
    """.bdf 하나를 (coords, shells, lines) 로 파싱한다."""
    coords: dict[int, tuple[float, float, float]] = {}
    shells: list[tuple[int, str, list[int]]] = []
    lines: list[tuple[int, str, tuple[int, int]]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("$"):
                continue
            if line.startswith("BEGIN") or line.startswith("ENDDATA"):
                continue

            f = _fields(line)
            card = f[0]

            if card == "GRID":
                if len(f) != 6:
                    raise BdfFormatError(
                        f"{path}:{lineno} GRID 필드 수가 6이 아님 ({len(f)}): {line!r}"
                    )
                nid = int(f[1])
                if nid in coords:
                    raise BdfFormatError(f"{path}:{lineno} GRID id 중복: {nid}")
                coords[nid] = (float(f[3]), float(f[4]), float(f[5]))

            elif card in SHELL_TYPES:
                want = 7 if card == "CQUAD4" else 6
                if len(f) != want:
                    raise BdfFormatError(
                        f"{path}:{lineno} {card} 필드 수가 {want}가 아님 ({len(f)}): {line!r}"
                    )
                shells.append((int(f[1]), card, [int(x) for x in f[3:]]))

            elif card in LINE_TYPES:
                if len(f) != 5:
                    raise BdfFormatError(
                        f"{path}:{lineno} {card} 필드 수가 5가 아님 ({len(f)}): {line!r}"
                    )
                lines.append((int(f[1]), card, (int(f[3]), int(f[4]))))

            else:
                raise BdfFormatError(f"{path}:{lineno} 예상치 못한 카드: {card!r}")

    return coords, shells, lines


def read_ext_json(path: str) -> dict[int, bool]:
    """절점 외부연결 .json 을 {절점 id: bool} 로 읽는다."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[int, bool] = {}
    for k, v in raw.items():
        if not isinstance(v, bool):
            raise BdfFormatError(f"{path} 값이 bool 이 아님: {k}={v!r}")
        out[int(k)] = v
    return out


def load_raw_mesh(bdf_path: str, json_path: str | None = None) -> RawMesh:
    """.bdf 와 짝 .json 을 읽어 :class:`RawMesh` 로 만든다.

    json 의 키 집합은 bdf 의 GRID id 집합과 정확히 일치해야 한다 (전 1602개 모델에서 확인됨).
    불일치는 짝이 틀린 파일이라는 신호이므로 예외로 처리한다.
    """
    name = os.path.splitext(os.path.basename(bdf_path))[0]
    if json_path is None:
        json_path = os.path.splitext(bdf_path)[0] + ".json"

    coords, shells, lines = read_bdf(bdf_path)
    if not coords:
        raise BdfFormatError(f"{bdf_path} 에 GRID 가 없음")
    if not shells:
        raise BdfFormatError(f"{bdf_path} 에 쉘 요소가 없음")

    # 요소가 참조하는 절점이 모두 정의되어 있는지
    for eid, etype, nodes in shells:
        missing = [n for n in nodes if n not in coords]
        if missing:
            raise BdfFormatError(f"{bdf_path} {etype} {eid} 이 미정의 절점 참조: {missing}")
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


def discover_pairs(folder: str) -> tuple[dict[str, tuple[str, str]], list[str], list[str]]:
    """폴더에서 .bdf/.json 짝을 찾는다.

    Returns:
        (짝이 맞는 {name: (bdf, json)}, json 없는 bdf 이름, bdf 없는 json 이름)

    이 데이터셋에는 알려진 불일치가 두 건 있다:
    ``SN2592_FWD_Clean_1603`` (json 없음), ``SN2592_FWD_Clean_4000`` (bdf 없는 고아 json).
    """
    bdfs, jsons = {}, {}
    for fn in os.listdir(folder):
        stem, ext = os.path.splitext(fn)
        if ext.lower() == ".bdf":
            bdfs[stem] = os.path.join(folder, fn)
        elif ext.lower() == ".json":
            jsons[stem] = os.path.join(folder, fn)

    paired = {n: (bdfs[n], jsons[n]) for n in sorted(bdfs.keys() & jsons.keys())}
    return paired, sorted(bdfs.keys() - jsons.keys()), sorted(jsons.keys() - bdfs.keys())

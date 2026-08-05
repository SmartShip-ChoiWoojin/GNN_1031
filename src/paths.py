"""프로젝트 경로 상수. 다른 모듈은 여기서만 경로를 가져온다."""

from __future__ import annotations

import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)

#: 데이터 폴더. "fe model 1D 0706"(1603개, 일부 누락)은 구버전이고,
#: "SN2592_FWD_Clean"이 라벨 엑셀 4000행과 완전히 일치하는 전체 데이터셋이다
#: (discover_pairs 로 확인: bdf/json 짝 4000개, 누락/고아 파일 0개).
FE_DIR = os.path.join(ROOT, "SN2592_FWD_Clean")
LABEL_XLSX = os.path.join(ROOT, "fe model 1D index 0721.xlsx")

CACHE_DIR = os.path.join(ROOT, "cache")
RUNS_DIR = os.path.join(ROOT, "runs")


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RUNS_DIR):
        os.makedirs(d, exist_ok=True)

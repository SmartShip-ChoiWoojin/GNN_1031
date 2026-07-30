"""프로젝트 경로 상수. 다른 모듈은 여기서만 경로를 가져온다."""

from __future__ import annotations

import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)

FE_DIR = os.path.join(ROOT, "fe model 1D 0706")
LABEL_XLSX = os.path.join(ROOT, "fe model 1D index 0721.xlsx")

CACHE_DIR = os.path.join(ROOT, "cache")
RUNS_DIR = os.path.join(ROOT, "runs")


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RUNS_DIR):
        os.makedirs(d, exist_ok=True)

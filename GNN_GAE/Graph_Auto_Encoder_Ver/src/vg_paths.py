"""경로 상수 + import 설정.

이 모듈을 import 하면 ``GNN_GAE/src`` (쌍 불러오기, 편집 연산, 메쉬 비교)와 그 부모
``GNN_Claude/src`` (bdf 파싱, 메쉬 위상)가 sys.path 에 들어간다.
"""

from __future__ import annotations

import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)                  # Graph_Auto_Encoder_Ver/
GAE_ROOT = os.path.dirname(ROOT)                 # GNN_GAE/
GAE_SRC = os.path.join(GAE_ROOT, "src")

if GAE_SRC not in sys.path:
    sys.path.insert(0, GAE_SRC)

import gae_paths as gp  # noqa: E402,F401  (부모 src 도 여기서 sys.path 에 들어간다)

CACHE_DIR = os.path.join(ROOT, "cache")
RUNS_DIR = os.path.join(ROOT, "runs")
OUT_DIR = os.path.join(ROOT, "out")
DATA_PATH = os.path.join(CACHE_DIR, "pairs_node.pkl")


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RUNS_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)

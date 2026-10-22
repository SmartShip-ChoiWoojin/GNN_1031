"""Graph_Auto_Encoder_Ver 경로 상수 + import 설정.

이 모듈을 import 하면 ``GNN_GAE/src`` (쌍 비교·메쉬 차이) 와 그 부모 ``GNN_Claude/src``
(파싱·그래프 변환·정규화) 가 sys.path 에 들어간다. 이름을 ``ae_paths`` 로 둔 건
``paths`` / ``gae_paths`` 와 겹치지 않게 하려는 것이다.
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

import gae_paths as gp  # noqa: E402  (부모 src 도 여기서 sys.path 에 들어간다)

PARENT_CACHE = gp._p.CACHE_DIR
#: 부모 프로젝트가 만들어 둔 요소기반 그래프 캐시 (corner_edges/include_lines 기본값)
PARENT_ELEM_CACHE = os.path.join(PARENT_CACHE, "graphs_elem_d7f31fefbb.pkl")
#: 재생성 작업의 고정 90/10 분할 — 같은 테스트 셋에서 비교하려고 그대로 읽는다
REGEN_SPLIT = os.path.join(gp.RUNS_DIR, "regen_split.csv")
#: node_baseline.py 결과 (절점 단위 지도학습 기준선과 같은 표본에서 비교)
NODE_TABLE = os.path.join(gp.OUT_DIR, "node_table.csv")

CACHE_DIR = os.path.join(ROOT, "cache")
RUNS_DIR = os.path.join(ROOT, "runs")


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RUNS_DIR):
        os.makedirs(d, exist_ok=True)

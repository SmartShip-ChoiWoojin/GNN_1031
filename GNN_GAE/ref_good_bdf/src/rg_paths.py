"""ref_good_bdf 경로 상수 + import 설정.

이 폴더는 "학습할 때 good(사람이 고친 .bdf)을 정답 라벨 이상으로 참고하면 나아지나" 를
따로 실험하는 곳이다. 재생성 파이프라인은 ``GNN_GAE/src`` 것을 그대로 쓰고, 결과(runs/,
out/)만 여기에 둬서 기존 ``GNN_GAE/runs`` 와 섞이지 않게 한다.

주의: ``regen_data`` 의 분할 파일 경로는 import 순간 ``GNN_GAE/runs/regen_split.csv`` 로
고정된다. 그 뒤에 ``gp.RUNS_DIR`` / ``gp.OUT_DIR`` 를 이 폴더로 돌린다 — 그래서 분할은
기존과 같은 것을 읽고, 새 결과만 여기로 간다. :func:`redirect` 를 다른 GNN_GAE 모듈보다
먼저 부르면 안 되는 이유다.
"""

from __future__ import annotations

import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)                  # ref_good_bdf/
GAE_ROOT = os.path.dirname(ROOT)                 # GNN_GAE/
GAE_SRC = os.path.join(GAE_ROOT, "src")

for p in (GAE_SRC, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import gae_paths as gp  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "runs")
OUT_DIR = os.path.join(ROOT, "out")
#: 원래 재생성 실행이 있는 곳 (비교 기준 regen_final)
GAE_RUNS_DIR = gp.RUNS_DIR


def redirect() -> None:
    """분할 파일 경로를 고정한 뒤, 결과가 이 폴더에 쌓이도록 경로를 돌린다."""
    import regen_data  # noqa: F401  (SPLIT_PATH 를 원래 위치로 먼저 고정)

    gp.RUNS_DIR = RUNS_DIR
    gp.OUT_DIR = OUT_DIR
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

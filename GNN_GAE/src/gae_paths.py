"""GNN_GAE 경로 상수 + 부모 프로젝트 import 설정.

분류 프로젝트(``GNN_Claude/src``)를 그대로 재사용한다 — 파싱·기하·그래프 변환·학습된
분류기까지 전부 검증된 코드라 다시 만들 이유가 없다. 이 모듈을 import 하면 부모 ``src``
가 sys.path 에 들어간다.

모듈 이름이 ``paths`` 가 아닌 이유: 부모 프로젝트에 이미 ``paths`` 가 있어서 같은 이름을
쓰면 부모 모듈들이 우리 것을 집어간다.
"""

from __future__ import annotations

import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)           # GNN_GAE/
PARENT_ROOT = os.path.dirname(ROOT)       # GNN_Claude/
PARENT_SRC = os.path.join(PARENT_ROOT, "src")

if PARENT_SRC not in sys.path:
    sys.path.insert(0, PARENT_SRC)

# Windows 콘솔 기본 코드페이지가 cp949 라 한글과 em-dash 출력이 깨진다.
# 이 모듈을 import 하는 모든 스크립트가 UTF-8 로 출력하도록 한 번에 고정한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

import paths as _p  # noqa: E402  (sys.path 조작 뒤여야 한다)

#: 원본 데이터와 라벨 — 분류 프로젝트와 같은 것을 본다
FE_DIR = _p.FE_DIR
LABEL_XLSX = _p.LABEL_XLSX

#: 분류기 심판. 학습에는 절대 쓰지 않고 평가에만 쓰는 봉인 모델.
JUDGE_RUN = os.path.join(_p.RUNS_DIR, "final")

CACHE_DIR = os.path.join(ROOT, "cache")
RUNS_DIR = os.path.join(ROOT, "runs")
#: 수리된 메쉬를 .bdf/.json 으로 내보내는 곳
OUT_DIR = os.path.join(ROOT, "out")


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RUNS_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)

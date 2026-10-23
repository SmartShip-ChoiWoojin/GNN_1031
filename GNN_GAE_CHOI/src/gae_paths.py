"""GNN_GAE_CHOI 의 경로 상수 + 부모 프로젝트(GNN_Claude/src) import 설정.

이 모듈을 import 하면 부모 프로젝트의 src 폴더가 sys.path 에 들어간다.
부모 프로젝트의 파싱(bdf_io), 기하(features), 그래프 변환(graphs), 분류기(models, explain)를
그대로 다시 쓰기 때문이다.

폴더 구성:
    GNN_Claude/
        src/                 부모 프로젝트 (파싱·기하·그래프·분류기)
        GNN_GAE/             원래 코드. 사람이 고친 메쉬 데이터(data_folder)는 여기 것을 읽는다
        GNN_GAE_CHOI/        이 폴더 (다시 짠 코드)
            src/             코드
            runs/            분할 파일, 학습된 모델 (GNN_GAE/runs 에서 복사해 왔다)
            out/             결과 파일 (.bdf, .html, .csv)
            cache/           1단계 후보 캐시

모듈 이름이 paths 가 아닌 이유: 부모 프로젝트에 이미 paths 모듈이 있어서
같은 이름을 쓰면 부모 모듈들이 우리 것을 집어간다.
"""

from __future__ import annotations

import os
import sys

# ──────────────────────────────────────────────────────────────────────────
# 폴더 위치
# ──────────────────────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))     # GNN_GAE_CHOI/src
ROOT = os.path.dirname(SRC_DIR)                           # GNN_GAE_CHOI
PARENT_ROOT = os.path.dirname(ROOT)                       # GNN_Claude
PARENT_SRC = os.path.join(PARENT_ROOT, "src")             # GNN_Claude/src
ORIGINAL_GAE_ROOT = os.path.join(PARENT_ROOT, "GNN_GAE")  # 원래 코드 폴더

# 부모 프로젝트의 src 를 import 경로 맨 앞에 넣는다
if PARENT_SRC not in sys.path:
    sys.path.insert(0, PARENT_SRC)

# Windows 콘솔 기본 코드페이지가 cp949 라 한글 출력이 깨질 수 있다.
# 이 모듈을 import 하는 모든 스크립트가 UTF-8 로 출력하도록 여기서 한 번 고정한다.
for output_stream in [sys.stdout, sys.stderr]:
    try:
        output_stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# sys.path 를 고친 뒤에야 부모 프로젝트의 paths 를 import 할 수 있다
import paths as parent_paths  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
# 데이터 위치
# ──────────────────────────────────────────────────────────────────────────
#: 원본 메쉬(.bdf/.json) 폴더와 라벨 엑셀 — 분류 프로젝트와 같은 것을 본다
FE_DIR = parent_paths.FE_DIR
LABEL_XLSX = parent_paths.LABEL_XLSX

#: 사람이 직접 고친 메쉬(<이름>_good.bdf)가 들어 있는 폴더. 원래 GNN_GAE 의 것을 그대로 읽는다.
GOOD_DIR = os.path.join(ORIGINAL_GAE_ROOT, "data_folder", "2026_10_08_n509")

#: 분류기 심판. 학습에는 절대 쓰지 않고 평가에만 쓰는 봉인 모델.
JUDGE_RUN = os.path.join(parent_paths.RUNS_DIR, "final")

#: 1단계 후보 캐시 / 학습 결과 / 출력 파일
CACHE_DIR = os.path.join(ROOT, "cache")
RUNS_DIR = os.path.join(ROOT, "runs")
OUT_DIR = os.path.join(ROOT, "out")


def ensure_dirs() -> None:
    """캐시·학습결과·출력 폴더가 없으면 만든다."""
    for folder in [CACHE_DIR, RUNS_DIR, OUT_DIR]:
        os.makedirs(folder, exist_ok=True)

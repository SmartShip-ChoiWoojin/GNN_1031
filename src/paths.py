"""프로젝트 경로 상수. 다른 모듈은 여기서만 경로를 가져온다."""

from __future__ import annotations

import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__)) # abspath : 해당 경로를 절대경로로 반환
                                                     # os.path.dirname() : 절대경로에서 파일명을 떼고 그 파일이 들어있는 폴더 경로만 추출
ROOT = os.path.dirname(SRC_DIR)

#: 데이터 폴더. "fe model 1D 0706"(1603개, 일부 누락)은 구버전이고,
#: "SN2592_FWD_Clean"이 라벨 엑셀 4000행과 완전히 일치하는 전체 데이터셋이다
#: (discover_pairs 로 확인: bdf/json 짝 4000개, 누락 파일 0개).
FE_DIR = os.path.join(ROOT, "SN2592_FWD_Clean")
LABEL_XLSX = os.path.join(ROOT, "fe model 1D index 0918.xlsx")

# 그래프 변환 캐시와 학습 실행 결과가 쌓이는 디렉토리
CACHE_DIR = os.path.join(ROOT, "cache") # pickle
RUNS_DIR = os.path.join(ROOT, "runs")

#: 실행 결과는 용도별로 나눠 둔다. 섞어 두면 어느 것이 배포 중인 모델인지 알 수 없고,
#: 실제로 그 혼동 때문에 PC 마다 다른 판정이 나오는 사고가 있었다.
#:   train/    - 받은 데이터 전부로 학습한 배포용 모델. 앱은 여기만 본다.
#:   validate/ - 성능을 재기 위한 5-fold 실행. 배포하지 않는다.
#:   validate/archive/ - 옛 라벨(0721)로 돌린 실험들. 지금 수치와 비교하면 안 된다.
RUNS_TRAIN = os.path.join(RUNS_DIR, "train")
RUNS_VALIDATE = os.path.join(RUNS_DIR, "validate")
RUNS_ARCHIVE = os.path.join(RUNS_VALIDATE, "archive")

# 위 두 디렉터리를 os.makedirs(exist_ok=True)로 만든다. train.py/dataset.py가 캐시를 쓰기 직전에 호출한다 — 존재 확인과 생성을 매번 따로 하지 않도록. 
    # (08/20 : 잘 이해가 안된다. 나중에 다시 확인하러 와보자)

def ensure_dirs() -> None: 
    for d in (CACHE_DIR, RUNS_DIR, RUNS_TRAIN, RUNS_VALIDATE):
        os.makedirs(d, exist_ok=True)

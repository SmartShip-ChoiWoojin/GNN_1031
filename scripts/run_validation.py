"""성능을 잰다. 5-fold 교차검증으로, 배포 모델과 같은 설정으로.

여기서 만들어지는 fold 모델 5개는 **버리는 모델**이다. 배포하지 않는다. 목적은
"이 설정의 성능이 얼마인가"라는 숫자 하나를 얻는 것뿐이다. 그래서 결과가
runs/validate/ 에 따로 쌓인다.

왜 따로 두는가: 예전에 측정용 fold 모델을 그대로 배포했다가, "이 메쉬를 몇 번 모델이
판정할지"를 실행할 때마다 다시 계산하는 구조가 되었고, 그 계산이 학습 대상 목록의
순서에 의존해서 PC 마다 다른 판정이 나오는 사고가 있었다. 측정과 배포는 분리한다.

    python scripts/run_validation.py                 # 기본 설정으로 5-fold
    python scripts/run_validation.py --tag cv1003    # 이름을 지정
    python scripts/run_validation.py --baseline      # 기준선(집계피처+GBM)도 같이
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import paths

#: scripts/train_deploy.py 의 MODEL_ARGS 와 같아야 한다. 다르면 여기서 잰 성능이
#: 배포 모델의 성능이라고 말할 수 없다.
MODEL_ARGS = [
    "--graph", "elem",
    "--conv", "gatv2",
    "--no-graph-features",
    "--separate-encoder",
]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="5-fold 교차검증으로 성능 측정")
    ap.add_argument("--tag", default=None,
                    help="실행 이름 (기본: cv + 오늘 날짜)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--baseline", action="store_true",
                    help="기준선(집계 피처 + GBM)도 같이 측정한다")
    ap.add_argument("--compare", action="store_true",
                    help="끝난 뒤 검증 실행들의 비교표를 출력한다")
    a = ap.parse_args()

    py = sys.executable
    tag = a.tag or "cv" + datetime.date.today().strftime("%m%d")

    if a.baseline:
        print("[기준선] 집계 피처 + HistGradientBoosting")
        run([py, os.path.join(ROOT, "src", "baseline.py")])

    print(f"\n[검증] {a.folds}-fold 교차검증  (결과: {os.path.join(paths.RUNS_VALIDATE, tag)})")
    run([py, os.path.join(ROOT, "src", "train.py"), *MODEL_ARGS,
         "--folds", str(a.folds), "--tag", tag])

    if a.compare:
        print("\n[비교표]")
        run([py, os.path.join(ROOT, "src", "compare.py")])

    print(f"\n완료. 이 수치가 보고용 성능이다: {os.path.join(paths.RUNS_VALIDATE, tag)}")
    print("배포 모델은 scripts/train_deploy.py 로 따로 만든다 (여기 모델은 쓰지 않는다).")
    return 0


if __name__ == "__main__":
    from fasoo_drm import setting_fasoo

    setting_fasoo()
    sys.exit(main())

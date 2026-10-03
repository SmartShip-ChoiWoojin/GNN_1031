"""배포용 모델을 만든다. 받은 데이터 전부로 학습한 모델 하나.

두 단계로 돈다. 검증셋 없이 바로 학습하면 "몇 epoch 에서 멈출지"와 "확률 보정 계수"를
정할 근거가 없기 때문이다.

  1단계 (calib) : 15% 를 검증셋으로 떼고 학습한다. 조기종료가 알려주는 최적 epoch 와
                  Platt 보정 계수를 여기서 얻는다. 이미 있으면 건너뛴다.
  2단계 (deploy): 검증셋 없이 받은 데이터 전부로, 1단계가 알려준 epoch 만큼만 학습한다.
                  보정 계수는 1단계 것을 그대로 쓴다.

마지막에 predict_all 로 예측 캐시까지 만든다. 앱이 오분류 목록을 즉시 띄우려면 필요하다.

    python scripts/train_deploy.py
    python scripts/train_deploy.py --recalibrate     # 1단계부터 다시
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import paths

#: 배포 모델의 구조 설정. 바꾸려면 검증(scripts/run_validation.py)도 같은 설정으로
#: 다시 돌려서 성능을 다시 재야 한다.
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
    ap = argparse.ArgumentParser(description="배포용 단일 모델 학습")
    ap.add_argument("--recalibrate", action="store_true",
                    help="1단계(calib)를 다시 돌려 최적 epoch 와 보정 계수를 새로 구한다")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="1단계에서 조기종료·보정용으로 뗄 비율")
    a = ap.parse_args()

    py = sys.executable
    train_py = os.path.join(ROOT, "src", "train.py")
    calib_dir = os.path.join(paths.RUNS_TRAIN, "calib")
    calib_json = os.path.join(calib_dir, "result.json")

    # ── 1단계 ────────────────────────────────────────────────
    if a.recalibrate or not os.path.isfile(calib_json):
        print("[1/3] 보정 실행 - 최적 epoch 와 확률 보정 계수를 구한다")
        run([py, train_py, *MODEL_ARGS, "--full",
             "--val-frac", str(a.val_frac), "--tag", "calib"])
    else:
        print(f"[1/3] 보정 실행이 이미 있다 - 건너뛴다 ({calib_dir})")

    with open(calib_json, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    best_epoch = int(calib["metrics"]["best_epoch"])
    a_, b_ = calib["metrics"]["platt_a"], calib["metrics"]["platt_b"]
    print(f"      최적 epoch {best_epoch},  보정 계수 a={a_:.3f} b={b_:.3f}")

    # ── 2단계 ────────────────────────────────────────────────
    print(f"\n[2/3] 배포 모델 학습 - 전 데이터, {best_epoch} epoch, 검증셋 없음")
    run([py, train_py, *MODEL_ARGS, "--full",
         "--val-frac", "0", "--epochs", str(best_epoch),
         "--platt-from", os.path.join(calib_dir, "full.pt"), "--tag", "deploy"])

    # ── 3단계 ────────────────────────────────────────────────
    print("\n[3/3] 예측 캐시 생성")
    run([py, os.path.join(ROOT, "src", "predict_all.py"),
         "--run", os.path.join(paths.RUNS_TRAIN, "deploy")])

    print(f"\n완료. 배포 모델: {os.path.join(paths.RUNS_TRAIN, 'deploy')}")
    print("화면으로 보려면:  streamlit run src/app.py")
    return 0


if __name__ == "__main__":
    from fasoo_drm import setting_fasoo

    setting_fasoo()
    sys.exit(main())

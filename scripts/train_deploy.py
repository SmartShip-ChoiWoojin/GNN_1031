"""배포용 모델을 만든다. 받은 데이터 전부로 학습한 모델 하나.

두 단계로 돈다. 검증셋 없이 바로 학습하면 "몇 epoch 에서 멈출지"와 "확률 보정 계수"를
정할 근거가 없기 때문이다.

  1단계 (calib) : 평가 전용 몫(--holdout)을 뺀 나머지에서 15% 를 검증셋으로 떼고 학습한다. 조기종료가 알려주는 최적 epoch 와
                  Platt 보정 계수를 여기서 얻는다. 이미 있으면 건너뛴다.
  2단계 (deploy): 검증셋 없이, 평가 전용 몫을 뺀 나머지로 1단계가 알려준 epoch 만큼 학습한다.
                  보정 계수는 1단계 것을 그대로 쓴다.

마지막에 predict_all 로 예측 캐시까지 만든다. 앱이 오분류 목록을 즉시 띄우려면 필요하다.

    python scripts/train_deploy.py
    python scripts/train_deploy.py --recalibrate     # 1단계부터 다시
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    ap.add_argument("--tag-suffix", default=None,
                    help="실행 이름 뒤에 붙일 라벨 버전 (기본: 라벨 엑셀 파일명에서 뽑는다). "
                         "calib<접미사> / deploy<접미사> 로 저장된다.")
    ap.add_argument("--recalibrate", action="store_true",
                    help="1단계(calib)를 다시 돌려 최적 epoch 와 보정 계수를 새로 구한다")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="1단계에서 조기종료·보정용으로 뗄 비율")
    ap.add_argument("--holdout", type=float, default=0.1,
                    help="평가 전용으로 떼어 둘 비율. 두 단계 모두 같은 몫을 뺀다 - "
                         "보정 단계가 이 데이터를 보면 epoch 선택이 이미 평가 데이터에 "
                         "맞춰진 것이 된다. 0 이면 전부 학습한다.")
    a = ap.parse_args()

    py = sys.executable
    train_py = os.path.join(ROOT, "src", "train.py")

    # 라벨 버전을 이름에 박아둔다. 안 그러면 라벨을 갈아끼웠을 때 어느 라벨로 만든
    # 모델인지 폴더만 보고는 알 수 없다.
    suffix = a.tag_suffix
    if suffix is None:
        m = re.search(r"(\d{4})", os.path.basename(paths.LABEL_XLSX))
        suffix = m.group(1) if m else ""
    calib_tag, deploy_tag = f"calib{suffix}", f"deploy{suffix}"
    print(f"실행 이름: {calib_tag} -> {deploy_tag}  (라벨 {os.path.basename(paths.LABEL_XLSX)})")

    calib_dir = os.path.join(paths.RUNS_TRAIN, calib_tag)
    calib_json = os.path.join(calib_dir, "result.json")

    # ── 1단계 ────────────────────────────────────────────────
    if a.recalibrate or not os.path.isfile(calib_json):
        print("[1/3] 보정 실행 - 최적 epoch 와 확률 보정 계수를 구한다")
        run([py, train_py, *MODEL_ARGS, "--full",
             "--val-frac", str(a.val_frac), "--holdout", str(a.holdout), "--tag", calib_tag])
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
         "--val-frac", "0", "--holdout", str(a.holdout), "--epochs", str(best_epoch),
         "--platt-from", os.path.join(calib_dir, "full.pt"), "--tag", deploy_tag])

    # ── 3단계 ────────────────────────────────────────────────
    print("\n[3/3] 예측 캐시 생성")
    run([py, os.path.join(ROOT, "src", "predict_all.py"),
         "--run", os.path.join(paths.RUNS_TRAIN, deploy_tag)])

    print(f"\n완료. 배포 모델: {os.path.join(paths.RUNS_TRAIN, 'deploy')}")
    print("화면으로 보려면:  streamlit run src/app.py")
    return 0


if __name__ == "__main__":
    from fasoo_drm import setting_fasoo

    setting_fasoo()
    sys.exit(main())

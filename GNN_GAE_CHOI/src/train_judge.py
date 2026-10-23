"""심판 분류기를 학습한다 — 수정 쌍 508개를 **학습에서 완전히 빼고**.

왜 따로 만드나: 수리 모델을 채점하려면 "고친 결과가 정상인가" 를 판정할 분류기가 필요한데,
그 분류기가 원본을 학습에 썼다면 채점이 자기 채점이 된다. 그래서 **원본 508개를 명단에서
빼고** 부모 프로젝트의 절차(`scripts/train_deploy.py`)로 하나 더 만든다.

    python src/train_judge.py              # 명단 생성 + 2단계 학습
    python src/train_judge.py --names-only # 명단만 만들고 끝낸다

결과: `../runs/train/deploy_10_14` (보정 중간본은 `calib_10_14`). 이미 있으면 다시 돌릴 필요 없다.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import gae_paths
import paths as parent_paths
from bdf_io import discover_pairs
from labels import load_label_table
from pairs import all_pairs

#: 실행 이름 접미사. 부모 규칙이 `<역할><접미사>` 라 deploy_10_14 / calib_10_14 가 된다.
TAG_SUFFIX = "_10_14"

#: 학습 명단 파일. 부모의 common_names.txt 옆에 둔다.
NAMES_PATH = os.path.join(parent_paths.RUNS_DIR, f"names{TAG_SUFFIX}.txt")


def write_names(path: str = NAMES_PATH):
    """학습 가능한 메쉬에서 수정 쌍 원본을 뺀 명단을 쓴다. (쓴 개수, 제외한 개수)."""
    label_table = load_label_table(parent_paths.LABEL_XLSX)
    mesh_paths, _, _ = discover_pairs(parent_paths.FE_DIR)
    trainable_names = []
    for row in label_table.itertuples():
        if row.trainable and row.name in mesh_paths:
            trainable_names.append(row.name)

    excluded = set(all_pairs())
    kept_names = []
    for name in trainable_names:
        if name not in excluded:
            kept_names.append(name)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(kept_names) + "\n")
    return len(kept_names), len(trainable_names) - len(kept_names)


def main() -> int:
    parser = argparse.ArgumentParser(description="수정 쌍을 제외한 심판 분류기 학습")
    parser.add_argument("--names-only", action="store_true", help="명단만 만들고 끝낸다")
    parser.add_argument("--holdout", type=float, default=0.1, help="심판 자신의 성능을 정직하게 재려고 떼어 둘 비율")
    parser.add_argument("--recalibrate", action="store_true")
    args = parser.parse_args()

    kept_count, dropped_count = write_names()
    print(f"[judge] 학습 명단 {kept_count}개  (수정 쌍 원본 {dropped_count}개 제외)")
    print(f"        {NAMES_PATH}")
    if args.names_only:
        return 0

    script_path = os.path.join(parent_paths.ROOT, "scripts", "train_deploy.py")
    command = [sys.executable, script_path, "--tag-suffix", TAG_SUFFIX, "--names-from", NAMES_PATH,
               "--holdout", str(args.holdout)]
    if args.recalibrate:
        command.append("--recalibrate")
    print("\n$ " + " ".join(command))
    completed = subprocess.run(command, cwd=parent_paths.ROOT)
    if completed.returncode != 0:
        return completed.returncode

    out_dir = os.path.join(parent_paths.RUNS_TRAIN, f"deploy{TAG_SUFFIX}")
    print(f"\n완료. 심판 모델: {out_dir}")
    print("수정 전/후를 이 모델로 채점하려면:")
    print(f"  python src/judge_pairs.py --run {out_dir}")
    return 0


if __name__ == "__main__":
    from fasoo_drm import setting_fasoo

    setting_fasoo()
    gae_paths.ensure_dirs()
    sys.exit(main())

"""기존 regen_eval.py 를 이 폴더의 실행에 대해 돌린다 (good 참고 채점기를 읽을 수 있게 등록만 한다).

    python src/rg_eval.py --run runs/regen_aux05 --split cv     # 학습 셋 out-of-fold
    python src/rg_eval.py --run runs/regen_aux05                # 테스트 셋 — 마지막에 한 번만

``--run`` 의 상대 경로는 ref_good_bdf/ 기준이다. 재생성 .bdf 는 ref_good_bdf/out/<tag>/ 에 쓴다.
"""

from __future__ import annotations

import os
import sys

import rg_paths

rg_paths.redirect()

import regen_eval  # noqa: E402
import rg_model  # noqa: E402,F401  (gnn_aux 채점기 등록)


def main() -> None:
    args = sys.argv[1:]
    if "--run" in args:
        i = args.index("--run") + 1
        if not os.path.isabs(args[i]):
            args[i] = os.path.join(rg_paths.ROOT, args[i])
    sys.argv = [sys.argv[0]] + args
    regen_eval.main()


if __name__ == "__main__":
    main()

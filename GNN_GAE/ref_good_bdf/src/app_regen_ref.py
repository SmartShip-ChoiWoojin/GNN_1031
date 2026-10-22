"""재생성 결과 화면 — 기존 app_regen.py 를 그대로 쓰되 목록에 두 곳의 실행을 함께 띄운다.

    streamlit run GNN_GAE/ref_good_bdf/src/app_regen_ref.py

목록: 기존 ``GNN_GAE/runs/regen_*`` (예: regen_final) + 이 폴더의 ``ref_good_bdf/runs/regen_*``.
화면은 기록된 연산을 다시 적용할 뿐 모델을 돌리지 않으므로 고칠 것이 목록뿐이다.
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rg_paths  # noqa: E402

APP = os.path.join(rg_paths.GAE_SRC, "app_regen.py")


def _list_runs() -> list[str]:
    out = []
    for d in (rg_paths.GAE_RUNS_DIR, rg_paths.RUNS_DIR):
        out += sorted(os.path.dirname(p) for p in glob.glob(os.path.join(d, "regen_*", "model.json")))
    return out


# app_regen 의 모듈 수준 코드가 list_runs() 를 부르기 전에 바꿔 끼워야 하므로,
# 소스의 함수 정의 뒤에 덮어쓰는 한 줄을 넣어 실행한다.
src = open(APP, encoding="utf-8").read().replace("\r\n", "\n")
anchor = "\n@st.cache_data\ndef run_meta"
if anchor not in src:
    raise RuntimeError("app_regen.py 구조가 바뀌었습니다 — app_regen_ref.py 의 anchor 를 고쳐야 합니다")
src = src.replace(anchor, "\nlist_runs = __rg_list_runs__\n" + anchor, 1)
g = {"__name__": "__main__", "__file__": APP, "__rg_list_runs__": _list_runs}
exec(compile(src, APP, "exec"), g)

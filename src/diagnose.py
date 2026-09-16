"""배포본이 설치된 PC 의 상태를 한 번에 찍어 보는 진단 스크립트.

같은 메쉬인데 PC 마다 판정이 다른 문제를 추적하기 위한 것. 판정에 쓰는 fold 는
"학습대상 목록 안에서 몇 번째인가"로 정해지므로, 그 목록이 PC 마다 같은지만
확인하면 원인이 갈린다:

  * 목록 지문이 다르다  -> 라벨 엑셀을 읽는 결과가 다르다 (DRM/서식 문제)
  * 목록 지문은 같은데 캐시 지문이 다르다 -> 캐시가 잘못된 시점에 만들어졌다

출력은 cmd 창 인코딩 문제를 피하려고 전부 ASCII 로만 찍는다.
"""

from __future__ import annotations

import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fasoo_drm import setting_fasoo

setting_fasoo()   # 암호화된 라벨 엑셀을 읽어야 하므로 가장 먼저

import numpy as np

import paths
from bdf_io import discover_pairs
from data_index import list_samples


def fp(names) -> str:
    return hashlib.sha1("\n".join(names).encode()).hexdigest()[:8]


def main() -> int:
    print("=" * 60)
    print(" FE MESH VIEWER - DIAGNOSTIC REPORT")
    print("=" * 60)

    # 1) 환경
    import sklearn
    print("\n[1] ENVIRONMENT")
    print(f"    python      : {sys.version.split()[0]}")
    print(f"    numpy       : {np.__version__}")
    print(f"    sklearn     : {sklearn.__version__}")
    print(f"    fasoo dll   : {os.path.isfile('C:\\Windows\\System32\\f_nxldr.dll')}")

    # 2) 데이터 폴더
    print("\n[2] DATA FOLDER")
    try:
        paired, bdf_only, json_only = discover_pairs(paths.FE_DIR)
        print(f"    bdf+json pairs : {len(paired)}      (expected 4000)")
        print(f"    bdf missing json : {len(bdf_only)}")
        print(f"    json missing bdf : {len(json_only)}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return 1

    # 3) 라벨 엑셀 -> 학습대상 목록 (fold 를 정하는 바로 그 목록)
    print("\n[3] LABEL TABLE  << most important")
    try:
        samples = list_samples(verbose=False)
        names = [s.name for s in samples]
        print(f"    trainable count : {len(names)}      (expected 3214)")
        print(f"    list fingerprint: {fp(names)}      (expected 7e18665e)")
        ok_list = (len(names) == 3214 and fp(names) == "7e18665e")
        print(f"    MATCH           : {'YES' if ok_list else 'NO  <<< PROBLEM HERE'}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return 1

    # 4) 캐시 — 앱이 실제로 fold 계산에 쓰는 목록
    print("\n[4] CACHE")
    cache_files = sorted(glob.glob(os.path.join(paths.CACHE_DIR, "graphs_*.pkl")))
    if not cache_files:
        print("    (no cache yet - it is built on first run)")
        cnames = None
    else:
        import pickle
        for c in cache_files:
            mb = os.path.getsize(c) / 1e6
            print(f"    file : {os.path.basename(c)}  ({mb:.0f} MB)")
        try:
            with open(cache_files[0], "rb") as fh:
                blob = pickle.load(fh)
            cnames = blob["names"]
            print(f"    cached count    : {len(cnames)}      (expected 3214)")
            print(f"    cache fingerprint: {fp(cnames)}      (expected 7e18665e)")
            ok_cache = (len(cnames) == 3214 and fp(cnames) == "7e18665e")
            print(f"    MATCH           : {'YES' if ok_cache else 'NO  <<< PROBLEM HERE'}")
        except Exception as e:
            print(f"    ERROR reading cache: {e}")
            cnames = None

    # 5) 문제의 두 메쉬가 어느 fold 로 가는지
    print("\n[5] FOLD ASSIGNMENT for the reported meshes")
    from sklearn.model_selection import StratifiedKFold
    use = cnames if cnames else names
    lab = {s.name: s.cls for s in samples}
    y = np.array([lab.get(n, 0) for n in use])
    sp = StratifiedKFold(5, shuffle=True, random_state=0)
    out = np.full(len(y), -1, int)
    for k, (_, te) in enumerate(sp.split(np.zeros(len(y)), y), 1):
        out[te] = k
    pos = {n: i for i, n in enumerate(use)}
    for t, exp in (("SN2592_FWD_Clean_2003", 2), ("SN2592_FWD_Clean_2004", 2)):
        i = pos.get(t)
        got = int(out[i]) if i is not None else None
        tag = "OK" if got == exp else "MISMATCH <<<"
        print(f"    {t} : fold {got}   (expected {exp})  {tag}")

    print("\n" + "=" * 60)
    print(" Please send this whole file back.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

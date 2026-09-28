"""배포본이 설치된 PC 의 상태를 한 번에 찍어 보는 진단 스크립트.

배포 모델이 하나(full.pt)로 바뀐 뒤로는 확인할 것이 단순해졌다:
모두가 **같은 라벨 목록**과 **같은 모델 파일**을 쓰고 있는가.

  * 목록 지문이 다르다   -> 라벨 엑셀을 읽은 결과가 다르다 (DRM/서식 문제)
  * 모델 지문이 다르다   -> 배포본 버전이 서로 다르다
  * 둘 다 같다           -> 어느 PC 에서든 같은 판정이 나와야 정상

출력은 cmd 창 인코딩 문제를 피하려고 전부 ASCII 로만 찍는다.
"""

from __future__ import annotations

import glob
import hashlib
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fasoo_drm import setting_fasoo

setting_fasoo()   # 암호화된 라벨 엑셀을 읽어야 하므로 가장 먼저

import numpy as np

import paths
from bdf_io import discover_pairs
from data_index import list_samples

#: 개발 PC 기준값. 라벨 엑셀이 바뀌면 이 두 줄을 같이 고쳐야 한다.
EXPECT_N = 3223
EXPECT_FP = "7a13192e"
EXPECT_XLSX = "fe model 1D index 0918.xlsx"


def fp(names) -> str:
    return hashlib.sha1(chr(10).join(names).encode()).hexdigest()[:8]


def cached_names(path: str) -> list[str] | None:
    """캐시 파일에서 이름 목록만 꺼낸다 (v1/v2 형식 모두 지원)."""
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    if blob.get("version") == 2:
        return list(blob["graphs"].keys())
    return list(blob.get("names", []))


def main() -> int:
    print("=" * 62)
    print(" FE MESH VIEWER - DIAGNOSTIC REPORT")
    print("=" * 62)

    import sklearn
    print("")
    print("[1] ENVIRONMENT")
    print(f"    python   : {sys.version.split()[0]}")
    print(f"    numpy    : {np.__version__}")
    print(f"    sklearn  : {sklearn.__version__}")
    print(f"    fasoo dll: {os.path.isfile(chr(67) + ':/Windows/System32/f_nxldr.dll')}")

    print("")
    print("[2] DATA FOLDER")
    try:
        paired, bdf_only, json_only = discover_pairs(paths.FE_DIR)
        print(f"    bdf+json pairs   : {len(paired)}   (expected 4000)")
        print(f"    bdf missing json : {len(bdf_only)}")
        print(f"    json missing bdf : {len(json_only)}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return 1

    print("")
    print("[3] LABEL TABLE  << most important")
    print(f"    file : {os.path.basename(paths.LABEL_XLSX)}   (expected {EXPECT_XLSX})")
    try:
        samples = list_samples(verbose=False)
        names = [s.name for s in samples]
        ok = (len(names) == EXPECT_N and fp(names) == EXPECT_FP)
        print(f"    trainable count  : {len(names)}   (expected {EXPECT_N})")
        print(f"    list fingerprint : {fp(names)}   (expected {EXPECT_FP})")
        print(f"    MATCH            : {'YES' if ok else 'NO  <<< PROBLEM HERE'}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return 1

    print("")
    print("[4] MODEL FILES  << most important")
    found_full = False
    for run in sorted(glob.glob(os.path.join(paths.RUNS_DIR, "*"))):
        full = os.path.join(run, "full.pt")
        if os.path.isfile(full):
            found_full = True
            h = hashlib.sha1(open(full, "rb").read()).hexdigest()[:8]
            mb = os.path.getsize(full) / 1e6
            print(f"    {os.path.basename(run)}/full.pt : {mb:.1f} MB  sha1 {h}")
        else:
            n_fold = len(glob.glob(os.path.join(run, "fold*.pt")))
            if n_fold:
                print(f"    {os.path.basename(run)}/ : {n_fold} fold models (OLD STYLE)")
    if not found_full:
        print("    NOTE: no full.pt - this is the old 5-fold build.")
        print("          Results can differ between PCs. Please update.")

    print("")
    print("[5] GRAPH CACHE")
    cache_files = sorted(glob.glob(os.path.join(paths.CACHE_DIR, "graphs_*.pkl")))
    if not cache_files:
        print("    (none yet - built on first run)")
    for c in cache_files:
        mb = os.path.getsize(c) / 1e6
        try:
            cn = cached_names(c)
            print(f"    {os.path.basename(c)} : {mb:.0f} MB, {len(cn)} graphs")
        except Exception as e:
            print(f"    {os.path.basename(c)} : {mb:.0f} MB, UNREADABLE ({e})")

    print("")
    print("=" * 62)
    print(" Please send this whole file back.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())

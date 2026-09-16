"""FASOO DRM(문서암호화)로 암호화된 파일을 읽을 수 있도록 하는 진입점 전용 유틸리티.

이 프로젝트는 독립 배포형 앱(run.bat + 번들 python)이라 D:\\python\\공통모듈 에 의존하지
않는다 — 그래서 공통모듈의 ``common.setting_fasoo()`` 를 import 하지 않고 이 파일 하나에
그대로 복사해 둔다.

사내 PC 에서는 라벨 엑셀(fe model 1D index 0721.xlsx)에 자동으로 암호가 걸리므로,
그 파일을 하나라도 열기 전에 진입점에서 :func:`setting_fasoo` 를 한 번 호출해야 한다.
"""

from __future__ import annotations

import ctypes
import os

FASOO_DLL = "C:\\Windows\\System32\\f_nxldr.dll"

#: 이미 해제했는지 — Streamlit 은 사용자가 위젯을 건드릴 때마다 스크립트 전체를 다시
#: 실행하므로, 가드가 없으면 클릭 한 번마다 EnableDRM() 이 다시 불린다.
_enabled = False

#: 한 번이라도 시도해 봤는지 — FASOO 미설치 PC 에서는 _enabled 가 계속 False 라서,
#: 이 플래그가 없으면 "건너뜀" 안내가 위젯 조작마다 터미널에 다시 찍힌다.
_checked = False


def setting_fasoo() -> bool:
    """FASOO 로 암호화된 파일을 읽을 수 있도록 한다. 이미 했으면 아무 것도 하지 않는다.

    FASOO 클라이언트가 설치되지 않은 PC(예: 사내망 밖 개발 환경)에는 이 dll 이 없을 수
    있으므로, 먼저 존재를 확인하고 없으면 조용히 건너뛴다 — 앱 전체가 죽을 이유는 아니다.

    * 주의: 이 함수를 실행한 이후에 **새로 만드는 파일에는 암호가 걸린다.** 이 앱이 만드는
      가장 큰 산출물은 ``cache/graphs_*.pkl`` (수백 MB) 이라, 그 캐시도 암호화된 채로
      저장된다. 같은 PC·같은 계정에서는 그대로 다시 읽히지만, 폴더를 다른 사람에게
      복사해 주면 캐시를 못 읽을 수 있다 — 그럴 때는 ``cache/`` 를 지우고 다시 만들면 된다.

    Returns:
        실제로 해제를 수행했으면 True, FASOO 미설치라 건너뛰었으면 False.
    """
    global _enabled, _checked
    if _enabled:
        return True

    if not os.path.isfile(FASOO_DLL):
        if not _checked:
            print(f"[fasoo_drm] {FASOO_DLL} 없음 - FASOO 미설치 환경으로 보고 건너뜀")
            _checked = True
        return False

    fasoo = ctypes.CDLL(FASOO_DLL)
    ret = fasoo.EnableDRM()
    if not ret:
        raise Exception("암호화 해제를 하지 못했습니다. ")

    _enabled = True
    print("[fasoo_drm] DRM 해제 완료 - 암호화된 파일을 읽을 수 있습니다")
    return True

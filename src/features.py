"""메쉬 기하 품질 지표 — 순수 numpy, MeshRep 에 의존하지 않는다.

전부 **평행이동·회전 불변량**으로만 구성한다. 절대 좌표는 어떤 피처에도 들어가지 않는다.
이 데이터셋의 패치들은 하나의 선박 메쉬를 타일링한 조각이라, 절대 좌표를 넣으면
모델이 "선박 어디쯤인가"를 암기할 수 있고 이는 누출 경로가 된다.
(기존 SHI/Ver1/bdf2info.py:98 은 raw x,y,z 를 그대로 노드 피처에 넣었다.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# 로컬 좌표계
# ──────────────────────────────────────────────────────────────────────────
def area_normal(P: np.ndarray) -> tuple[float, np.ndarray]:
    """부채꼴 삼각분할로 면적과 단위 법선을 구한다.

    Args:
        P: (k, 3) 코너 좌표, 요소 정의 순서.

    Returns:
        (면적, 단위 법선). 법선은 누적 외적 방향이라 요소의 절점 순서(감김 방향)를 따른다.
    """
    acc = np.zeros(3)
    area = 0.0
    for i in range(1, len(P) - 1):
        cr = np.cross(P[i] - P[0], P[i + 1] - P[0])
        acc += cr
        area += 0.5 * float(np.linalg.norm(cr))
    n = float(np.linalg.norm(acc))
    return area, acc / (n + EPS)


def local_frame(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """요소 평면에 붙은 정규직교 로컬 좌표계.

    Returns:
        (원점=중심, ex, ey, 법선). ``ex`` 는 첫 변 방향, ``ey = n × ex``.
    """
    _, n = area_normal(P)
    o = P.mean(axis=0) # 해당 shell의 중심점
    ex = P[1] - P[0] # 로컬 x축 방향 
    ex = ex / (np.linalg.norm(ex) + EPS) # ex 가 법선과 거의 평행한 퇴화 상황 방어
    
    if abs(float(np.dot(ex, n))) > 0.99:
        ex = P[2] - P[0]
        ex = ex / (np.linalg.norm(ex) + EPS)
    ex = ex - float(np.dot(ex, n)) * n # 성분 제거 -> 실제 shell이 완벽하게 평평하지 않을 수도 있기 때문
    ex = ex / (np.linalg.norm(ex) + EPS)
    ey = np.cross(n, ex)
    return o, ex, ey, n


def project_2d(P: np.ndarray) -> np.ndarray:
    """요소를 자기 평면의 로컬 2D 좌표로 투영한다. (k, 2)"""
    o, ex, ey, _ = local_frame(P)
    d = P - o
    return np.stack([d @ ex, d @ ey], axis=1)


def interior_angles(P: np.ndarray) -> np.ndarray:
    """각 코너의 내각(도). (k,)"""
    k = len(P)
    out = np.empty(k)
    for i in range(k):
        v1 = P[i - 1] - P[i]
        v2 = P[(i + 1) % k] - P[i]
        c = float(np.dot(v1, v2)) / (
            float(np.linalg.norm(v1)) * float(np.linalg.norm(v2)) + EPS
        )
        out[i] = math.degrees(math.acos(min(1.0, max(-1.0, c))))
    return out


def edge_lengths(P: np.ndarray) -> np.ndarray:
    """변 길이. (k,) — i번째는 P[i]→P[i+1]."""
    return np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)


# ──────────────────────────────────────────────────────────────────────────
# 자코비안
# ──────────────────────────────────────────────────────────────────────────
def normalized_jacobian(P: np.ndarray) -> tuple[float, float]:
    """정규화 자코비안 (min_det/scale, min_det/max_det).

    **반드시 요소의 로컬 2D 평면에서 계산한다.** 기존 SHI/Ver1/bdf2info_v2.py:107-110 의
    CQUAD4 분기는 전역 X, Y 좌표를 그대로 썼는데, 이 데이터셋 쉘의 상당수는 YZ/XZ 평면에
    놓여 있어 전역 XY 로 만든 자코비안이 퇴화(≈0)한다. 로컬 투영으로 그 문제를 없앤다.

    Returns:
        (norm_J, det_ratio) — 둘 다 [-1, 1] / [0, 1] 범위. 완벽한 요소는 (0.5, 1.0) 근처.
    """
    Q = project_2d(P)
    k = len(Q)

    if k == 3:
        v1, v2 = Q[1] - Q[0], Q[2] - Q[0]
        J = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])
        det = float(np.linalg.det(J))
        scale = float(np.linalg.norm(v1)) * float(np.linalg.norm(v2)) + EPS
        return float(np.clip(det / scale, -1.0, 1.0)), 1.0

    dets = []
    for xi, eta in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        J = np.array(
            [
                [float(dN_dxi @ Q[:, 0]), float(dN_deta @ Q[:, 0])],
                [float(dN_dxi @ Q[:, 1]), float(dN_deta @ Q[:, 1])],
            ]
        )
        dets.append(float(np.linalg.det(J)))

    lo, hi = min(dets), max(dets)
    L = edge_lengths(P)
    norm_J = lo / (L[0] * L[3] + EPS)
    # det_ratio 는 [-1, 1] 로 클립한다. 음수는 코너 자코비안의 부호가 갈렸다는 뜻,
    # 즉 요소가 접혀(fold-over) 있다는 강한 불량 신호다. [0, 1] 로 클립하면 이 신호가 사라진다.
    det_ratio = lo / (abs(hi) + EPS)
    return float(np.clip(norm_J, -1.0, 1.0)), float(np.clip(det_ratio, -1.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────
# 쉘 요소 품질
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ShellQuality:
    """쉘 요소 하나의 형상 품질. 전부 좌표 불변량."""

    is_quad: bool
    area: float
    normal: np.ndarray = field(repr=False)
    #: 최장변 / 최단변
    aspect: float
    #: 대각선 비율 (사각형만, 삼각형은 1.0)
    diag_ratio: float
    angle_min: float
    angle_max: float
    #: 이상각(사각 90°, 삼각 60°) 으로부터의 최대 이탈
    skew: float
    #: 사각형 뒤틀림 — 두 분할 삼각형 법선 사이 각(도). 삼각형은 0.
    warp: float
    #: 코너 삼각형 면적의 min/max — 1에 가까울수록 균일 (사각형만 유효)
    taper: float
    norm_jacobian: float
    det_ratio: float
    len_min: float
    len_max: float
    len_mean: float
    len_std: float
    perimeter: float


def shell_quality(P: np.ndarray) -> ShellQuality:
    """(k, 3) 코너 좌표에서 쉘 품질 지표를 계산한다. k 는 3 또는 4."""
    k = len(P)
    if k not in (3, 4):
        raise ValueError(f"쉘 코너 수는 3 또는 4 여야 함: {k}")
    is_quad = k == 4

    area, normal = area_normal(P)
    ang = interior_angles(P)
    L = edge_lengths(P)
    ideal = 90.0 if is_quad else 60.0

    if is_quad:
        d1 = float(np.linalg.norm(P[2] - P[0]))
        d2 = float(np.linalg.norm(P[3] - P[1]))
        diag_ratio = max(d1, d2) / (min(d1, d2) + EPS)

        n1 = np.cross(P[1] - P[0], P[2] - P[0])
        n2 = np.cross(P[2] - P[0], P[3] - P[0])
        c = float(np.dot(n1, n2)) / (
            float(np.linalg.norm(n1)) * float(np.linalg.norm(n2)) + EPS
        )
        warp = math.degrees(math.acos(min(1.0, max(-1.0, c))))

        corner = [
            0.5 * float(np.linalg.norm(np.cross(P[i - 1] - P[i], P[(i + 1) % 4] - P[i])))
            for i in range(4)
        ]
        taper = min(corner) / (max(corner) + EPS)
    else:
        diag_ratio, warp, taper = 1.0, 0.0, 1.0

    norm_J, det_ratio = normalized_jacobian(P)

    return ShellQuality(
        is_quad=is_quad,
        area=area,
        normal=normal,
        aspect=float(L.max() / (L.min() + EPS)),
        diag_ratio=float(diag_ratio),
        angle_min=float(ang.min()),
        angle_max=float(ang.max()),
        skew=float(max(ang.max() - ideal, ideal - ang.min())),
        warp=float(warp),
        taper=float(taper),
        norm_jacobian=norm_J,
        det_ratio=det_ratio,
        len_min=float(L.min()),
        len_max=float(L.max()),
        len_mean=float(L.mean()),
        len_std=float(L.std()),
        perimeter=float(L.sum()),
    )


def is_bad_shell(q: ShellQuality) -> bool:
    """관용적인 '불량 요소' 판정 — 최소내각 30° 미만 또는 최대내각 150° 초과.

    라벨과의 상관이 가장 강한 단순 규칙 중 하나이고, explain.py 에서 어텐션 검증의
    기준선으로도 쓴다.
    """
    return q.angle_min < 30.0 or q.angle_max > 150.0


# ──────────────────────────────────────────────────────────────────────────
# 1D 요소
# ──────────────────────────────────────────────────────────────────────────
def line_length_dir(p1: np.ndarray, p2: np.ndarray) -> tuple[float, np.ndarray]:
    """1D 요소의 길이와 부호없는 방향코사인 ``|dx|, |dy|, |dz|``.

    부호를 없애는 이유: CBAR 의 절점 순서는 임의라 방향 부호에 의미가 없다.
    """
    d = p2 - p1
    ln = float(np.linalg.norm(d))
    return ln, np.abs(d / (ln + EPS))


def unsigned_normal(n: np.ndarray) -> np.ndarray:
    """법선을 부호 정규화한다 — 절대값 최대 성분이 양수가 되도록.

    쉘의 절점 감김 방향은 임의이므로 법선 부호에 의미가 없다. 성분 절대값을 쓰지 않고
    부호만 정규화하면 서로 직교하는 두 패널을 구분하는 정보가 남는다.
    """
    i = int(np.argmax(np.abs(n)))
    return n if n[i] >= 0 else -n


def dihedral_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """두 쉘 법선 사이의 각(도). 감김 방향에 무관하도록 |cos| 을 쓴다.

    0° = 같은 평면, 90° = 직교(패널 교차).
    """
    c = abs(float(np.dot(n1, n2)))
    return math.degrees(math.acos(min(1.0, max(0.0, c))))

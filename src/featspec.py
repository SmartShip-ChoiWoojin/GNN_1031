"""피처 스펙과 전역 정규화.

**기존 구현에서 가장 크게 바꾸는 부분이다.**
`SHI/Ver1/bdf2info.py:106-109`, `bdf2info_v2.py:180-183`, `bdf2info_v3.py:180-183` 은
피처를 *각 그래프 내부에서* z-score 정규화했다. 그러면

    "모든 요소가 나쁜 메쉬"  와  "모든 요소가 좋은 메쉬"

가 완전히 동일한 입력이 된다. 라벨을 결정하는 것은 바로 그 절대적 품질 수준이므로,
그래프별 정규화는 판별 정보를 조직적으로 제거한다.

여기서는 그래프 빌더가 **원시값**을 내보내고, :class:`Normalizer` 가
**데이터셋 전역 통계**로 변환한다. 통계는 **학습 fold 에서만** 적합시켜 val/test 에 적용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 피처 종류
#:   "binary" — 0/1 지시자·one-hot. 정규화하지 않는다 (이미 잘 스케일되어 있고
#:              의미가 명확해서 변환하면 해석만 어려워진다).
#:   "linear" — 각도, 비율, 개수 등. z-score.
#:   "log"    — 면적·길이처럼 꼬리가 두꺼운 양수량. log1p 후 z-score.
FeatureKind = str


@dataclass(frozen=True)
class FeatureSpec:
    """이름과 종류가 짝지어진 피처 벡터의 스키마."""

    names: tuple[str, ...]
    kinds: tuple[FeatureKind, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.kinds):
            raise ValueError(
                f"names({len(self.names)}) 와 kinds({len(self.kinds)}) 길이가 다름"
            )
        bad = set(self.kinds) - {"binary", "linear", "log"}
        if bad:
            raise ValueError(f"알 수 없는 kind: {bad}")

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def log_mask(self) -> np.ndarray:
        return np.array([k == "log" for k in self.kinds])

    @property
    def scale_mask(self) -> np.ndarray:
        """z-score 를 적용할 열 (binary 가 아닌 전부)."""
        return np.array([k != "binary" for k in self.kinds])

    @staticmethod
    def of(*pairs: tuple[str, FeatureKind]) -> "FeatureSpec":
        """``FeatureSpec.of(("area", "log"), ("skew", "linear"), ...)``"""
        return FeatureSpec(tuple(p[0] for p in pairs), tuple(p[1] for p in pairs))


#: 스케일 추정 시 꼬리를 잘라내는 분위 (이상치 1개가 스케일을 부풀리지 않도록)
WINSOR_LO, WINSOR_HI = 0.5, 99.5

#: 정규화 후 클립 범위 — 진짜 극단값만 걸러내는 안전장치
CLIP = 10.0


class Normalizer:
    """전역 통계 기반 피처 변환기. log1p → 중심이동 → 스케일.

    스케일은 **윈저화(0.5~99.5 분위 클립) 후의 표준편차**를 쓴다.

    MAD(중위절대편차)를 쓰면 안 된다. 이 데이터의 여러 피처는 강하게 이봉분포다 —
    이면각은 0°(같은 평면) 또는 90°(패널 교차), 법선 성분은 0 또는 ±1, taper/det_ratio 는
    1 근처에 큰 질량이 몰려 있다. 이런 분포에서 MAD 는 매우 작게 나와서 반대쪽 봉우리
    전체가 클립 경계에 붙어버린다(실측: 이면각 42%, s_ny 47%가 클립됨).
    윈저화 표준편차는 이봉 분포의 퍼짐을 제대로 반영하면서 단일 이상치에는 견딘다.
    """

    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    # ── 적합 / 적용 ────────────────────────────────────────────────────
    def _pre(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).copy()
        if X.ndim != 2 or X.shape[1] != len(self.spec):
            raise ValueError(f"기대 차원 (n, {len(self.spec)}), 실제 {X.shape}")
        lm = self.spec.log_mask
        if lm.any():
            # log 열은 음수가 들어올 수 없는 양수량이지만, 수치오차 방어로 0 하한을 둔다
            X[:, lm] = np.log1p(np.maximum(X[:, lm], 0.0))
        return X

    def fit(self, X: np.ndarray, valid: np.ndarray | None = None) -> "Normalizer":
        """학습 fold 의 전 노드(또는 전 엣지)를 쌓은 (n, d) 배열로 적합.

        Args:
            valid: (n, d) bool. False 인 칸은 "해당 없음"(예: 1D 요소의 면적)이라
                통계에서 제외한다. 이걸 하지 않으면 타입별로 마스킹된 피처의 0 이
                통계를 끌어당겨, 반대 타입 노드가 전부 클립 경계값을 받게 된다.
        """
        Z = self._pre(X)
        n, d = Z.shape
        if valid is None:
            valid = np.ones((n, d), dtype=bool)

        med = np.zeros(d)
        scale = np.ones(d)
        for j in range(d):
            col = Z[valid[:, j], j]
            if col.size == 0:
                continue  # 학습 fold 에 유효값이 없는 열 → 항등 변환
            m = float(np.median(col))
            lo, hi = np.percentile(col, [WINSOR_LO, WINSOR_HI])
            s = float(np.clip(col, lo, hi).std())
            if s < 1e-8:
                s = float(col.std())          # 윈저화가 분포를 다 잘라낸 경우
            if s < 1e-8:
                s = max(float(np.abs(col - m).max()), 1.0)  # 사실상 상수인 열
            med[j], scale[j] = m, s

        sm = self.spec.scale_mask
        med[~sm] = 0.0
        scale[~sm] = 1.0
        self.center, self.scale = med, scale
        return self

    def transform(self, X: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """정규화 적용. ``valid`` 가 False 인 칸은 정확히 0.0 으로 둔다.

        "해당 없음"을 0 으로 두면 타입 one-hot 이 어느 칸을 읽어야 하는지 알려주므로
        네트워크가 혼동하지 않는다.
        """
        if self.center is None or self.scale is None:
            raise RuntimeError("fit() 을 먼저 호출해야 함")
        Z = self._pre(X)
        Z = (Z - self.center) / self.scale
        Z = np.clip(Z, -CLIP, CLIP)
        if valid is not None:
            Z = np.where(valid, Z, 0.0)
        return Z.astype(np.float32)

    # ── 직렬화 ────────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {
            "names": list(self.spec.names),
            "kinds": list(self.spec.kinds),
            "center": None if self.center is None else self.center.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "Normalizer":
        spec = FeatureSpec(tuple(state["names"]), tuple(state["kinds"]))
        n = cls(spec)
        if state["center"] is not None:
            n.center = np.array(state["center"], dtype=np.float64)
            n.scale = np.array(state["scale"], dtype=np.float64)
        return n

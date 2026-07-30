"""데이터셋 목록화 — 어떤 모델을 어떤 라벨로 쓸지 한 곳에서 정한다.

torch 를 import 하지 않는다. baseline.py 와 dataset.py 가 공유한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import paths
from bdf_io import discover_pairs
from labels import load_label_table


@dataclass(frozen=True)
class Sample:
    name: str
    bdf: str
    json: str
    cls: int
    ambig: int


def list_samples(
    fe_dir: str | None = None,
    label_xlsx: str | None = None,
    trainable_only: bool = True,
    verbose: bool = True,
) -> list[Sample]:
    """.bdf/.json 짝이 있고 라벨이 있는 모델 목록.

    Args:
        trainable_only: True 면 확실(ambig=0) 이고 분류 가능(cls<2) 한 것만.
            False 면 애매/분류어려움까지 모두 (별도 평가·검증용).

    Returns:
        모델 이름 정렬 순서의 :class:`Sample` 목록.
    """
    fe_dir = fe_dir or paths.FE_DIR
    label_xlsx = label_xlsx or paths.LABEL_XLSX

    paired, bdf_only, json_only = discover_pairs(fe_dir)
    table = load_label_table(label_xlsx)
    if trainable_only:
        table = table[table["trainable"]]

    lab = {r.name: (int(r.cls), int(r.ambig)) for r in table.itertuples()}

    out: list[Sample] = []
    n_no_label = 0
    for name in sorted(paired):
        if name not in lab:
            n_no_label += 1
            continue
        cls, ambig = lab[name]
        bdf, js = paired[name]
        out.append(Sample(name=name, bdf=bdf, json=js, cls=cls, ambig=ambig))

    if verbose:
        print(f"[data_index] {fe_dir}")
        print(f"  .bdf/.json 짝     : {len(paired)}")
        if bdf_only:
            print(f"  json 없어 제외    : {bdf_only}")
        if json_only:
            print(f"  bdf 없는 고아 json: {json_only}")
        print(f"  라벨 조건 미충족   : {n_no_label}")
        print(f"  사용              : {len(out)}"
              f"  (정상 {sum(1 for s in out if s.cls == 0)}"
              f" / 비정상 {sum(1 for s in out if s.cls == 1)}"
              f" / 분류어려움 {sum(1 for s in out if s.cls == 2)})")
    return out


def labels_array(samples: list[Sample]) -> np.ndarray:
    return np.array([s.cls for s in samples], dtype=np.int64)


def spatial_groups(samples: list[Sample], block: float = 6000.0) -> np.ndarray:
    """패치 중심좌표를 ``block`` mm 격자로 뭉갠 그룹 id.

    1603개 패치는 하나의 선박 메쉬에서 잘라낸 조각들이라 절점 id 가 서로 겹친다
    (한 id 가 평균 3개, 최대 13개 모델에 등장). 랜덤 분할은 이 때문에 낙관적으로 나온다.
    ``--split group`` 을 쓸 때 이 그룹을 StratifiedGroupKFold 에 넘긴다.

    참고: '절점 id 를 공유하면 같은 그룹' 방식은 쓸 수 없다 — 추이적으로 이어져
    1602개 중 1598개가 한 덩어리가 된다(측정 확인). 그래서 공간 격자를 쓴다.
    """
    from bdf_io import read_bdf  # 지연 import — 목록화만 할 때는 필요 없다

    keys = []
    for s in samples:
        coords, _, _ = read_bdf(s.bdf)
        c = np.array(list(coords.values()), dtype=float).mean(axis=0)
        keys.append("_".join(str(int(v // block)) for v in c))
    return np.array(keys)

"""라벨 엑셀 로더.

`fe model 1D index 0721.xlsx` 는 위쪽 12행이 통계 요약이고 실제 표 헤더는 13행째에 있다.
컬럼명은 `모델 이름 / 분류 결과(문자) / 분류 결과(숫자) / 확실(0)/애매(1)`.

기존 `SHI/Ver1/dataset.py:17-36` 은 `파일 이름`, `분류결과` 를 찾으므로 이 파일에는 쓸 수 없다.
헤더 위치를 하드코딩하지 않고 스캔해서 찾는다.

라벨 의미
    분류 결과(숫자): 0=정상, 1=비정상, 2=분류 어려움
    확실(0)/애매(1): 사람이 판정에 확신했는지

학습 대상은 **확실(0) 이고 분류 어려움(2) 이 아닌** 것들뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CLS_NORMAL = 0
CLS_ABNORMAL = 1
CLS_UNDECIDABLE = 2

AMBIG_CERTAIN = 0
AMBIG_VAGUE = 1


@dataclass(frozen=True)
class LabelRow:
    name: str
    cls: int
    ambig: int

    @property
    def is_trainable(self) -> bool:
        """확실하고 분류 가능한 라벨인가 — 학습/평가에 쓸 수 있는지."""
        return self.cls in (CLS_NORMAL, CLS_ABNORMAL) and self.ambig == AMBIG_CERTAIN


def _find_header_row(path: str, max_scan: int = 40) -> int:
    """'이름' 컬럼이 있는 헤더 행 번호(0-based)를 찾는다."""
    probe = pd.read_excel(path, header=None, nrows=max_scan)
    for i, row in probe.iterrows():
        cells = [str(c) for c in row.tolist()]
        if any("이름" in c for c in cells) and any("분류" in c for c in cells):
            return int(i)
    raise ValueError(f"{path}: '이름'/'분류' 를 포함한 헤더 행을 찾을 수 없음")


def _pick(cols: list[str], *musts: str) -> str | None:
    for c in cols:
        if all(m in c for m in musts):
            return c
    return None


def load_label_table(path: str) -> pd.DataFrame:
    """라벨 엑셀을 ``name / cls / ambig / trainable`` 컬럼의 DataFrame 으로 읽는다."""
    hdr = _find_header_row(path)
    df = pd.read_excel(path, header=hdr)
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    name_col = _pick(cols, "이름")
    cls_col = _pick(cols, "분류", "숫자") or _pick(cols, "분류")
    ambig_col = _pick(cols, "애매")

    if name_col is None or cls_col is None:
        raise ValueError(f"{path}: 필요한 컬럼을 못 찾음 (컬럼: {cols})")

    out = pd.DataFrame(
        {
            "name": df[name_col].astype(str).str.strip(),
            "cls": pd.to_numeric(df[cls_col], errors="coerce"),
        }
    )
    if ambig_col is None:
        # 확실/애매 컬럼이 없는 구버전 엑셀 — 전부 확실로 취급
        out["ambig"] = AMBIG_CERTAIN
    else:
        out["ambig"] = pd.to_numeric(df[ambig_col], errors="coerce")

    out = out.dropna(subset=["name", "cls"])
    out = out[out["name"].str.len() > 0]
    out = out[~out["name"].isin(["nan", "None"])]
    out["cls"] = out["cls"].astype(int)
    out["ambig"] = out["ambig"].fillna(AMBIG_VAGUE).astype(int)
    out["trainable"] = (out["cls"] <= CLS_ABNORMAL) & (out["ambig"] == AMBIG_CERTAIN)
    return out.reset_index(drop=True)


def load_labels(path: str, trainable_only: bool = True) -> dict[str, LabelRow]:
    """``{모델 이름: LabelRow}``. ``trainable_only`` 면 확실·분류가능 항목만."""
    df = load_label_table(path)
    if trainable_only:
        df = df[df["trainable"]]
    return {
        r.name: LabelRow(name=r.name, cls=int(r.cls), ambig=int(r.ambig))
        for r in df.itertuples()
    }


def label_summary(path: str) -> str:
    """라벨 분포 요약 문자열 — 로딩이 제대로 됐는지 눈으로 확인할 때."""
    df = load_label_table(path)
    lines = [f"총 {len(df)}행"]
    for c, n in sorted(df["cls"].value_counts().items()):
        tag = {0: "정상", 1: "비정상", 2: "분류어려움"}.get(int(c), f"cls={c}")
        lines.append(f"  cls {c} ({tag}): {n}")
    for a, n in sorted(df["ambig"].value_counts().items()):
        tag = {0: "확실", 1: "애매"}.get(int(a), f"ambig={a}")
        lines.append(f"  ambig {a} ({tag}): {n}")
    lines.append(f"  학습 가능(확실 & cls<2): {int(df['trainable'].sum())}")
    return "\n".join(lines)

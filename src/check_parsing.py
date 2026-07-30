"""파싱 계층 회귀 테스트 — 1602개 전 모델을 읽고 가정이 성립하는지 검증한다.

    python src/check_parsing.py
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

import paths
from bdf_io import discover_pairs, load_raw_mesh
from labels import label_summary, load_label_table
from meshrep import MeshRep

# 파싱 회귀 테스트 기준값 — SN2592_FWD_Clean_0001 을 손으로 센 값
EXPECT_0001 = dict(n_node=55, n_quad=42, n_tri=1, n_bar=29, n_beam=9)


def main() -> int:
    fails: list[str] = []

    print("=" * 70)
    print("라벨 엑셀")
    print("=" * 70)
    print(label_summary(paths.LABEL_XLSX))
    table = load_label_table(paths.LABEL_XLSX)

    print()
    print("=" * 70)
    print("파일 짝 맞추기")
    print("=" * 70)
    paired, bdf_only, json_only = discover_pairs(paths.FE_DIR)
    print(f"짝 맞는 모델      : {len(paired)}")
    print(f"json 없는 bdf     : {bdf_only}")
    print(f"bdf 없는 json     : {json_only}")
    # 계획서에 기록된 알려진 불일치 두 건
    if bdf_only != ["SN2592_FWD_Clean_1603"]:
        fails.append(f"예상과 다른 bdf-only 목록: {bdf_only}")
    if json_only != ["SN2592_FWD_Clean_4000"]:
        fails.append(f"예상과 다른 json-only 목록: {json_only}")

    print()
    print("=" * 70)
    print("전 모델 파싱")
    print("=" * 70)
    labeled = set(table["name"])
    reps: list[MeshRep] = []
    n_off_edge_total = 0
    n_dup_bar_total = 0
    type_counter: Counter[str] = Counter()

    for i, (name, (bdf, js)) in enumerate(sorted(paired.items()), start=1):
        try:
            rep = MeshRep(load_raw_mesh(bdf, js))
        except Exception as exc:  # noqa: BLE001 — 어느 파일이 왜 깨졌는지 전부 모으고 싶다
            fails.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

        reps.append(rep)
        if name not in labeled:
            fails.append(f"{name}: 라벨 엑셀에 없음")

        for s in rep.shells:
            type_counter[s.etype] += 1
        for l in rep.lines:
            type_counter[l.etype] += 1

        # 계획서에 기록된 사실: 1D 요소는 100% 쉘 변과 일치, 중복 1D 없음
        seen = set()
        for l in rep.lines:
            if l.edge not in rep.edge_shells:
                n_off_edge_total += 1
            if l.edge in seen:
                n_dup_bar_total += 1
            seen.add(l.edge)

        if i % 400 == 0:
            print(f"  ... {i}/{len(paired)}")

    print(f"파싱 성공          : {len(reps)}/{len(paired)}")
    print(f"카드 집계          : {dict(type_counter)}")
    print(f"쉘 변에 없는 1D    : {n_off_edge_total}  (0이어야 함)")
    print(f"중복 1D            : {n_dup_bar_total}  (0이어야 함)")
    if n_off_edge_total:
        fails.append(f"쉘 변 위에 없는 1D 요소가 {n_off_edge_total}개 발견됨")
    if n_dup_bar_total:
        fails.append(f"중복 1D 요소가 {n_dup_bar_total}개 발견됨")

    print()
    print("=" * 70)
    print("회귀 테스트: SN2592_FWD_Clean_0001")
    print("=" * 70)
    r0 = next(r for r in reps if r.name == "SN2592_FWD_Clean_0001")
    got = dict(
        n_node=len(r0.node_ids),
        n_quad=sum(1 for s in r0.shells if s.etype == "CQUAD4"),
        n_tri=sum(1 for s in r0.shells if s.etype == "CTRIA3"),
        n_bar=sum(1 for l in r0.lines if l.etype == "CBAR"),
        n_beam=sum(1 for l in r0.lines if l.etype == "CBEAM"),
    )
    for k, want in EXPECT_0001.items():
        mark = "OK " if got[k] == want else "!! "
        print(f"  {mark}{k:8s} 기대 {want:4d}  실제 {got[k]:4d}")
        if got[k] != want:
            fails.append(f"0001 {k}: 기대 {want}, 실제 {got[k]}")

    print()
    print("=" * 70)
    print("학습 대상 집계")
    print("=" * 70)
    trainable = set(table[table["trainable"]]["name"])
    usable = [r for r in reps if r.name in trainable]
    cls_map = dict(zip(table["name"], table["cls"]))
    ys = [int(cls_map[r.name]) for r in usable]
    print(f"파싱 가능 & 학습 가능 : {len(usable)}")
    print(f"  정상(0)   : {ys.count(0)}")
    print(f"  비정상(1) : {ys.count(1)}")
    if len(usable) != 1262:
        fails.append(f"학습 가능 모델 수가 1262가 아님: {len(usable)}")

    print()
    print("=" * 70)
    print("규모 통계 (전 모델)")
    print("=" * 70)
    for label, vals in [
        ("절점", [len(r.node_ids) for r in reps]),
        ("쉘", [len(r.shells) for r in reps]),
        ("1D", [len(r.lines) for r in reps]),
        ("변", [len(r.edge_shells) for r in reps]),
    ]:
        a = np.array(vals)
        print(f"  {label:4s} 평균 {a.mean():6.1f}  최소 {a.min():4d}  최대 {a.max():4d}")

    print()
    print("=" * 70)
    if fails:
        print(f"실패 {len(fails)}건")
        for f in fails[:30]:
            print("  -", f)
        if len(fails) > 30:
            print(f"  ... 외 {len(fails) - 30}건")
        return 1
    print("전 항목 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())

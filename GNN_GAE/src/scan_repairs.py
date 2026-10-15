"""규칙 기반 수리기가 실제로 손댈 수 있는 메쉬를 전부 찾아 목록으로 남긴다.

``repair_rule.py --n`` 은 합계만 보여준다. 어떤 메쉬가 몇 번 고쳐졌는지 알아야
뷰어로 볼 대상을 고르고, 나중에 학습 모델과 메쉬 단위로 비교할 수 있다.

    python src/scan_repairs.py                 # 비정상 전체
    python src/scan_repairs.py --cls 0         # 정상도 같이 보고 싶을 때
    python src/scan_repairs.py --viz 5         # 상위 5개는 뷰어 HTML 까지 생성

결과는 ``out/repairs_cls{N}.csv`` 로 저장된다.
"""

from __future__ import annotations

import argparse
import csv
import os

import gae_paths as gp
import features as F
from bdf_io import discover_pairs
from labels import load_labels
from meshedit import EditableMesh
from repair_rule import RepairConfig, repair


def scan(names: list[str], pairs: dict, cfg: RepairConfig) -> list[dict]:
    rows = []
    for i, nm in enumerate(names, 1):
        if i % 100 == 0:
            print(f"  {i}/{len(names)} ...", flush=True)
        m = EditableMesh.load(*pairs[nm])
        fixed, ops = repair(m, cfg)
        t0, t1 = len(m.tri_indices()), len(fixed.tri_indices())
        b0 = sum(1 for k in range(len(m.shells)) if F.is_bad_shell(m.quality(k)))
        b1 = sum(1 for k in range(len(fixed.shells)) if F.is_bad_shell(fixed.quality(k)))
        rows.append({
            "name": nm,
            "n_op": len(ops),
            "n_shell": len(m.shells),
            "tri_before": t0,
            "tri_after": t1,
            "tri_frac_before": round(t0 / max(1, len(m.shells)), 4),
            "tri_frac_after": round(t1 / max(1, len(fixed.shells)), 4),
            "bad_before": b0,
            "bad_after": b1,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="수리 가능한 메쉬 전수 조사")
    ap.add_argument("--cls", type=int, default=1, help="1=비정상(기본), 0=정상")
    ap.add_argument("--limit", type=int, help="앞에서 N개만 (기본: 전체)")
    ap.add_argument("--viz", type=int, default=0, metavar="N",
                    help="연산 수 상위 N개는 뷰어 HTML 까지 만든다")
    args = ap.parse_args()

    gp.ensure_dirs()
    pairs, _, _ = discover_pairs(gp.FE_DIR)
    lab = load_labels(gp.LABEL_XLSX)
    names = [n for n in sorted(lab) if n in pairs and lab[n].cls == args.cls]
    if args.limit:
        names = names[: args.limit]

    tag = "비정상" if args.cls == 1 else "정상"
    print(f"{tag} 메쉬 {len(names)}개에 규칙 기반 수리기를 돌린다")
    rows = scan(names, pairs, RepairConfig())

    hit = [r for r in rows if r["n_op"] > 0]
    hit.sort(key=lambda r: (-r["n_op"], -(r["tri_before"] - r["tri_after"])))

    out_csv = os.path.join(gp.OUT_DIR, f"repairs_cls{args.cls}.csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (-r["n_op"], r["name"])))

    tri_b = sum(r["tri_before"] for r in rows)
    tri_a = sum(r["tri_after"] for r in rows)
    bad_b = sum(r["bad_before"] for r in rows)
    bad_a = sum(r["bad_after"] for r in rows)
    print(f"\n손댈 수 있던 메쉬: {len(hit)} / {len(rows)} ({len(hit)/len(rows):.1%})")
    print(f"총 연산 횟수:      {sum(r['n_op'] for r in rows)}")
    print(f"삼각형:            {tri_b} → {tri_a} ({(tri_b-tri_a)/max(1,tri_b):.1%} 제거)")
    print(f"불량각 요소:       {bad_b} → {bad_a} ({(bad_b-bad_a)/max(1,bad_b):.1%} 제거)")
    print(f"\n연산 횟수 분포:")
    from collections import Counter
    for k, v in sorted(Counter(r["n_op"] for r in rows).items()):
        print(f"  {k}회: {v}개")
    print(f"\n상위 15개")
    for r in hit[:15]:
        print(f"  {r['name']}  연산 {r['n_op']}회  삼각형 {r['tri_before']}→{r['tri_after']}"
              f"  불량각 {r['bad_before']}→{r['bad_after']}")
    print(f"\n전체 목록: {out_csv}")

    if args.viz:
        from viz_compare import build_html, stats
        print()
        for r in hit[: args.viz]:
            m = EditableMesh.load(*pairs[r["name"]])
            fixed, ops = repair(m, RepairConfig())
            sb, sa = stats(m), stats(fixed)
            sub = (f"연산 {len(ops)}회 · 삼각형 {sb['삼각형']} → {sa['삼각형']} · "
                   f"불량각 요소 {sb['불량각 요소']} → {sa['불량각 요소']}")
            p = os.path.join(gp.OUT_DIR, f"compare_{r['name']}.html")
            build_html(m, fixed, ops, p, sub)
            print(f"  {p}")


if __name__ == "__main__":
    main()

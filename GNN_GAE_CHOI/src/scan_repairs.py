"""규칙 기반 수리기가 실제로 손댈 수 있는 메쉬를 전부 찾아 목록으로 남긴다 (참고용).

``repair_rule.py --n`` 은 합계만 보여준다. 어떤 메쉬가 몇 번 고쳐졌는지 알아야
뷰어로 볼 대상을 고르고, 나중에 학습 모델과 메쉬 단위로 비교할 수 있다.

    python src/scan_repairs.py                 # 비정상 전체
    python src/scan_repairs.py --cls 0         # 정상
    python src/scan_repairs.py --viz 5         # 상위 5개는 뷰어 HTML 까지 생성

결과는 ``out/repairs_cls{N}.csv`` 로 저장된다.
"""

from __future__ import annotations

import argparse
import csv
import os

import gae_paths
from bdf_io import discover_pairs
from labels import load_labels
from meshedit import EditableMesh
from repair_rule import RepairConfig, count_bad_shells, repair


def scan(names: list, mesh_paths: dict, config: RepairConfig) -> list:
    """메쉬마다 수리기를 돌려 한 줄씩."""
    rows = []
    count = 0
    for name in names:
        count = count + 1
        if count % 100 == 0:
            print(f"  {count}/{len(names)} ...", flush=True)
        mesh = EditableMesh.load(*mesh_paths[name])
        fixed_mesh, operations = repair(mesh, config)
        triangles_before = len(mesh.tri_indices())
        triangles_after = len(fixed_mesh.tri_indices())
        rows.append({
            "name": name,
            "n_op": len(operations),
            "n_shell": len(mesh.shells),
            "tri_before": triangles_before,
            "tri_after": triangles_after,
            "tri_frac_before": round(triangles_before / max(1, len(mesh.shells)), 4),
            "tri_frac_after": round(triangles_after / max(1, len(fixed_mesh.shells)), 4),
            "bad_before": count_bad_shells(mesh),
            "bad_after": count_bad_shells(fixed_mesh),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="수리 가능한 메쉬 전수 조사")
    parser.add_argument("--cls", type=int, default=1, help="1=비정상(기본), 0=정상")
    parser.add_argument("--limit", type=int, help="앞에서 N개만 (기본: 전체)")
    parser.add_argument("--viz", type=int, default=0, metavar="N", help="연산 수 상위 N개는 뷰어 HTML 까지 만든다")
    args = parser.parse_args()

    gae_paths.ensure_dirs()
    mesh_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    labels = load_labels(gae_paths.LABEL_XLSX)
    names = []
    for name in sorted(labels):
        if name in mesh_paths and labels[name].cls == args.cls:
            names.append(name)
    if args.limit:
        names = names[:args.limit]

    if args.cls == 1:
        tag = "비정상"
    else:
        tag = "정상"
    print(f"{tag} 메쉬 {len(names)}개에 규칙 기반 수리기를 돌린다")
    rows = scan(names, mesh_paths, RepairConfig())

    def hit_order(row):
        return (-row["n_op"], -(row["tri_before"] - row["tri_after"]))

    def file_order(row):
        return (-row["n_op"], row["name"])

    repaired_rows = []
    for row in rows:
        if row["n_op"] > 0:
            repaired_rows.append(row)
    repaired_rows.sort(key=hit_order)

    out_csv = os.path.join(gae_paths.OUT_DIR, f"repairs_cls{args.cls}.csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=file_order))

    triangles_before = 0
    triangles_after = 0
    bad_before = 0
    bad_after = 0
    operation_total = 0
    operation_count_table = {}
    for row in rows:
        triangles_before = triangles_before + row["tri_before"]
        triangles_after = triangles_after + row["tri_after"]
        bad_before = bad_before + row["bad_before"]
        bad_after = bad_after + row["bad_after"]
        operation_total = operation_total + row["n_op"]
        if row["n_op"] not in operation_count_table:
            operation_count_table[row["n_op"]] = 0
        operation_count_table[row["n_op"]] = operation_count_table[row["n_op"]] + 1

    print(f"\n손댈 수 있던 메쉬: {len(repaired_rows)} / {len(rows)} ({len(repaired_rows) / len(rows):.1%})")
    print(f"총 연산 횟수:      {operation_total}")
    print(f"삼각형:            {triangles_before} → {triangles_after} "
          f"({(triangles_before - triangles_after) / max(1, triangles_before):.1%} 제거)")
    print(f"불량각 요소:       {bad_before} → {bad_after} ({(bad_before - bad_after) / max(1, bad_before):.1%} 제거)")
    print("\n연산 횟수 분포:")
    for operation_count, mesh_count in sorted(operation_count_table.items()):
        print(f"  {operation_count}회: {mesh_count}개")
    print("\n상위 15개")
    for row in repaired_rows[:15]:
        print(f"  {row['name']}  연산 {row['n_op']}회  삼각형 {row['tri_before']}→{row['tri_after']}"
              f"  불량각 {row['bad_before']}→{row['bad_after']}")
    print(f"\n전체 목록: {out_csv}")

    if args.viz:
        from viz_compare import build_html, stats

        print()
        for row in repaired_rows[:args.viz]:
            mesh = EditableMesh.load(*mesh_paths[row["name"]])
            fixed_mesh, operations = repair(mesh, RepairConfig())
            before_stats = stats(mesh)
            after_stats = stats(fixed_mesh)
            subtitle = (f"연산 {len(operations)}회 · 삼각형 {before_stats['삼각형']} → {after_stats['삼각형']} · "
                        f"불량각 요소 {before_stats['불량각 요소']} → {after_stats['불량각 요소']}")
            path = os.path.join(gae_paths.OUT_DIR, f"compare_{row['name']}.html")
            build_html(mesh, fixed_mesh, operations, path, subtitle)
            print(f"  {path}")


if __name__ == "__main__":
    main()

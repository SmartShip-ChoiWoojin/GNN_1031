"""수정 쌍 508건을 종류별로 묶어 .md 목록으로 내보낸다 (사람이 하나씩 열어보며 확인하는 문서).

절점 병합 판정은 `nodemap.build_node_map` 을 쓴다 (연결관계 기준). 좌표로는 450건 중
99건밖에 못 잡고, 셋이 하나로 합쳐지는 경우(`_0021`)나 합친 뒤 자리를 다시 잡은 경우(`_1081`)를 놓친다.

    python src/export_pair_list.py            # out/pair_list.md + out/pair_list.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import gae_paths
from bdf_io import discover_pairs
from meshrep import Edge
from nodemap import XYZ_TOL, build_node_map, where_survivor_landed
from pairs import GOOD_DIR, all_pairs, count_triangles, discover_good_pairs, load_pair
from regen_data import shell_edge_set


def collect() -> pd.DataFrame:
    """쌍마다 한 줄 — 사라진/생긴 절점 수, 병합 판정, 쉘·1D 변화."""
    good_paths = discover_good_pairs()
    original_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    rows = []
    count = 0
    for name in all_pairs():
        count = count + 1
        pair = load_pair(name, good_paths, original_paths)
        before = pair.before
        after = pair.after
        node_map = build_node_map(before, after)
        landed = where_survivor_landed(before, after, node_map)

        gone_nodes = set(before.coords) - set(after.coords)
        new_nodes = set(after.coords) - set(before.coords)
        # 좌표이동은 pairs.match_nodes 가 판정해 둔 것을 쓴다 — 거기서 "사라진 절점을 흡수하느라
        # 따라 움직인 것" 을 이미 뺐다.
        pure_moved = set(pair.nodes.moved)
        extra_move = set()
        for survivor_node, where in landed.items():
            if where == "그 밖의 위치":
                extra_move.add(survivor_node)

        edges_before = shell_edge_set(before)
        edges_after = shell_edge_set(after)
        lines_before = set()
        for element_id, element_type, node_ids in before.lines:
            lines_before.add(Edge(tuple(node_ids)))
        lines_after = set()
        for element_id, element_type, node_ids in after.lines:
            lines_after.add(Edge(tuple(node_ids)))
        before_shells = set()
        for element_id, element_type, node_ids in before.shells:
            before_shells.add(frozenset(node_ids))
        after_shells = set()
        for element_id, element_type, node_ids in after.shells:
            after_shells.add(frozenset(node_ids))

        # 좌표가 겹쳐 버린 절점 (병합이 아니라 작업 실수로 보임)
        nodes_at_position = {}
        for node_id, point in after.coords.items():
            key = tuple(np.round(np.asarray(point, float) / XYZ_TOL).astype(np.int64))
            if key not in nodes_at_position:
                nodes_at_position[key] = []
            nodes_at_position[key].append(node_id)
        coincident_count = 0
        for node_list in nodes_at_position.values():
            if len(node_list) > 1:
                coincident_count = coincident_count + 1

        weak_count = 0
        nonadjacent_count = 0
        for basis_text in node_map.basis.values():
            if basis_text.startswith("퇴화"):
                weak_count = weak_count + 1
            if "비인접" in basis_text or basis_text == "퇴화추론+좌표":
                nonadjacent_count = nonadjacent_count + 1
        removed_shell_count = 0
        for shell_key in before_shells:
            if shell_key not in after_shells:
                removed_shell_count = removed_shell_count + 1
        created_shell_count = 0
        for shell_key in after_shells:
            if shell_key not in before_shells:
                created_shell_count = created_shell_count + 1
        landed_parts = []
        for survivor_node, where in sorted(landed.items()):
            landed_parts.append(f"{survivor_node}:{where}")

        rows.append(dict(
            name=name,
            n_del_total=len(gone_nodes),           # 사라진 절점 전부 (= 절점삭제 축)
            n_merge=len(node_map.absorbed), n_drop=len(node_map.dropped), n_repl=len(node_map.replaced_by_new),
            n_add=len(new_nodes), n_mov=len(pure_moved), n_extra_move=len(extra_move),
            n_coin=coincident_count,
            weak=weak_count,
            nonadj=nonadjacent_count,
            explained=node_map.explained,
            edge_chg=len(edges_before - edges_after) + len(edges_after - edges_before),
            line_gone=len(lines_before - lines_after), line_new=len(lines_after - lines_before),
            n_rm=removed_shell_count,
            n_cr=created_shell_count,
            tri_b=count_triangles(before), tri_a=count_triangles(after),
            land=";".join(landed_parts),
        ))
        if count % 100 == 0:
            print(f"  ... {count}/508")
    table = pd.DataFrame(rows)
    kinds = []
    for _, row in table.iterrows():
        kinds.append(kind_of(row))
    table["kind"] = kinds
    return table


def kind_of(row) -> str:
    """네 축으로 판정하되 이름에서는 '변변화' 를 뺀다 (regen_data.kind_of 와 같은 규칙, 표의 열로 판정)."""
    parts = []
    if row.n_del_total:
        parts.append("절점삭제")
    if row.n_add:
        parts.append("절점추가")
    if row.n_mov:
        parts.append("좌표이동")
    if len(parts) == 0:
        if row.edge_chg:
            return "연결만 변경"
        return "변화없음"
    kind = " + ".join(parts)
    if not row.edge_chg:
        kind = kind + " (연결 변화 없음)"
    return kind


def probability_text(row) -> str:
    if np.isnan(row.p_b):
        return "–"
    return f"{row.p_b:.3f} → {row.p_a:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="수정 쌍 목록을 .md 로 내보낸다")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    gae_paths.ensure_dirs()
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(gae_paths.OUT_DIR, "pair_list.md")
    print("쌍 508건을 훑는 중...")
    table = collect()

    judge_path = os.path.join(gae_paths.OUT_DIR, "judge_pairs.csv")
    if not os.path.exists(judge_path):
        judge_path = os.path.join(gae_paths.ORIGINAL_GAE_ROOT, "out", "judge_pairs.csv")
    if os.path.exists(judge_path):
        judge_table = pd.read_csv(judge_path).set_index("name")
        probabilities_before = []
        probabilities_after = []
        for name in table.name:
            if name in judge_table.index:
                probabilities_before.append(judge_table.loc[name, "p_before"])
                probabilities_after.append(judge_table.loc[name, "p_after"])
            else:
                probabilities_before.append(np.nan)
                probabilities_after.append(np.nan)
        table["p_b"] = probabilities_before
        table["p_a"] = probabilities_after
    else:
        table["p_b"] = np.nan
        table["p_a"] = np.nan

    good_relative = os.path.relpath(GOOD_DIR, gae_paths.PARENT_ROOT).replace(os.sep, "/")
    kind_order = table.groupby("kind").size().sort_values(ascending=False).index.tolist()

    lines = ["# 수정 쌍 508건 — 종류별 목록\n"]
    lines.append(f"원본 `SN2592_FWD_Clean/<이름>.bdf` · 수정본 `{good_relative}/<이름>_good.bdf`\n")
    lines.append("`p` 는 심판 분류기(`runs/train/deploy_10_14`)가 본 비정상 확률. "
                 "이 모델은 이 메쉬들을 학습에 쓰지 않았습니다.\n")
    lines.append("확인한 항목은 `☐` → `☑` 로 바꾸세요.\n")

    lines.append("\n## 절점 병합 판정에 대해\n")
    lines.append("좌표가 아니라 **연결관계**로 판정했습니다 — 사라진 쉘이 새 쉘로 바뀔 때 "
                 "누가 그 자리를 이어받았는지 봅니다. 좌표만 보면 합친 뒤 자리를 다시 잡은 "
                 "경우(`_1081`)나 셋이 하나로 합쳐진 경우(`_0021`)를 놓칩니다.\n")
    merge_total = int(table.n_merge.sum())
    drop_total = int(table.n_drop.sum())
    replace_total = int(table.n_repl.sum())
    nonadjacent_total = int(table.nonadj.sum())
    lines.append(f"- 사라진 절점 {int(table.n_merge.sum() + table.n_drop.sum() + table.n_repl.sum())}개 → "
                 f"**병합 {merge_total}** / 순수삭제 {drop_total} / 신규교체 {replace_total}")
    lines.append(f"- 판정의 {int(table.n_merge.sum() - table.nonadj.sum())}건은 **원래 변으로 이어진 "
                 f"이웃**끼리입니다. 이어지지 않은 {nonadjacent_total}건은 아래 '먼저 봐야 할 것' 에 "
                 f"따로 모아 뒀습니다 — `_0032` 에서 쉘이 우연히 한 자리만 달라 엉뚱한 절점이 "
                 f"흡수자로 뽑힌 적이 있어, 같은 종류의 오판일 수 있습니다.")
    lines.append("- `검산` 은 병합된 절점을 쓰던 쉘이 대응과 맞아떨어지는 비율입니다. "
                 "1.00 이 아니어도 판정이 틀렸다는 뜻은 아닙니다 — 병합과 **동시에 주변을 "
                 "다시 깔면** 맞아떨어지지 않습니다. 직접 봐주셨으면 하는 부분입니다.\n")

    # ── 요약 ──
    lines.append("\n## 종류 요약\n")
    lines.append("| 종류 | 건수 | 비율 | 쉘 삭제→생성 | 1D 바뀜 | 검산 1.00 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for kind in kind_order:
        kind_table = table[table.kind == kind]
        line_changed_rate = ((kind_table.line_gone + kind_table.line_new) > 0).mean()
        explained_rate = (kind_table.explained >= 0.999).mean()
        lines.append(f"| {kind} | {len(kind_table)} | {100 * len(kind_table) / len(table):.1f}% | "
                     f"{kind_table.n_rm.median():.0f}→{kind_table.n_cr.median():.0f} | "
                     f"{100 * line_changed_rate:.0f}% | {100 * explained_rate:.0f}% |")
    lines.append(f"\n합계 **{len(table)}건**, 조합 {len(kind_order)}종. 숫자는 중앙값.\n")

    # ── 먼저 볼 것 ──
    def add_table(sub_table) -> None:
        lines.append("| ☐ | 이름 | 종류 | 쉘 | 절점 삭제/추가/이동 | 승계(병합/신규/없음) | 검산 | p 전 → 후 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in sub_table.itertuples():
            lines.append(f"| ☐ | `{row.name}` | {row.kind} | {row.n_rm}→{row.n_cr} | "
                         f"{row.n_del_total}/{row.n_add}/{row.n_mov} | "
                         f"{row.n_merge}/{row.n_repl}/{row.n_drop} | {row.explained:.2f} | {probability_text(row)} |")

    lines.append("\n## 먼저 봐야 할 것\n")
    sections = [
        ("검산이 1.00 이 아닌 쌍 — 병합과 재배치가 겹친 것으로 보임",
         table[table.explained < 0.999].sort_values("explained")),
        ("병합한 뒤 **또 옮긴** 절점이 있는 쌍", table[table.n_extra_move > 0]),
        ("좌표가 겹친 절점이 있는 쌍 (작업 실수로 보임)", table[table.n_coin > 0]),
        ("아무도 이어받지 않은 **순수 삭제** 가 있는 쌍", table[table.n_drop > 0]),
        ("사라진 절점을 **새 절점이 대신한** 쌍", table[table.n_repl > 0]),
        ("**비인접 판정** 이 있는 쌍 — `_0032` 같은 오판일 수 있어 꼭 확인 필요", table[table.nonadj > 0]),
        ("약한 근거(퇴화추론)로 판정된 병합이 있는 쌍", table[table.weak > 0]),
        ("고쳤는데 심판 확률이 **올라간** 쌍", table[table.p_a > table.p_b].sort_values("p_a", ascending=False)),
        ("원본과 파일이 완전히 같은 쌍", table[table.kind == "변화없음"]),
    ]
    for title, sub_table in sections:
        lines.append(f"\n### {title} — {len(sub_table)}건\n")
        if len(sub_table) > 0:
            add_table(sub_table)
        else:
            lines.append("없음\n")

    # ── 종류별 전체 ──
    lines.append("\n\n## 종류별 전체 목록\n")
    for kind in kind_order:
        kind_table = table[table.kind == kind].sort_values("name")
        lines.append(f"\n### {kind} — {len(kind_table)}건\n")
        lines.append("| ☐ | 이름 | 쉘 삭제→생성 | 절점 삭제/추가/이동 | 승계(병합/신규/없음) | 1D | 삼각형 | 검산 | p 전 → 후 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in kind_table.itertuples():
            if row.line_gone or row.line_new:
                line_text = f"{row.line_gone}/{row.line_new}"
            else:
                line_text = "–"
            lines.append(f"| ☐ | `{row.name}` | {row.n_rm}→{row.n_cr} | "
                         f"{row.n_del_total}/{row.n_add}/{row.n_mov} | "
                         f"{row.n_merge}/{row.n_repl}/{row.n_drop} | {line_text} | "
                         f"{row.tri_b}→{row.tri_a} | {row.explained:.2f} | {probability_text(row)} |")

    with open(out_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines) + "\n")
    print(f"\n저장: {out_path}  ({len(table)}건 / {len(kind_order)}종)")
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"      {csv_path}")


if __name__ == "__main__":
    main()

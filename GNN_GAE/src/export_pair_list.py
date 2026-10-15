"""수정 쌍 508건을 종류별로 묶어 .md 목록으로 내보낸다.

사람이 하나씩 열어보며 확인하려고 만드는 문서다.

절점 병합 판정은 `nodemap.build_node_map` 을 쓴다 (연결관계 기준). `pairs.match_nodes`
의 좌표 기준 판정은 **쓰지 않는다** — 좌표로는 450건 중 99건밖에 못 잡고, 셋이 하나로
합쳐지는 경우(`_0021`)나 합친 뒤 자리를 다시 잡은 경우(`_1081`)를 놓친다.

    python src/export_pair_list.py            # out/pair_list.md
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import gae_paths as gp
from bdf_io import discover_pairs
from meshrep import Edge
from nodemap import XYZ_TOL, build_node_map, where_survivor_landed
from pairs import GOOD_DIR, all_pairs, discover_good_pairs, load_pair


def shell_edges(mesh) -> set:
    return {Edge(e) for _, _, n in mesh.shells for e in zip(n, n[1:] + n[:1])}


def collect() -> pd.DataFrame:
    good = discover_good_pairs()
    orig, _, _ = discover_pairs(gp.FE_DIR)
    rows = []
    for i, nm in enumerate(all_pairs(), 1):
        p = load_pair(nm, good, orig)
        b, a = p.before, p.after
        m = build_node_map(b, a)
        land = where_survivor_landed(b, a, m)

        gone = set(b.coords) - set(a.coords)
        new = set(a.coords) - set(b.coords)
        # 좌표이동은 pairs.match_nodes 가 이미 판정해 둔 것을 쓴다 — 거기서 "사라진
        # 절점을 흡수하느라 따라 움직인 것"을 이미 빼 놨다. 여기서 좌표만으로 다시
        # 세면 `_1081`(흡수 뒤 중간 지점에 자리잡은 경우)처럼 화면과 목록이 어긋난다.
        pure_moved = set(p.nodes.moved)
        extra_move = {s for s, w in land.items() if w == "그 밖의 위치"}

        eb, ea = shell_edges(b), shell_edges(a)
        lb = {Edge(tuple(n)) for _, _, n in b.lines}
        la = {Edge(tuple(n)) for _, _, n in a.lines}
        bset = {frozenset(n) for _, _, n in b.shells}
        aset = {frozenset(n) for _, _, n in a.shells}
        # 좌표가 겹쳐 버린 절점 (병합이 아니라 작업 실수로 보임)
        akey = {}
        for n, c in a.coords.items():
            akey.setdefault(tuple(np.round(np.asarray(c, float) / XYZ_TOL).astype(np.int64)), []).append(n)
        coincident = sum(1 for v in akey.values() if len(v) > 1)

        tri = lambda mesh: sum(1 for _, t, _ in mesh.shells if t == "CTRIA3")
        rows.append(dict(
            name=nm,
            n_del_total=len(gone),            # 사라진 절점 전부 (= 절점삭제 축)
            n_merge=len(m.absorbed), n_drop=len(m.dropped), n_repl=len(m.replaced_by_new),
            n_add=len(new), n_mov=len(pure_moved), n_extra_move=len(extra_move),
            n_coin=coincident,
            weak=sum(1 for v in m.basis.values() if v.startswith("퇴화")),
            nonadj=sum(1 for v in m.basis.values() if "비인접" in v
                       or v == "퇴화추론+좌표"),
            explained=m.explained,
            edge_chg=len(eb - ea) + len(ea - eb),
            line_gone=len(lb - la), line_new=len(la - lb),
            n_rm=sum(1 for s in bset if s not in aset),
            n_cr=sum(1 for s in aset if s not in bset),
            tri_b=tri(b), tri_a=tri(a),
            land=";".join(f"{k}:{v}" for k, v in sorted(land.items())),
        ))
        if i % 100 == 0:
            print(f"  ... {i}/508")
    df = pd.DataFrame(rows)
    df["kind"] = df.apply(kind_of, axis=1)
    return df


def kind_of(r) -> str:
    """네 축으로 판정하되 이름에서는 '변변화' 를 뺀다 — 508쌍 중 503쌍에 붙어 있어
    이름에 넣으면 길기만 하고 구분이 안 된다. 대신 **연결이 안 바뀐 5건**만 따로 적는다:
    절점도 연결도 그대로면 "변화없음"(4건), 절점만 움직이고 연결은 그대로면
    "(연결 변화 없음)" 을 덧붙인다(`_0267` 1건).

    '절점병합' 과 '신규교체' 를 축에서 내린 이유: 파일 수준에서는 둘 다 **절점이
    사라진 것**이고, 그 참조를 누가 이어받았는지는 세부 속성이다 (451 대 12 대 18 로
    쏠려 분류 값어치가 없다). 승계 정보는 열/표에 그대로 남겨 둔다.
    """
    parts = []
    if r.n_del_total:
        parts.append("절점삭제")
    if r.n_add:
        parts.append("절점추가")
    if r.n_mov:
        parts.append("좌표이동")
    if not parts:
        return "연결만 변경" if r.edge_chg else "변화없음"
    return " + ".join(parts) + ("" if r.edge_chg else " (연결 변화 없음)")


def main() -> None:
    ap = argparse.ArgumentParser(description="수정 쌍 목록을 .md 로 내보낸다")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gp.ensure_dirs()
    out = a.out or os.path.join(gp.OUT_DIR, "pair_list.md")
    print("쌍 508건을 훑는 중...")
    df = collect()

    jp_path = os.path.join(gp.OUT_DIR, "judge_pairs.csv")
    jp = pd.read_csv(jp_path).set_index("name") if os.path.exists(jp_path) else None
    if jp is not None:
        df["p_b"] = [jp.loc[n, "p_before"] if n in jp.index else np.nan for n in df.name]
        df["p_a"] = [jp.loc[n, "p_after"] if n in jp.index else np.nan for n in df.name]
    else:
        df["p_b"] = df["p_a"] = np.nan

    good_rel = os.path.relpath(GOOD_DIR, gp.PARENT_ROOT).replace(os.sep, "/")
    order = df.groupby("kind").size().sort_values(ascending=False).index.tolist()

    L = ["# 수정 쌍 508건 — 종류별 목록\n"]
    L.append(f"원본 `SN2592_FWD_Clean/<이름>.bdf` · 수정본 `{good_rel}/<이름>_good.bdf`\n")
    L.append("`p` 는 심판 분류기(`runs/train/deploy_10_14`)가 본 비정상 확률. "
             "이 모델은 이 메쉬들을 학습에 쓰지 않았습니다.\n")
    L.append("확인한 항목은 `☐` → `☑` 로 바꾸세요.\n")

    L.append("\n## 절점 병합 판정에 대해\n")
    L.append("좌표가 아니라 **연결관계**로 판정했습니다 — 사라진 쉘이 새 쉘로 바뀔 때 "
             "누가 그 자리를 이어받았는지 봅니다. 좌표만 보면 합친 뒤 자리를 다시 잡은 "
             "경우(`_1081`)나 셋이 하나로 합쳐진 경우(`_0021`)를 놓칩니다.\n")
    L.append(f"- 사라진 절점 {int(df.n_merge.sum() + df.n_drop.sum() + df.n_repl.sum())}개 → "
             f"**병합 {int(df.n_merge.sum())}** / 순수삭제 {int(df.n_drop.sum())} / "
             f"신규교체 {int(df.n_repl.sum())}")
    L.append(f"- 판정의 {int(df.n_merge.sum() - df.nonadj.sum())}건은 **원래 변으로 이어진 "
             f"이웃**끼리입니다. 이어지지 않은 {int(df.nonadj.sum())}건은 아래 '먼저 봐야 할 것' 에 "
             f"따로 모아 뒀습니다 — `_0032` 에서 쉘이 우연히 한 자리만 달라 엉뚱한 절점이 "
             f"흡수자로 뽑힌 적이 있어, 같은 종류의 오판일 수 있습니다.")
    L.append(f"- `검산` 은 병합된 절점을 쓰던 쉘이 대응과 맞아떨어지는 비율입니다. "
             f"1.00 이 아니어도 판정이 틀렸다는 뜻은 아닙니다 — 병합과 **동시에 주변을 "
             f"다시 깔면** 맞아떨어지지 않습니다. 직접 봐주셨으면 하는 부분입니다.\n")

    # ── 요약 ──
    L.append("\n## 종류 요약\n")
    L.append("| 종류 | 건수 | 비율 | 쉘 삭제→생성 | 1D 바뀜 | 검산 1.00 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k in order:
        g = df[df.kind == k]
        l1d = ((g.line_gone + g.line_new) > 0).mean()
        ok = (g.explained >= 0.999).mean()
        L.append(f"| {k} | {len(g)} | {100*len(g)/len(df):.1f}% | "
                 f"{g.n_rm.median():.0f}→{g.n_cr.median():.0f} | {100*l1d:.0f}% | {100*ok:.0f}% |")
    L.append(f"\n합계 **{len(df)}건**, 조합 {len(order)}종. 숫자는 중앙값.\n")

    # ── 먼저 볼 것 ──
    def table(sub, cols_note=""):
        L.append("| ☐ | 이름 | 종류 | 쉘 | 절점 삭제/추가/이동 | 승계(병합/신규/없음) | 검산 | p 전 → 후 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in sub.itertuples():
            p = f"{r.p_b:.3f} → {r.p_a:.3f}" if not np.isnan(r.p_b) else "–"
            L.append(f"| ☐ | `{r.name}` | {r.kind} | {r.n_rm}→{r.n_cr} | "
                     f"{r.n_del_total}/{r.n_add}/{r.n_mov} | "
                     f"{r.n_merge}/{r.n_repl}/{r.n_drop} | {r.explained:.2f} | {p} |")

    L.append("\n## 먼저 봐야 할 것\n")
    sec = [
        ("검산이 1.00 이 아닌 쌍 — 병합과 재배치가 겹친 것으로 보임",
         df[df.explained < 0.999].sort_values("explained")),
        ("병합한 뒤 **또 옮긴** 절점이 있는 쌍", df[df.n_extra_move > 0]),
        ("좌표가 겹친 절점이 있는 쌍 (작업 실수로 보임)", df[df.n_coin > 0]),
        ("아무도 이어받지 않은 **순수 삭제** 가 있는 쌍", df[df.n_drop > 0]),
        ("사라진 절점을 **새 절점이 대신한** 쌍", df[df.n_repl > 0]),
        ("**비인접 판정** 이 있는 쌍 — `_0032` 같은 오판일 수 있어 꼭 확인 필요",
         df[df.nonadj > 0]),
        ("약한 근거(퇴화추론)로 판정된 병합이 있는 쌍", df[df.weak > 0]),
        ("고쳤는데 심판 확률이 **올라간** 쌍", df[df.p_a > df.p_b].sort_values("p_a", ascending=False)),
        ("원본과 파일이 완전히 같은 쌍", df[df.kind == "변화없음"]),
    ]
    for title, sub in sec:
        L.append(f"\n### {title} — {len(sub)}건\n")
        if len(sub):
            table(sub)
        else:
            L.append("없음\n")

    # ── 종류별 전체 ──
    L.append("\n\n## 종류별 전체 목록\n")
    for k in order:
        g = df[df.kind == k].sort_values("name")
        L.append(f"\n### {k} — {len(g)}건\n")
        L.append("| ☐ | 이름 | 쉘 삭제→생성 | 절점 삭제/추가/이동 | 승계(병합/신규/없음) | 1D | 삼각형 | 검산 | p 전 → 후 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in g.itertuples():
            lines = f"{r.line_gone}/{r.line_new}" if (r.line_gone or r.line_new) else "–"
            p = f"{r.p_b:.3f} → {r.p_a:.3f}" if not np.isnan(r.p_b) else "–"
            L.append(f"| ☐ | `{r.name}` | {r.n_rm}→{r.n_cr} | "
                     f"{r.n_del_total}/{r.n_add}/{r.n_mov} | "
                     f"{r.n_merge}/{r.n_repl}/{r.n_drop} | {lines} | "
                     f"{r.tri_b}→{r.tri_a} | {r.explained:.2f} | {p} |")

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n저장: {out}  ({len(df)}건 / {len(order)}종)")
    csv = os.path.splitext(out)[0] + ".csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"      {csv}")


if __name__ == "__main__":
    main()

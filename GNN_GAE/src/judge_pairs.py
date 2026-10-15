"""수정 전/후 쌍을 기존 분류기에 통과시켜 본다.

이 프로젝트에서 제일 먼저 확인해야 할 질문이다 — **사람이 고친 결과를 분류기가
정상이라고 보는가?**

  - 본다면: 분류기를 수리기의 심판으로 쓸 수 있고, "p 를 낮추는 쪽" 이 곧 목표가 된다.
  - 안 본다면: 분류기의 '비정상' 과 사람의 '고쳐야 할 것' 이 다른 것을 보고 있다는 뜻이라,
    목표 함수를 분류기 확률로 잡으면 안 된다.

분류기는 **수리 학습에 절대 쓰지 않는다**. 여기서는 심판으로만 쓴다.

    python src/judge_pairs.py
    python src/judge_pairs.py --run ../runs/train/deploy0918
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import gae_paths as gp
import dataset as DS
import graphs
import paths as parent_paths
from bdf_io import discover_pairs
from explain import load_full_model
from meshrep import MeshRep
from pairs import all_pairs, discover_good_pairs, load_pair


def pick_deploy_run() -> str:
    """현재 라벨로 학습된 배포본 폴더. ``predict_all.pick_deploy_run`` 과 같은 규칙."""
    import glob

    cur = os.path.basename(parent_paths.LABEL_XLSX)
    for d in sorted(glob.glob(os.path.join(parent_paths.RUNS_TRAIN, "*"))):
        if os.path.basename(d).startswith("calib"):
            continue
        rj = os.path.join(d, "result.json")
        if not (os.path.isfile(rj) and os.path.isfile(os.path.join(d, "full.pt"))):
            continue
        with open(rj, encoding="utf-8") as fh:
            if json.load(fh).get("label_xlsx") == cur:
                return d
    raise SystemExit("현재 라벨로 학습된 배포본을 찾지 못했습니다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="수정 전/후를 분류기에 통과시킨다")
    ap.add_argument("--run", default=None, help="배포 모델 폴더 (기본: 자동 선택)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int, help="앞에서 N쌍만")
    a = ap.parse_args()

    run = a.run or pick_deploy_run()
    device = torch.device("cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu")
    model, norm, args, platt = load_full_model(run, device)
    opts = {}
    if args.graph == "elem":
        opts = {"corner_edges": not args.no_corner_edges, "include_lines": not args.no_lines}
    print(f"[judge] {run} · device={device} · graph={args.graph}")

    def prob(mesh) -> float:
        ga = graphs.build(MeshRep(mesh), args.graph, **opts)
        d = DS.to_pyg(ga, 0, mesh.name, norm).to(device)
        d.batch = torch.zeros(d.x.size(0), dtype=torch.long, device=device)
        logit = float(model(d).item())
        return 1.0 / (1.0 + np.exp(-(platt[0] * logit + platt[1])))

    good = discover_good_pairs()
    orig, _, _ = discover_pairs(gp.FE_DIR)
    names = all_pairs()[: a.limit] if a.limit else all_pairs()

    rows = []
    model.eval()
    with torch.no_grad():
        for i, nm in enumerate(names, 1):
            p = load_pair(nm, good, orig)
            rows.append((nm, prob(p.before), prob(p.after)))
            if i % 100 == 0:
                print(f"  ... {i}/{len(names)}")

    gp.ensure_dirs()
    out = os.path.join(gp.OUT_DIR, "judge_pairs.csv")
    with open(out, "w", encoding="utf-8-sig") as fh:
        fh.write("name,p_before,p_after\n")
        for nm, b, c in rows:
            fh.write(f"{nm},{b:.6f},{c:.6f}\n")

    B = np.array([r[1] for r in rows])
    A = np.array([r[2] for r in rows])
    print(f"\n쌍 {len(rows)}개 — 분류기가 보는 '비정상 확률'")
    print(f"  수정 전  평균 {B.mean():.3f}  중앙값 {np.median(B):.3f}  "
          f"비정상 판정(p>0.5) {np.mean(B > 0.5):.1%}")
    print(f"  수정 후  평균 {A.mean():.3f}  중앙값 {np.median(A):.3f}  "
          f"비정상 판정(p>0.5) {np.mean(A > 0.5):.1%}")
    print(f"  p 가 내려간 쌍: {np.mean(A < B):.1%}   평균 변화 {np.mean(A - B):+.3f}")
    print(f"  '비정상 → 정상' 으로 뒤집힌 쌍: {np.mean((B > 0.5) & (A <= 0.5)):.1%}")
    print(f"  수정 후에도 비정상으로 남은 쌍:  {np.mean((B > 0.5) & (A > 0.5)):.1%}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()

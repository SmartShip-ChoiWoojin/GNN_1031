"""배포 모델(full.pt)로 라벨이 있는 전 메쉬를 한 번 예측해 csv 로 저장한다.

앱이 오분류 목록이나 집합별 일치율을 보여줄 때마다 수천 번 추론하면 느리다.
배포 모델은 고정이므로 결과도 고정 - 배포 전에 한 번 계산해 같이 넣어둔다.

    python src/predict_all.py --run runs/train/deploy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import graphs
import dataset as DS
import paths
from bdf_io import discover_pairs, load_raw_mesh
from explain import load_full_model
from labels import load_label_table
from meshrep import MeshRep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="배포 모델로 전 메쉬 예측 캐시 생성")
    ap.add_argument("--run", default=os.path.join(paths.RUNS_TRAIN, "deploy"))
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu")
    model, norm, args, platt = load_full_model(a.run, device)
    opts = {}
    if args.graph == "elem":
        opts = {"corner_edges": not args.no_corner_edges, "include_lines": not args.no_lines}

    table = load_label_table(paths.LABEL_XLSX)
    paired, _, _ = discover_pairs(paths.FE_DIR)
    print(f"[predict_all] {a.run} · device={device} · 대상 {len(table)}행")

    rows = []
    model.eval()
    with torch.no_grad():
        for i, r in enumerate(table.itertuples(), start=1):
            if r.name not in paired:
                continue
            bdf, js = paired[r.name]
            ga = graphs.build(MeshRep(load_raw_mesh(bdf, js)), args.graph, **opts)
            d = DS.to_pyg(ga, 0, r.name, norm).to(device)
            d.batch = torch.zeros(d.x.size(0), dtype=torch.long, device=device)
            logit = float(model(d).item())
            prob = 1.0 / (1.0 + np.exp(-(platt[0] * logit + platt[1])))
            rows.append((r.name, int(r.cls), int(r.ambig), int(r.trainable), prob))
            if i % 500 == 0:
                print(f"  ... {i}/{len(table)}")

    out = os.path.join(a.run, "predictions.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("name,cls,ambig,trainable,prob" + chr(10))
        for nm, cls, amb, tr, p in rows:
            fh.write(f"{nm},{cls},{amb},{tr},{p:.6f}" + chr(10))

    P = np.array([r[4] for r in rows])
    C = np.array([r[1] for r in rows])
    T = np.array([r[3] for r in rows], dtype=bool)
    known = C <= 1
    agree = ((P > 0.5).astype(int) == C)[known]
    print(f"[predict_all] 저장 {out}  ({len(rows)}개)")
    print(f"  라벨이 정상/비정상인 {known.sum()}개 기준 일치율 {agree.mean():.4f}")
    print(f"    train {int((known & T).sum())}개 : "
          f"{((P > 0.5).astype(int) == C)[known & T].mean():.4f}")
    print(f"    test  {int((known & ~T).sum())}개 : "
          f"{((P > 0.5).astype(int) == C)[known & ~T].mean():.4f}")
    return 0


if __name__ == "__main__":
    from fasoo_drm import setting_fasoo

    setting_fasoo()
    sys.exit(main())

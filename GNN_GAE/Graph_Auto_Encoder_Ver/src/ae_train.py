"""GAE 학습 — 정상 메쉬로만.

    python src/ae_train.py --mode mae --tag mae
    python src/ae_train.py --mode ae  --tag ae
    python src/ae_train.py --mode ae  --tag ae_nograph --layers 0

결과: ``runs/<tag>/`` 에 model.pt (가중치 + 설정 + 정규화기) 와 history.csv.
정규화 통계는 학습 정상 메쉬에서만 적합한다. 조기종료는 검증 정상 메쉬의 손실로 한다
(수정 쌍은 여기서 한 번도 보지 않는다).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import ae_paths as ap
from ae_data import load_normal_graphs, normal_split
from ae_model import GraphAE
from dataset import NormalizerSet, _valid_of


def to_data(ga, norm: NormalizerSet, name: str = "") -> Data:
    xv = _valid_of(ga, "x")
    x = torch.from_numpy(norm.node.transform(ga.x, xv))
    if ga.n_edge:
        ea = torch.from_numpy(norm.edge.transform(ga.edge_attr, _valid_of(ga, "e")))
    else:
        ea = torch.zeros((0, len(norm.edge.spec)), dtype=torch.float32)
    return Data(x=x, edge_index=torch.from_numpy(ga.edge_index), edge_attr=ea,
                x_valid=torch.from_numpy(xv), node_kind=torch.from_numpy(ga.node_kind),
                name=name)


def load_run(run: str, device):
    ck = torch.load(os.path.join(run, "model.pt"), map_location=device, weights_only=False)
    model = GraphAE(**ck["cfg"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, NormalizerSet.from_state_dict(ck["norm"]), ck


@torch.no_grad()
def eval_loss(model, loader, device, seed: int = 123) -> float:
    """검증 손실 — 가림 위치를 고정해서 epoch 끼리 비교가 되게 한다."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot, n = 0.0, 0
    for d in loader:
        d = d.to(device)
        l, _ = model.loss(d, gen)
        tot += float(l) * d.num_graphs
        n += d.num_graphs
    return tot / n


def main() -> None:
    ar = argparse.ArgumentParser(description="GAE 학습 (정상 메쉬만)")
    ar.add_argument("--mode", choices=("mae", "ae"), default="mae")
    ar.add_argument("--tag", required=True)
    ar.add_argument("--hidden", type=int, default=128)
    ar.add_argument("--layers", type=int, default=3)
    ar.add_argument("--latent", type=int, default=16)
    ar.add_argument("--mask-rate", type=float, default=0.3)
    ar.add_argument("--struct-w", type=float, default=1.0)
    ar.add_argument("--dropout", type=float, default=0.1)
    ar.add_argument("--epochs", type=int, default=200)
    ar.add_argument("--patience", type=int, default=30)
    ar.add_argument("--lr", type=float, default=1e-3)
    ar.add_argument("--batch-size", type=int, default=64)
    ar.add_argument("--seed", type=int, default=0)
    a = ar.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = normal_split()
    tr_names = sp[sp.split == "train"].name.tolist()
    va_names = sp[sp.split == "val"].name.tolist()
    tr_g = load_normal_graphs(tr_names)
    va_g = load_normal_graphs(va_names)
    norm = NormalizerSet("elem").fit(tr_g, np.arange(len(tr_g)))
    tr = [to_data(g, norm, n) for g, n in zip(tr_g, tr_names)]
    va = [to_data(g, norm, n) for g, n in zip(va_g, va_names)]
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True)
    vl = DataLoader(va, batch_size=128)

    model = GraphAE(d_node=tr[0].x.size(1), d_edge=tr[0].edge_attr.size(1), mode=a.mode,
                    hidden=a.hidden, layers=a.layers, latent=a.latent, dropout=a.dropout,
                    mask_rate=a.mask_rate, struct_w=a.struct_w).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    print(f"[{a.tag}] mode={a.mode} layers={a.layers} 학습 {len(tr)} / 검증 {len(va)} "
          f"/ 파라미터 {sum(p.numel() for p in model.parameters()):,}  ({dev})")

    run = os.path.join(ap.RUNS_DIR, a.tag)
    os.makedirs(run, exist_ok=True)
    best, best_ep, hist = float("inf"), -1, []
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, parts = 0.0, {}
        for d in tl:
            d = d.to(dev)
            loss, p = model.loss(d, gen)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * d.num_graphs
            for k, v in p.items():
                parts[k] = parts.get(k, 0.0) + v * d.num_graphs
        sched.step()
        trl = tot / len(tr)
        val = eval_loss(model, vl, dev)
        hist.append({"epoch": ep, "train": trl, "val": val,
                     **{f"train_{k}": v / len(tr) for k, v in parts.items()}})
        if val < best - 1e-5:
            best, best_ep = val, ep
            torch.save({"cfg": model.cfg, "state": model.state_dict(),
                        "norm": norm.state_dict(), "args": vars(a), "epoch": ep,
                        "val": val}, os.path.join(run, "model.pt"))
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}  train {trl:.4f}  val {val:.4f}  (best {best:.4f} @ {best_ep})"
                  f"  {time.time() - t0:.0f}s")
        if ep - best_ep >= a.patience:
            print(f"  조기종료 — {a.patience} epoch 동안 개선 없음")
            break
    pd.DataFrame(hist).to_csv(os.path.join(run, "history.csv"), index=False)
    print(f"[{a.tag}] 최적 epoch {best_ep}, 검증 손실 {best:.4f} → {run}")


if __name__ == "__main__":
    main()

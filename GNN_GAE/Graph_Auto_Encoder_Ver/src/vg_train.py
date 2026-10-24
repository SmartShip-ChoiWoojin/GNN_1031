"""VGAE 학습 — 508쌍 전부를 학습에 쓴다 (테스트는 나중에).

손실 = 변 복원 BCE + 절점 남음 BCE + β·KL

  - 변 복원: 메쉬 안의 모든 절점 쌍. 정답은 Good 의 연결.
    이어진 쌍은 드물어서(약 7%) pos_w 로, 바뀌는 쌍(Bad 와 Good 이 다른 쌍)은 더 드물어서
    w_chg 로 가중한다.
  - 절점 남음: 사라지는 절점이 메쉬당 1개꼴이라 w_rm 으로 가중한다.

    python src/vg_train.py --tag vgae
    python src/vg_train.py --tag vgae_skip --skip
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as Fnn
from torch_geometric.loader import DataLoader

import vg_paths as vp
from vg_model import CONV_KINDS, MeshVGAE, Norm, to_data


def load_data():
    with open(vp.DATA_PATH, "rb") as fh:
        return pickle.load(fh)


def make_norms(D):
    nx = Norm().fit(np.vstack([d["x"] for d in D]))
    ne = Norm().fit(np.vstack([d["edge_attr"] for d in D]))
    return nx, ne


def loss_fn(model, b, cfg):
    z, s, k = model(b)
    y, a = b.y_adj, b.a_bad
    chg = (y != a).float()
    w = torch.where(y > 0, cfg["pos_w"], 1.0) * (1.0 + (cfg["w_chg"] - 1.0) * chg)
    w = w * torch.where((b.near > 0) & (y == 0), cfg.get("w_near", 1.0), 1.0)
    l_adj = (Fnn.binary_cross_entropy_with_logits(s, y, reduction="none") * w).sum() / w.sum()
    wk = torch.where(b.y_keep > 0, 1.0, cfg["w_rm"])
    l_keep = (Fnn.binary_cross_entropy_with_logits(k, b.y_keep, reduction="none") * wk).sum() / wk.sum()
    l_kl = model.kl_loss()
    return l_adj + l_keep + cfg["beta"] * l_kl, (l_adj.item(), l_keep.item(), l_kl.item())


@torch.no_grad()
def predict(model, data, device, bs: int = 64):
    """[(변 확률, 절점 남음 확률)] — 추론은 z = μ (노이즈 없음)."""
    model.eval()
    out = []
    for b in DataLoader(data, batch_size=bs, shuffle=False):
        b = b.to(device)
        z, s, k = model(b)
        ps, pk = torch.sigmoid(s).cpu().numpy(), torch.sigmoid(k).cpu().numpy()
        pb = b.batch[b.pair_index[0]].cpu().numpy()
        nb = b.batch.cpu().numpy()
        for g in range(b.num_graphs):
            out.append((ps[pb == g], pk[nb == g]))
    return out


def edge_report(data, preds) -> dict:
    """변·절점 단위 맞힘 (0.5 기준)."""
    rm_hit = rm_n = new_hit = new_n = false_chg = 0
    node_tp = node_fp = node_fn = 0
    adj_exact = 0
    for d, (ps, pk) in zip(data, preds):
        y, a = d.y_adj.numpy(), d.a_bad.numpy()
        p = ps > 0.5
        rm, new, same = (a == 1) & (y == 0), (a == 0) & (y == 1), (a == y)
        rm_n += rm.sum(); rm_hit += (~p[rm]).sum()
        new_n += new.sum(); new_hit += p[new].sum()
        false_chg += (p[same] != (y[same] == 1)).sum()
        adj_exact += int((p == (y == 1)).all())
        gone_t, gone_p = d.y_keep.numpy() == 0, pk < 0.5
        node_tp += (gone_t & gone_p).sum(); node_fp += (~gone_t & gone_p).sum()
        node_fn += (gone_t & ~gone_p).sum()
    n = len(data)
    return {
        "없어질 변 맞힘": f"{rm_hit}/{rm_n} ({rm_hit / max(rm_n, 1):.1%})",
        "새로 생길 변 맞힘": f"{new_hit}/{new_n} ({new_hit / max(new_n, 1):.1%})",
        "그대로여야 할 쌍을 틀림 (메쉬당)": f"{false_chg / n:.2f}",
        "Good 연결 전체 일치 메쉬": f"{adj_exact}/{n} ({adj_exact / n:.1%})",
        "사라질 절점 재현율": f"{node_tp / max(node_tp + node_fn, 1):.1%}",
        "사라질 절점 정밀도": f"{node_tp / max(node_tp + node_fp, 1):.1%}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="VGAE 학습 (508쌍 전부)")
    ap.add_argument("--tag", default="vgae")
    ap.add_argument("--skip", action="store_true", help="변 디코더에 Bad 연결을 더한다")
    ap.add_argument("--conv", default="gine", choices=CONV_KINDS, help="인코더 메시지 전달 층")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=1e-2, help="KL 가중치")
    ap.add_argument("--pos_w", type=float, default=5.0)
    ap.add_argument("--w_chg", type=float, default=10.0)
    ap.add_argument("--w_rm", type=float, default=10.0)
    ap.add_argument("--w_near", type=float, default=1.0,
                    help="두 칸 이내인데 이어지지 않은 쌍의 가중치 (내적 디코더가 헷갈리는 곳)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = load_data()
    nx, ne = make_norms(D)
    data = [to_data(d, nx, ne) for d in D]
    cfg = {k: getattr(a, k) for k in ("conv", "skip", "hidden", "latent", "layers", "dropout",
                                      "beta", "pos_w", "w_chg", "w_rm", "w_near", "epochs", "lr",
                                      "seed")}
    model = MeshVGAE(data[0].x.shape[1], data[0].edge_attr.shape[1], a.hidden, a.latent,
                     a.layers, a.dropout, a.skip, a.conv).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loader = DataLoader(data, batch_size=a.bs, shuffle=True)

    run = os.path.join(vp.RUNS_DIR, a.tag)
    os.makedirs(run, exist_ok=True)
    print(f"학습: {len(data)}쌍 전부 / {device} / {cfg}")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = np.zeros(4)
        for b in loader:
            b = b.to(device)
            loss, parts = loss_fn(model, b, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += [loss.item(), *parts]
        sched.step()
        if ep % 50 == 0 or ep == 1:
            t = tot / len(loader)
            print(f"  ep {ep:4d}  loss {t[0]:.4f}  (변 {t[1]:.4f} / 절점 {t[2]:.4f} / KL {t[3]:.2f})"
                  f"  {time.time() - t0:.0f}s")

    preds = predict(model, data, device)
    rep = edge_report(data, preds)
    print("\n[학습 데이터에서 잰 값 — 학습에 쓴 데이터라 낙관적]")
    for k, v in rep.items():
        print(f"  {k:28s} {v}")
    torch.save({"state": model.state_dict(), "cfg": cfg,
                "d_in": data[0].x.shape[1], "d_edge": data[0].edge_attr.shape[1],
                "nx": (nx.mu, nx.sd), "ne": (ne.mu, ne.sd)}, os.path.join(run, "model.pt"))
    with open(os.path.join(run, "edge_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"cfg": cfg, **rep}, fh, ensure_ascii=False, indent=2, default=str)
    print(f"저장: {run}")


def load_model(run: str, device):
    ck = torch.load(os.path.join(run, "model.pt"), map_location=device, weights_only=False)
    c = ck["cfg"]
    # conv 가 없는 옛 체크포인트는 GINE 로 학습한 것이다
    m = MeshVGAE(ck["d_in"], ck["d_edge"], c["hidden"], c["latent"], c["layers"],
                 c["dropout"], c["skip"], c.get("conv", "gine")).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, Norm(*ck["nx"]), Norm(*ck["ne"])


if __name__ == "__main__":
    main()

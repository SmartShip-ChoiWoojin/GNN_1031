"""화면이 보여줄 값과 저장된 값이 일치하는지 전 실행에 대해 표본 검사한다.

화면(app.py)은 메쉬를 열 때마다 모델을 돌려 확률을 다시 계산한다. 그 값이 학습 때
저장해 둔 값(교차검증은 oof.csv, 배포 모델은 predictions.csv)과 달라지면, 목록과
화면이 서로 다른 판정을 내놓게 된다. 실제로 그런 적이 있다 - 화면이 fold 배정을
재계산하는 바람에 '그 메쉬를 학습에 쓴 모델' 로 채점해, 틀린 것이 맞다고 떴다.

    python src/check_runs.py

실행마다 무작위 15개를 뽑아 대조한다. 코드를 고친 뒤 한 번씩 돌려보면 좋다.
"""
import os, sys, json, glob, random
import numpy as np
sys.path.insert(0, "src")
import torch, graphs
import dataset as DS
from explain import load_fold_map, load_fold_model, load_full_model, attention_for
from labels import load_label_table
from bdf_io import discover_pairs, load_raw_mesh
from meshrep import MeshRep
import paths

dev = torch.device("cpu")
paired, _, _ = discover_pairs(paths.FE_DIR)
rng = random.Random(0)
_cache = {}

def rep_graph(n, kind, opts):
    key = (n, kind, tuple(sorted(opts.items())))
    if key not in _cache:
        _cache[key] = graphs.build(MeshRep(load_raw_mesh(*paired[n])), kind, **opts)
    return _cache[key]

def check(run, kind):
    res = json.load(open(os.path.join(run, "result.json"), encoding="utf-8"))
    a = res.get("args", {})
    gkind = a.get("graph", "elem")
    opts = ({"corner_edges": not a.get("no_corner_edges"),
             "include_lines": not a.get("no_lines")} if gkind == "elem" else {})

    if kind == "full":
        pf = os.path.join(run, "predictions.csv")
        if not os.path.isfile(pf):
            return f"{os.path.relpath(run):38s} predictions.csv 없음 (화면 오분류 목록 못 씀)"
        import pandas as pd
        df = pd.read_csv(pf).set_index("name")
        names = rng.sample(list(df.index), min(15, len(df)))
        model, norm, _, platt = load_full_model(run, dev)
        model.eval(); bad = []
        with torch.no_grad():
            for n in names:
                lo, _ = attention_for(rep_graph(n, gkind, opts), n, 0, model, norm, dev)
                p = 1/(1+np.exp(-(platt[0]*lo+platt[1])))
                if abs(p - df.loc[n, "prob"]) > 1e-3:
                    bad.append((n, p, df.loc[n, "prob"]))
    else:
        fm = load_fold_map(run)
        if fm is None:
            return f"{os.path.relpath(run):38s} fold_of.csv 없음 (화면이 fold 를 재계산 -> 위험)"
        arr = np.loadtxt(os.path.join(run, "oof.csv"), delimiter=",", skiprows=1)
        nm = open(os.path.join(run, "oof_names.txt"), encoding="utf-8").read().splitlines()
        oof = dict(zip(nm, arr[:, 2]))
        names = rng.sample(nm, min(15, len(nm)))
        bad = []
        for n in names:
            model, norm, _, platt = load_fold_model(run, fm[n], dev)
            model.eval()
            with torch.no_grad():
                lo, _ = attention_for(rep_graph(n, gkind, opts), n, 0, model, norm, dev)
            p = 1/(1+np.exp(-(platt[0]*lo+platt[1])))
            if abs(p - oof[n]) > 1e-3:
                bad.append((n, p, oof[n]))

    if bad:
        msg = f"{os.path.relpath(run):38s} 불일치 {len(bad)}/{len(names)}"
        for n, p, q in bad[:3]:
            msg += f"\n      {n} 화면 {p:.4f} vs 저장 {q:.4f}"
        return msg
    return f"{os.path.relpath(run):38s} OK ({len(names)}개 표본, {gkind})"

print("=== 배포/전체학습 실행 (predictions.csv 대조) ===")
for d in sorted(glob.glob("runs/train/*")):
    if os.path.isfile(os.path.join(d, "full.pt")):
        print("  " + check(d, "full"))
print()
print("=== 교차검증 실행 (oof.csv 대조) ===")
for d in sorted(glob.glob("runs/validate/*") + glob.glob("runs/validate/archive/*")):
    if os.path.isfile(os.path.join(d, "fold1.pt")) and os.path.isfile(os.path.join(d, "result.json")):
        print("  " + check(d, "cv"))

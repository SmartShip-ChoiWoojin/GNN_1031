"""테스트 — 0918 라벨의 '확실 + 비정상' 메쉬를 7개 모델로 재생성한다. 정답은 없다.

테스트 메쉬에는 사람이 고친 Good 이 없다. 그래서 여기서는 채점하지 않고, 재생성 결과를
.bdf/.json 과 표로 남긴다. 좋은지 나쁜지는 사람이 화면(vg_app.py)에서 보고 판정한다.

테스트 대상
    0918 라벨에서 분류=비정상(1), 확실(0) 인 1,291개 중 학습 508쌍에 든 390개를 뺀 901개

학습 메쉬와 겹침 (`overlap_train`)
    패치들이 한 선박에서 잘라낸 조각이라 서로 겹친다. 테스트 메쉬의 쉘 요소 중 학습 508쌍의
    Bad 메쉬에도 (좌표가 같은) 있는 요소의 비율을 적어 둔다. 높으면 모델이 학습에서 본
    자리일 수 있다.

    python src/vg_test.py                                   # 7개 모델 전부
    python src/vg_test.py --models vgae_near_l64 --limit 20

출력
    runs/test/test_meshes.csv            테스트 메쉬 목록 + 원본 정보 + overlap_train
    runs/test/<모델>/regen.csv           메쉬마다 적용한 편집, 요소·삼각형·불량각 수 전후
    out/test/<모델>/<이름>.bdf, .json    재생성 메쉬
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

import vg_paths as vp
import features as F
from meshdiff import pkey
from meshedit import EditableMesh, check

MODELS = ["vgae", "vgae_skip", "vgae_near", "vgae_near_l64",
          "gatv2_l64", "transformer_l64", "nnconv_l64"]
TEST_DIR = os.path.join(vp.RUNS_DIR, "test")
TEST_OUT = os.path.join(vp.OUT_DIR, "test")
TEST_CACHE = os.path.join(vp.CACHE_DIR, "test_node.pkl")


def quality(mesh) -> dict:
    """요소 수, 삼각형 수, 불량각 요소 수 (최소 내각 30° 미만 또는 최대 내각 150° 초과)."""
    n_tri = n_bad = 0
    for _, _, ns in mesh.shells:
        q = F.shell_quality(np.array([mesh.coords[n] for n in ns], dtype=float))
        n_tri += not q.is_quad
        n_bad += F.is_bad_shell(q)
    return {"n_shell": len(mesh.shells), "n_tri": n_tri, "n_bad": n_bad, "n_node": len(mesh.coords)}


def _shell_keys(mesh) -> list:
    return [frozenset(pkey(mesh.coords[n]) for n in s[2]) for s in mesh.shells]


def test_names() -> list[str]:
    from fasoo_drm import setting_fasoo
    from labels import load_label_table

    # 사내 PC 에서는 라벨 엑셀에 FASOO 암호가 걸린다. 다른 진입점(app.py 등)과 같이 읽기 전에 푼다.
    # FASOO 가 없는 PC 에서는 아무 일도 하지 않는다.
    setting_fasoo()
    lab = load_label_table(vp.gp.LABEL_XLSX)
    sure_bad = set(lab[(lab.cls == 1) & (lab.ambig == 0)].name)
    with open(vp.DATA_PATH, "rb") as fh:
        train = {d["name"] for d in pickle.load(fh)}
    return sorted(sure_bad - train)


def build_test_set(limit: int | None = None) -> list[dict]:
    """테스트 메쉬의 입력 그래프 (cache/test_node.pkl 에 저장해 다시 쓴다)."""
    from bdf_io import discover_pairs, load_raw_mesh
    from vg_data import graph_of

    names = test_names()
    if os.path.exists(TEST_CACHE):
        with open(TEST_CACHE, "rb") as fh:
            T = pickle.load(fh)
        if [d["name"] for d in T] == names:
            return T[:limit] if limit else T

    orig, _, _ = discover_pairs(vp.gp.FE_DIR)
    with open(vp.DATA_PATH, "rb") as fh:
        train_keys = set().union(*(_shell_keys(d["before"]) for d in pickle.load(fh)))
    T = []
    t0 = time.time()
    for k, nm in enumerate(names, 1):
        g = graph_of(load_raw_mesh(*orig[nm]))
        keys = _shell_keys(g["before"])
        g["overlap_train"] = float(np.mean([kk in train_keys for kk in keys])) if keys else 0.0
        T.append(g)
        if k % 200 == 0:
            print(f"  그래프 {k}/{len(names)} ({time.time() - t0:.0f}s)")
    with open(TEST_CACHE, "wb") as fh:
        pickle.dump(T, fh)
    return T[:limit] if limit else T


def run_model(tag: str, T: list[dict], device) -> pd.DataFrame:
    from vg_decode import logit_tables, regenerate
    from vg_model import to_data
    from vg_train import load_model, predict

    model, nx, ne = load_model(os.path.join(vp.RUNS_DIR, tag), device)
    preds = predict(model, [to_data(d, nx, ne) for d in T], device)
    out_dir = os.path.join(TEST_OUT, tag)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    t0 = time.time()
    for k, (d, (ps, pk)) in enumerate(zip(T, preds), 1):
        L, Lq = logit_tables(d, ps, pk)
        bad = EditableMesh(d["before"])
        pred, ops, gains = regenerate(bad, d["ids"], L, Lq)
        pred.write_bdf(os.path.join(out_dir, f"{d['name']}.bdf"))
        pred.write_json(os.path.join(out_dir, f"{d['name']}.json"))
        qb, qa = quality(bad), quality(pred)
        rows.append(dict(
            name=d["name"], model=tag, n_ops=len(ops),
            n_collapse=sum(o.startswith("collapse") for o in ops),
            n_merge=sum(o.startswith("merge") for o in ops),
            ops=";".join(ops), gains=";".join(map(str, gains)),
            valid=check(pred, bad, regen=True).ok,
            # VGAE 가 "사라진다" 고 본 절점 수 (q < 0.5) — 편집으로 옮겨졌는지와 별개
            n_gone_pred=int((pk < 0.5).sum()),
            **{f"{k}_before": v for k, v in qb.items()},
            **{f"{k}_after": v for k, v in qa.items()},
        ))
        if k % 200 == 0:
            print(f"  [{tag}] {k}/{len(T)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(TEST_DIR, tag), exist_ok=True)
    df.to_csv(os.path.join(TEST_DIR, tag, "regen.csv"), index=False, encoding="utf-8-sig")
    return df


def summary(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "메쉬": n,
        "편집함": f"{int((df.n_ops > 0).sum())} ({(df.n_ops > 0).mean():.1%})",
        "절점 병합/메쉬": round(df.n_collapse.mean(), 2),
        "삼각형 쌍 병합/메쉬": round(df.n_merge.mean(), 2),
        "삼각형 합 전→후": f"{int(df.n_tri_before.sum())} → {int(df.n_tri_after.sum())}",
        "불량각 요소 합 전→후": f"{int(df.n_bad_before.sum())} → {int(df.n_bad_after.sum())}",
        "검사 위반": int((~df.valid).sum()),
    }


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="테스트 메쉬 재생성 (정답 없음 — 사람이 화면에서 판정)")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    vp.ensure_dirs()
    os.makedirs(TEST_DIR, exist_ok=True)
    T = build_test_set(a.limit)
    meta = pd.DataFrame([dict(name=d["name"], overlap_train=round(d["overlap_train"], 3),
                              **quality(d["before"])) for d in T])
    if not a.limit:
        meta.to_csv(os.path.join(TEST_DIR, "test_meshes.csv"), index=False, encoding="utf-8-sig")
    print(f"테스트 메쉬 {len(T)}개 (0918 확실·비정상 − 학습 508쌍)")
    print(f"  학습 메쉬와 요소가 하나라도 겹침: {int((meta.overlap_train > 0).sum())}, "
          f"절반 넘게 겹침: {int((meta.overlap_train > 0.5).sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = {}
    for tag in a.models:
        S[tag] = summary(run_model(tag, T, device))
    print("\n[테스트 — 정답 없음. 아래는 편집 통계일 뿐 맞고 틀림이 아니다]")
    print(pd.DataFrame(S).to_string())
    if not a.limit:
        # 일부 모델만 다시 돌려도 다른 모델의 요약은 남긴다
        path = os.path.join(TEST_DIR, "summary.csv")
        old = pd.read_csv(path, index_col=0) if os.path.exists(path) else pd.DataFrame()
        new = pd.DataFrame(S)
        old = old.drop(columns=[c for c in new.columns if c in old.columns])
        pd.concat([old, new], axis=1).to_csv(path, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

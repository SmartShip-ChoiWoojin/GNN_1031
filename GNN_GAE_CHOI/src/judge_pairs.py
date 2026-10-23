"""수정 전/후 쌍을 기존 분류기에 통과시켜 본다.

이 프로젝트에서 제일 먼저 확인한 질문이다 — **사람이 고친 결과를 분류기가 정상이라고 보는가?**

  - 본다면: 분류기를 수리기의 심판으로 쓸 수 있다.
  - 안 본다면: 분류기의 '비정상' 과 사람의 '고쳐야 할 것' 이 다른 것을 보고 있다는 뜻이다.

분류기는 **수리 학습에 절대 쓰지 않는다**. 여기서는 심판으로만 쓴다.

    python src/judge_pairs.py --run ../runs/train/deploy_10_14
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

import gae_paths
import dataset as DS
import graphs
import paths as parent_paths
from bdf_io import discover_pairs
from explain import load_full_model
from meshrep import MeshRep
from pairs import all_pairs, discover_good_pairs, load_pair


def pick_deploy_run() -> str:
    """현재 라벨로 학습된 배포본 폴더. 부모 ``predict_all.pick_deploy_run`` 과 같은 규칙."""
    current_label_file = os.path.basename(parent_paths.LABEL_XLSX)
    for run_dir in sorted(glob.glob(os.path.join(parent_paths.RUNS_TRAIN, "*"))):
        if os.path.basename(run_dir).startswith("calib"):
            continue
        result_json = os.path.join(run_dir, "result.json")
        if not os.path.isfile(result_json):
            continue
        if not os.path.isfile(os.path.join(run_dir, "full.pt")):
            continue
        with open(result_json, encoding="utf-8") as file_handle:
            if json.load(file_handle).get("label_xlsx") == current_label_file:
                return run_dir
    raise SystemExit("현재 라벨로 학습된 배포본을 찾지 못했습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description="수정 전/후를 분류기에 통과시킨다")
    parser.add_argument("--run", default=None, help="배포 모델 폴더 (기본: 자동 선택)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, help="앞에서 N쌍만")
    args = parser.parse_args()

    run_dir = args.run
    if run_dir is None:
        run_dir = pick_deploy_run()
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    judge_model, normalizer, classifier_args, platt = load_full_model(run_dir, device)
    graph_options = {}
    if classifier_args.graph == "elem":
        graph_options = {"corner_edges": not classifier_args.no_corner_edges,
                         "include_lines": not classifier_args.no_lines}
    print(f"[judge] {run_dir} · device={device} · graph={classifier_args.graph}")

    def abnormal_probability(mesh) -> float:
        graph_arrays = graphs.build(MeshRep(mesh), classifier_args.graph, **graph_options)
        data = DS.to_pyg(graph_arrays, 0, mesh.name, normalizer).to(device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        logit = float(judge_model(data).item())
        return 1.0 / (1.0 + np.exp(-(platt[0] * logit + platt[1])))

    good_paths = discover_good_pairs()
    original_paths, _, _ = discover_pairs(gae_paths.FE_DIR)
    names = all_pairs()
    if args.limit:
        names = names[:args.limit]

    rows = []
    judge_model.eval()
    with torch.no_grad():
        count = 0
        for name in names:
            count = count + 1
            pair = load_pair(name, good_paths, original_paths)
            rows.append((name, abnormal_probability(pair.before), abnormal_probability(pair.after)))
            if count % 100 == 0:
                print(f"  ... {count}/{len(names)}")

    gae_paths.ensure_dirs()
    out_path = os.path.join(gae_paths.OUT_DIR, "judge_pairs.csv")
    with open(out_path, "w", encoding="utf-8-sig") as file_handle:
        file_handle.write("name,p_before,p_after\n")
        for name, probability_before, probability_after in rows:
            file_handle.write(f"{name},{probability_before:.6f},{probability_after:.6f}\n")

    before_values = []
    after_values = []
    for name, probability_before, probability_after in rows:
        before_values.append(probability_before)
        after_values.append(probability_after)
    before_values = np.array(before_values)
    after_values = np.array(after_values)
    print(f"\n쌍 {len(rows)}개 — 분류기가 보는 '비정상 확률'")
    print(f"  수정 전  평균 {before_values.mean():.3f}  중앙값 {np.median(before_values):.3f}  "
          f"비정상 판정(p>0.5) {np.mean(before_values > 0.5):.1%}")
    print(f"  수정 후  평균 {after_values.mean():.3f}  중앙값 {np.median(after_values):.3f}  "
          f"비정상 판정(p>0.5) {np.mean(after_values > 0.5):.1%}")
    print(f"  p 가 내려간 쌍: {np.mean(after_values < before_values):.1%}   "
          f"평균 변화 {np.mean(after_values - before_values):+.3f}")
    print(f"  '비정상 → 정상' 으로 뒤집힌 쌍: {np.mean((before_values > 0.5) & (after_values <= 0.5)):.1%}")
    print(f"  수정 후에도 비정상으로 남은 쌍:  {np.mean((before_values > 0.5) & (after_values > 0.5)):.1%}")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()

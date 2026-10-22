"""good 참고 학습 — 기존 regen_train.py 를 그대로 돌리되 1단계 GNN 에 보조 과제를 붙인다.

바꾸는 것은 세 가지뿐이고 나머지(분할, 5-fold, 중간 상태, 임계값 고르기, 2단계 GBM, 배포
앙상블)는 기존과 한 줄도 다르지 않다 — 그래야 차이가 보조 과제 때문인지 알 수 있다.
  1. 1단계 채점기 'gnn' → AuxGNNScorer (auxiliary head + auxiliary loss)
  2. 1단계 표본마다 good 과 비교한 절점·요소 라벨을 붙인다
  3. 결과를 ref_good_bdf/runs/regen_<tag>/ 에 쓴다

    # 기준 (보조 과제 끔 — 같은 코드 경로)
    python src/rg_train.py --aux-w 0   --tag base   --loss list --scorer2 gbm --seeds 0,1,2
    # good 참고
    python src/rg_train.py --aux-w 0.5 --tag aux05  --loss list --scorer2 gbm --seeds 0,1,2

``--aux-w`` / ``--aux-pos-weight`` 외의 인자는 regen_train.py 로 그대로 넘어간다.
"""

from __future__ import annotations

import argparse
import sys

import rg_paths

rg_paths.redirect()

import regen_model as RM  # noqa: E402
import regen_train as RT  # noqa: E402
import rg_model  # noqa: E402
from meshdiff import signature  # noqa: E402

_GOOD_SIG: dict = {}
_orig_build_items = RT.build_items
_orig_stage1_item = RT.stage1_item


def _build_items(samples, norm, gopts, ctx, states, verbose=True):
    # 이름 -> good 서명. 중간 상태의 키는 '이름|적용한 병합' 이다
    for s in samples:
        _GOOD_SIG[s.name] = signature(s.pair.after)
    return _orig_build_items(samples, norm, gopts, ctx, states, verbose)


def _stage1_item(name, mesh, norm, gopts, cands=None, truth=None, ctx=False):
    it = _orig_stage1_item(name, mesh, norm, gopts, cands, truth, ctx=ctx)
    sig = _GOOD_SIG.get(name.split("|")[0])
    if sig is not None and truth is not None:   # 학습 표본일 때만 (추론 때는 truth 가 없다)
        it.aux_v, it.aux_e = rg_model.aux_labels(mesh, sig, it.g.vertex_ids, it.g.n_shell,
                                                 gopts.get("include_lines", True))
    return it


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--aux-w", type=float, default=0.5)
    ap.add_argument("--aux-pos-weight", type=float, default=4.0)
    a, rest = ap.parse_known_args()
    rg_model.DEFAULTS.update(aux_w=a.aux_w, aux_pos_weight=a.aux_pos_weight)

    RT.build_items = _build_items
    RT.stage1_item = _stage1_item
    RM.SCORERS["gnn"] = rg_model.AuxGNNScorer   # regen_train 이 '--scorer gnn' 으로 고르는 것
    print(f"[ref_good] 보조 과제 가중치 {a.aux_w} (양성 가중치 {a.aux_pos_weight}) → "
          f"{rg_paths.RUNS_DIR}", flush=True)
    sys.argv = [sys.argv[0]] + rest
    RT.main()


if __name__ == "__main__":
    main()

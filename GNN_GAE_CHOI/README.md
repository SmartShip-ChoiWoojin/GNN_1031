# GNN_GAE_CHOI — GNN_GAE 를 읽기 쉽게 다시 짠 코드

`GNN_GAE/src` 의 코드 전체(30개 파일)를 **같은 기능**으로 다시 짰다. 무엇을 왜 하는지는
원래 문서([../GNN_GAE/README.md](../GNN_GAE/README.md), [../GNN_GAE/메쉬_재생성_설명.md](../GNN_GAE/메쉬_재생성_설명.md))를 그대로 보면 된다.
파일 이름과 실행 명령은 원래와 같다. 다른 것은 **코드를 쓰는 방식**뿐이다.

## 코드 규칙

| 규칙 | 예 |
|---|---|
| 기초 문법만 — 리스트/딕셔너리 컴프리헨션, 제너레이터 식, `lambda`, `x if c else y` 를 쓰지 않는다 | `[f(x) for x in xs]` → `for` 문으로 하나씩 `append` |
| 한 줄짜리 `if` / `for` / `while` / `except` 를 쓰지 않는다 | `if a: b` → 두 줄 |
| 변수·함수 이름을 풀어 쓴다 | `m` → `mesh`, `r, k` → `remove_node, keep_node`, `s1` → `stage1_scorer`, `thr1` → `stage1_threshold` |
| 실수의 합은 내장 `sum()` 으로 한다 | 아래 '알아둘 점' 참고 |

`ast` 로 30개 파일을 전부 검사해 위 문법이 0건임을 확인했다 (HTML 화면 안의 JavaScript 는 Python 이 아니라 제외).

## 폴더

```
GNN_GAE_CHOI/
  src/        코드 (원래 GNN_GAE/src 와 같은 파일 이름) + debug_*.py
  runs/       regen_split.csv (고정 분할), regen_final/ (학습된 최종 모델) — GNN_GAE/runs 에서 복사
  out/        결과 파일 (.bdf, .html, .csv)
  cache/      1단계 후보 캐시 (처음 돌릴 때 만든다)
```

- 사람이 고친 메쉬 데이터는 복사하지 않고 `GNN_GAE/data_folder` 를 그대로 읽는다 (`gae_paths.GOOD_DIR`).
- 원본 메쉬·라벨·심판 분류기는 원래처럼 부모 프로젝트(`GNN_Claude/src`, `GNN_Claude/runs`)의 것을 쓴다.
- `out/` 에 `judge_pairs.csv`, `repairs_cls1.csv`, `merge_pred.csv` 가 없으면 화면·목록이 `GNN_GAE/out` 의 것을 읽는다.

## 실행 (원래와 같은 명령, `GNN_GAE_CHOI` 폴더에서)

```bash
# 재생성
python src/regen_oracle.py                      # 사람 수정 → 연산 대본, 재현 상한
python src/regen_data.py                        # 90/10 분할 요약
python src/regen_train.py --tag final --loss list --scorer2 gbm --seeds 0,1,2   # (~25분)
python src/regen_eval.py --run runs/regen_final            # 테스트 셋 채점
python src/regen_eval.py --run runs/regen_final --split cv # 학습 셋 out-of-fold
python src/viz_steps.py --name SN2592_FWD_Clean_0199       # 편집 과정 단계별 HTML
streamlit run GNN_GAE_CHOI/src/app_regen.py                # 재생성 결과 화면 (GNN_Claude 폴더에서)

# 그 밖
python src/check_edits.py --n 200               # 편집 계층 회귀 테스트 11종
streamlit run GNN_GAE_CHOI/src/app_pairs.py     # Bad ↔ Good 쌍 뷰어
python src/export_pair_list.py / node_baseline.py / judge_pairs.py / task_merge.py / merge_gnn.py / repair_rule.py / scan_repairs.py
```

## 직접 돌려 보는 디버그 스크립트

VSCode 의 실행 및 디버그(F5) 목록에 `GNN_GAE_CHOI ...` 로 들어 있다 (`.vscode/launch.json`).
각 단계에서 F11 로 들어가면 실제 함수로 이어진다.

| 파일 | 따라가는 흐름 | 예 |
|---|---|---|
| `debug_regen.py` | 학습된 모델로 메쉬 하나 재생성. 1단계 반복(후보 → GNN 점수 → 여유 비교 → 병합)을 펼쳐 매 단계의 후보 수·1위·멈춤 점수를 출력 | `python src/debug_regen.py --name SN2592_FWD_Clean_0199 --single` |
| `debug_pair.py` | 쌍 읽기 → 절점 대응 → 정답 병합 그룹 → 병합 적용 → 삼각형 병합 → 사람 결과와 비교 | `python src/debug_pair.py --name SN2592_FWD_Clean_0021` |
| `debug_train.py` | 작은 규모 학습: 표본 → 중간 상태 → 채점 단위 → GNN·GBM 학습 → 임계값 조합별 채점 (파일 저장 없음, 1~2분) | `python src/debug_train.py --n 40 --epochs 2` |

`--single` 은 최종 모델(15개 평균) 대신 첫 모델 하나만 써서 `CandidateNet.forward` 에 한 번씩만 들어가게 한다.

## 원래 이름 → 새 이름 (주요한 것)

| 원래 | 새 이름 | 위치 |
|---|---|---|
| `EditableMesh.P(n)` | `point(node_id)` (+ `points_of(node_ids)`) | meshedit |
| `_compact`, `_relation`, `_edge_count`, `_convex_quad`, `_num` | `compact_nodes`, `node_relation`, `count_edge_use`, `is_convex_quad`, `number_text` | meshedit |
| `EditableMesh._dirty()` | `_mark_changed()` | meshedit |
| `MeshSig` | `MeshSignature` | meshdiff |
| `PairSample.m1` | `mesh_after_stage1` | regen_data |
| `Item` (`.g`, `.cands`, `.y`, `.gfeat`, `.n`) | `ScoringItem` (`.graph`, `.candidates`, `.labels`, `.mesh_features`, `.candidate_count`) | regen_model |
| `RegenModel.s1/s2/thr1/thr2/gopts/norm` | `stage1_scorer/stage2_scorer/stage1_threshold/stage2_threshold/graph_options/normalizer` | regen_model |
| `TrainCfg` (`lr`, `enc_lr_scale`, `ctx`) | `TrainConfig` (`learning_rate`, `encoder_lr_scale`, `use_context`) | regen_model |
| `CandidateNet.enc/head` | `encoder/candidate_head` | regen_model |
| `FeatNorm` (`mu`, `sd`) | `FeatureNormalizer` (`mean`, `std`) | regen_model |
| `RegenResult.m1/p2/c2` | `mesh_after_stage1/stage2_probabilities/stage2_candidates` | regen_model |
| `regen_train._finish` | `finish_stage2` | regen_train |
| `viz_regen._mesh3d/_outline/_lines` | `faces_trace/outline_trace/lines_trace` | viz_regen |
| `check_edits.t_*` | `test_*` | check_edits |

**저장 파일 형식은 그대로다.** `model.pt` / `model.json` 안의 키(`s1`, `thr1`, `cfg`, `lr`, `enc.*` …)는 원래 이름을
유지하고, 불러오고 저장하는 함수에서만 새 이름으로 바꾼다. 그래서 원래 코드로 학습한 `runs/regen_final` 을 그대로
쓰고, 새 코드로 저장한 모델도 원래 코드가 읽을 수 있다 (둘 다 확인).

## 원래 코드와 결과 비교 (2026-10-23)

같은 입력으로 원래 코드와 새 코드를 나란히 돌려 비교했다.

| 비교 | 결과 |
|---|---|
| 테스트 셋 채점 `regen_eval --run runs/regen_final` (52쌍, 15개 모델) | `eval_test.csv` 전 열 동일 — 정확일치 25.0%, 여유(margin)·병합·불변식 경고까지 |
| 학습 셋 out-of-fold `--split cv` (456쌍) | `eval_cv.csv` 전 열 동일 — 정확일치 39.7% |
| 정답 대본 `regen_oracle` (508쌍) | `regen_oracle.csv` 바이트 단위 동일 — 재현 388/508 |
| 학습 `regen_train --scorer gbm --decode oneshot` (GPU 무작위성이 없는 설정) | `cv_pred.csv`, `cv_grid.csv` 동일 — 임계값 0.45/0.05, CV 34.9% |
| 회귀 테스트 `check_edits --n 200` | 출력 동일 (11종 통과, 무작위 편집 횟수까지 같음) |
| `node_baseline` / `repair_rule` / `scan_repairs` / `regen_data` / `task_merge` | 출력·파일 동일 |
| 화면 데이터 `viz_regen` (8쌍), `viz_steps`, `viz_compare` | 그림 데이터·설명 문구 동일 (`viz_steps` 의 '실행되는 코드' 칸은 새 코드를 보여준다) |
| 앱 4개 (`app_regen`, `app_pairs`, `app`, `app_merge`) | 오류 없이 열리고 첫 화면이 같다 |
| `merge_gnn`, `regen_nested`, `debug_train` | 끝까지 돈다 (GPU 학습이라 숫자는 매번 조금씩 다르다 — 원래 코드도 같다) |

## 알아둘 점

- **실수 합은 `sum()` 으로.** Python 3.12 부터 `sum()` 은 오차를 보정하며 더한다. `for` 문으로 하나씩 더하면
  마지막 자리가 달라져, 처음 다시 짰을 때 학습 셋 2쌍의 면적 경고 문구가 0.1 차이 났다 (`18293278.7` vs `.8`).
  면적 합은 목록을 만든 뒤 `sum(목록)` 으로 더한다.
- **GPU 계산은 매번 아주 조금 다르다.** 심판 확률이 소수 6~8째 자리에서 실행마다 흔들린다 (원래 코드끼리도 같다).
  CPU 로 돌리면 원래와 바이트 단위로 같다.
- **`viz_steps.py` 는 소스 코드의 줄을 글자로 찾아 강조한다.** `meshedit.py`·`regen_model.py`·`regen_feats.py` 에서
  강조되는 줄을 고치면 `viz_steps.build_steps` 의 marks 도 같이 고친다.
- 화면 HTML 안의 JavaScript 는 원래 것을 그대로 썼고, 주석만 일부 줄였다.
- `regen_train.py` 에 디버그용 `--limit N` (학습 셋 앞 N쌍만) 을 더했다. 안 주면 원래와 같다.

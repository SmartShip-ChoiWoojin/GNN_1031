# FE 메쉬 정상/비정상 분류 GNN

조선 FE 모델 패치(`.bdf`)가 정상 메쉬인지 비정상 메쉬인지 판정하고,
비정상이면 **어느 요소 때문인지** 지목한다.

## 실행 환경

`SHI_GNN` conda 환경을 쓴다 (base 에는 PyG 가 없다).

```bash
C:/Users/user/anaconda3/envs/SHI_GNN/python.exe -c "import torch,torch_geometric;print(torch.__version__,torch_geometric.__version__)"
```

torch 2.8.0+cu129 / PyG 2.7.0 / RTX 5090.

## 결과

층화 랜덤 5-fold CV, 학습 대상 1262개(정상 758 / 비정상 504). 숫자는 fold 평균 ± 표준편차.

| 실행 | AUC | ACC | F1 | 설정 |
|---|---|---|---|---|
| 기준선 | 0.9603 | 0.8835 | 0.8534 | 집계 피처 37개 + HistGradientBoosting |
| node_gine | 0.9649 ± 0.0101 | 0.9017 ± 0.0077 | 0.8726 | 절점기반 |
| node_gatv2 | 0.9658 ± 0.0096 | 0.9097 ± 0.0089 | 0.8856 | 절점기반 |
| elem_gine | 0.9696 ± 0.0059 | 0.9105 ± 0.0108 | 0.8881 | 요소기반 |
| elem_gatv2 | 0.9753 ± 0.0051 | 0.9152 ± 0.0207 | 0.8900 | 요소기반 |
| **final** | **0.971~0.977** | **0.920~0.924** | **0.903** | 요소기반 + GATv2 + 집계피처 concat 제거 |

**최종 구성**: 요소기반 그래프 + GATv2 + mean/max/attention readout, 집계 피처 concat 없음.
기준선 대비 **ACC +0.040, F1 +0.050, AUC +0.011**.
혼동행렬(out-of-fold): 오탐 40 / 미탐 56.

```bash
python src/train.py --graph elem --conv gatv2 --no-graph-features --tag final
```

### 세 가지 설계 질문에 대한 답

**1) 그래프를 어떻게 표현할 것인가 → 요소기반이 낫다.**
같은 학습 루프로 비교했을 때 요소기반이 절점기반보다 AUC +0.005~0.010, ACC +0.005~0.010 높다.
라벨을 지배하는 신호(삼각형 비율, skew, 면적 편차, 불량 내각)가 전부 요소 속성이기 때문이고,
"어느 요소가 문제인가" 산출물도 요소 노드에서 바로 나온다.

**2) 어떤 피처를 넣을 것인가 → 좌표 불변량 + 위상 + json(ext).**
요소 노드 47차원 / 엣지 15차원. 절대 좌표는 넣지 않는다.
- 1D 요소(CBAR/CBEAM)를 노드로 포함하면 AUC 0.9698 → 0.9753 (+0.0055)
- **집계 피처를 readout 뒤에 concat 하면 오히려 나빠진다** (아래 ablation)

**3) 어떤 모델을 쓸 것인가 → GATv2 + max pooling 포함 readout.**
GATv2 > GINE (AUC +0.006, ACC +0.005). 어텐션 게이트가 요소 지목의 근거가 되므로
성능과 설명력이 같은 방향으로 맞아떨어진다.

### Ablation

| 변경 | AUC | ACC | 판단 |
|---|---|---|---|
| 기준 (elem gatv2, 집계피처 on) | 0.9753 | 0.9152 | — |
| 집계 피처 concat 제거 | 0.9757 ± 0.0008* | 0.9196* | **제거가 낫다** |
| 1D 요소 노드 제거 | 0.9698 | 0.9097 | 1D 포함이 낫다 |
| 꼭짓점 인접 엣지 제거 | 0.9760 | 0.9231 | 차이 없음 (잡음 범위) |
| ext 하드 마스킹 | 0.9732 | 0.9200 | 피처로 주는 쪽이 낫다 |

\* seed 0~3 네 번의 평균. 단일 실행 차이 ±0.003 AUC 는 GPU 비결정성(scatter atomicAdd) 범위라
seed 를 여러 번 돌려야 판단할 수 있다.

**집계 피처 concat 이 해로웠던 이유**는 성능보다 설명력에서 더 분명하다. 집계 피처가 있으면
분류 헤드가 그쪽으로 지름길을 타서 어텐션이 의미를 잃는다:

| 어텐션 품질 | 집계피처 on | 집계피처 off |
|---|---|---|
| 그래프 내 불량요소 판별 AUC | 0.598 | **0.825** |
| 불량 요소 / 나머지 평균 중요도 | 2.7배 | **6.4배** |
| 비정상 예측 시 top-1 적중 | 0.271 | **0.568** |
| 비정상 예측 시 top-3 적중 | — | **0.841** |

### json(외부연결) 도메인 규칙은 학습된다

"ext=true 인 절점 주변의 이상함은 감점하지 말 것"이라는 규칙을 하드 마스킹 없이
피처로만 주었는데, 어텐션이 실제로 그렇게 동작한다:

```
절단면(all_ext) 요소 평균 중요도 0.829x   vs   내부 요소 1.389x   → 비율 0.597
```

단 이 규칙이 영향을 주는 모델은 1262개 중 20개(1.6%)뿐이라 전체 지표는 거의 움직이지 않는다.
정확한 도메인 로직이지만 큰 성능 향상을 기대할 대상은 아니다.

## 사용법

```bash
# 0) 파싱 계층 검증 — 1602개 전 모델 + 회귀 테스트
python src/check_parsing.py

# 1) 기준선 (넘어야 할 선 확정)
python src/baseline.py --ablation

# 2) 학습
python src/train.py --graph elem --conv gatv2 --no-graph-features --tag final
python src/train.py --graph node --conv gine          # 표현 비교
python src/train.py --graph elem --conv gatv2 --split group   # 공간블록 CV (참고)

# 3) 결과 비교표
python src/compare.py

# 4) 요소 지목
python src/explain.py --run runs/final --name SN2592_FWD_Clean_0001
python src/explain.py --run runs/final --worst 6      # 비정상 확신도 상위
python src/explain.py --run runs/final --errors 6     # 오분류 사례
python src/explain.py --run runs/final --audit        # 어텐션 전체 검증
```

## 구조

```
src/
  bdf_io.py        .bdf + .json 파싱 (CBAR/CBEAM 포함). 형식 가정을 assert 로 검증
  meshrep.py       MeshRep — 위상(엣지 incidence, free/nonmanifold, 1D↔엣지) + 기하 캐시
                   + graph_features() 집계 피처 37개
  features.py      기하 품질 지표. 전부 평행이동·회전 불변량
  featspec.py      피처 스펙 + 전역 정규화 (윈저화 표준편차, 유효성 마스크)
  graph_base.py    GraphArrays — 정규화 전 원시 그래프 컨테이너
  graph_elem.py    요소기반 그래프 (노드=요소, 엣지 4종)
  graph_node.py    절점기반 그래프 (노드=GRID, 엣지=쉘 변)
  graphs.py        --graph elem|node 디스패처
  labels.py        라벨 엑셀 로더 (확실/애매/분류어려움 필터)
  data_index.py    사용할 모델 목록 + 공간 그룹 키
  dataset.py       그래프 캐시 + fold 단위 정규화 + PyG 변환
  models.py        MeshGNN (GINE/GATv2 교체) + AttnPool
  train.py         5-fold CV + Platt 캘리브레이션
  baseline.py      tabular 기준선
  compare.py       실행 결과 비교표
  explain.py       어텐션 요소 지목 + 3D 시각화 + 전체 검증
  check_parsing.py 파싱 회귀 테스트
```

## 데이터

| 항목 | 값 |
|---|---|
| `.bdf` / `.json` | 각 1603개 (`SN2592_FWD_Clean_0001`~`_1603`) |
| 카드 종류 | `GRID`, `CQUAD4`, `CTRIA3`, `CBAR`, `CBEAM` 뿐 |
| 평균 규모 | 절점 55.6 / 쉘 47.0 / 1D 38.8 / 변 101.8 |
| 학습 대상 | `cls∈{0,1}` AND `ambig==0` AND json 존재 → **1262개** (758 / 504) |

알려진 데이터 불일치 (코드가 자동 처리):
- `SN2592_FWD_Clean_1603.bdf` — 짝 json 없음 → 제외
- `SN2592_FWD_Clean_4000.json` — bdf 없는 고아 파일, 절점 id 도 1603번과 불일치 → 무시
- 라벨 엑셀은 4000행이지만 bdf 는 1603개까지만 있다

검증된 데이터 사실:
- json 키 집합 == bdf GRID id 집합 (1602개 전부 일치)
- 1D 요소는 100% 쉘 변과 일치 (허공에 뜬 보 0건, 중복 0건)

## 기존 구현(`D:\choiwoojin\3_GNN\SHI`) 대비 수정한 것

1. **그래프별 z-score 정규화 제거** — `SHI/Ver1/bdf2info.py:106`, `bdf2info_v2.py:180`,
   `bdf2info_v3.py:180` 은 피처를 *각 그래프 내부에서* 표준화했다. 그러면 "모든 요소가
   나쁜 메쉬"와 "모든 요소가 좋은 메쉬"가 동일한 입력이 되어 라벨을 결정하는 절대적
   품질 정보가 사라진다. → 데이터셋 전역 통계, 그것도 **학습 fold 에서만** 적합.
2. **절대 좌표 제외** — `bdf2info.py:98` 은 raw x,y,z(x≈243680)를 노드 피처로 썼다.
   패치들이 한 선박을 타일링하므로 위치 암기 = 누출 경로.
3. **1D 요소 포함** — `SHI/Ver1/bdf.py:126` 파서는 CBAR/CBEAM 을 인식조차 하지 않았다
   ("fe model 1D" 데이터셋인데 1D 가 그래프에 없었다).
4. **꼭짓점 인접을 버리지 않음** — `bdf2info_v2.py:203` 의 `if cnt < 2: continue` 는
   절점 1개만 공유하는 쌍을 전부 버렸고 1D 연결도 함께 사라졌다.
5. **CQUAD4 자코비안을 로컬 평면에서 계산** — `bdf2info_v2.py:107-110` 은 전역 X,Y 를
   썼는데 이 데이터셋 쉘의 상당수가 YZ/XZ 평면에 놓여 자코비안이 퇴화(≈0)했다.
6. **평가 프로토콜 분리** — `SHI/Ver1/Train.py:198` 은 보고용 val 로 best 를 골라
   성능이 낙관적이었다. fold 의 test 조각은 early stopping 에 쓰지 않는다.
7. json(ext) 사용, 확실/애매 필터, `pos_weight=1.50`, fold 간 Platt 캘리브레이션 추가.

## 알아둘 점

- **랜덤 분할을 쓴다** (사용자 결정). 1603개 패치는 한 선박 메쉬에서 잘라낸 조각들이라
  절점 id 가 겹친다(한 id 가 평균 3개, 최대 13개 모델에 등장). 공간블록 그룹 CV 로 바꾸면
  기준선이 AUC 0.961→0.953, ACC 0.884→0.876 으로 내려간다. `--split group` 으로 확인 가능.
- **fold 평균과 out-of-fold 종합을 구분**한다. fold 별 모델의 확률 스케일이 달라 그냥 합치면
  AUC 가 낮게 나온다(실측 0.9707 → 0.9522). 내부 val 로 Platt 스케일링을 적합해
  스케일을 맞췄다 — 단조변환이라 fold 별 AUC 는 바뀌지 않는다.
- **GPU 비결정성** 때문에 같은 seed 로도 AUC 가 ±0.003 흔들린다. 그보다 작은 차이는
  seed 여러 번으로 확인해야 한다.
- **오분류 사례 중 일부는 라벨이 논쟁적이다.** 예: `SN2592_FWD_Clean_1559` 는 정상 라벨인데
  최소내각 21.2° 삼각형이 있고 모델이 그 요소를 32배 중요도로 짚으며 비정상(p=0.995)으로
  판정한다. `--errors` 로 이런 사례를 확인할 수 있다.

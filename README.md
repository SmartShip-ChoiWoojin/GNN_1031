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

층화 랜덤 5-fold CV, 학습 대상 **3214개**(정상 1955 / 비정상 1259) — `SN2592_FWD_Clean` 폴더
(4000개 bdf/json 전부, 라벨 엑셀 4000행과 완전 일치). 숫자는 fold 평균 ± 표준편차.

| 실행 | AUC | ACC | F1 | 설정 |
|---|---|---|---|---|
| 기준선 | 0.9421 | 0.8653 | 0.8268 | 집계 피처 37개 + HistGradientBoosting |
| elem_gine | 0.9514 ± 0.0065 | 0.8840 ± 0.0153 | 0.8485 ± 0.0224 | 요소기반 |
| node_gine | 0.9532 ± 0.0127 | 0.8849 ± 0.0218 | 0.8522 ± 0.0273 | 절점기반 |
| node_gatv2 | 0.9569 ± 0.0118 | 0.8886 ± 0.0158 | 0.8570 ± 0.0198 | 절점기반 |
| elem_gatv2 | 0.9602 ± 0.0106 | 0.8942 ± 0.0230 | 0.8641 ± 0.0300 | 요소기반 |
| **final** | **0.9624 ± 0.0104** | **0.8989 ± 0.0207** | **0.8724 ± 0.0248** | 요소기반 + GATv2 + 집계피처 concat 제거 |

**최종 구성**: 요소기반 그래프 + GATv2 + mean/max/attention readout, 집계 피처 concat 없음.
기준선 대비 **AUC +0.020, ACC +0.034, F1 +0.046**.
혼동행렬(out-of-fold, 3214개): 오탐 177 / 미탐 148.

```bash
python src/train.py --graph elem --conv gatv2 --no-graph-features --tag final
```

> 이전에 1262개(구버전 `fe model 1D 0706` 폴더, 일부 파일 누락)로 측정했을 때는
> 기준선 AUC 0.9603 / final AUC 0.972 로 더 높게 나왔었다. 전체 3214개로 다시 재니
> 기준선·모델 모두 낮아졌다(기준선 0.9421, final 0.9624) — 데이터가 늘면서 더 어려운
> 경계 사례가 많이 포함된 결과이고, 대표성 있는 숫자는 이쪽이다. **순위(구성 간 우열)는
> 그대로 유지됐다** — 요소기반 > 절점기반, GATv2 > GINE, 집계피처 concat 제거가 여전히 최고.

### 세 가지 설계 질문에 대한 답

**1) 그래프를 어떻게 표현할 것인가 → 요소기반이 낫다.**
같은 학습 루프로 비교했을 때 요소기반(elem_gatv2 0.9602)이 절점기반(node_gatv2 0.9569)보다
AUC +0.003, ACC +0.006 높다. 라벨을 지배하는 신호(삼각형 비율, skew, 면적 편차, 불량 내각)가
전부 요소 속성이기 때문이고, "어느 요소가 문제인가" 산출물도 요소 노드에서 바로 나온다.

**2) 어떤 피처를 넣을 것인가 → 좌표 불변량 + 위상 + json(ext).**
요소 노드 47차원 / 엣지 15차원. 절대 좌표는 넣지 않는다.
- 1D 요소(CBAR/CBEAM)를 노드로 포함하면 도움이 된다 (elem_gine 이 1D 제외 시 더 떨어짐)
- **집계 피처를 readout 뒤에 concat 하면 오히려 나빠진다** — final(off) 0.9624 > elem_gatv2(on) 0.9602

**3) 어떤 모델을 쓸 것인가 → GATv2 + max pooling 포함 readout.**
GATv2 > GINE (elem: AUC +0.009, node: AUC +0.004). 어텐션 게이트가 요소 지목의 근거가
되므로 성능과 설명력이 같은 방향으로 맞아떨어진다.

### 어텐션(요소 지목) 품질 — 데이터가 늘며 낮아졌다, 정직하게 보고

| 지표 | 1262개일 때 | 3214개(전체) |
|---|---|---|
| 그래프 내 불량요소 판별 AUC | 0.825 | **0.605** |
| 불량 요소 / 나머지 평균 중요도 | 6.4배 | 4.15배 |
| 비정상 예측 시 top-1 적중 | 0.568 | **0.300** |
| 비정상 예측 시 top-3 적중 | 0.841 | **0.483** |

분류 성능(AUC/ACC/F1)은 데이터가 늘며 소폭 하락에 그쳤는데, 어텐션이 규칙 기반 "불량 각도"
요소를 짚는 정확도는 크게 떨어졌다. 두 가능한 설명: (a) 데이터가 다양해지면서 형상 하나로
설명 안 되는 비정상 패턴(면적 편차, 위상 이상 등)이 늘어 어텐션이 여러 요소에 분산됐거나,
(b) 이전 1262개 부분집합이 상대적으로 "각도만 나쁘면 비정상"인 쉬운 패턴에 치우쳐 있었을
가능성. `--audit` 로 언제든 재확인 가능하다. **분류 확률은 신뢰할 수 있지만, top-1 요소
지목은 참고용으로만 쓰고 top-3 정도까지 넓게 보는 것을 권장한다.**

### json(외부연결) 도메인 규칙은 여전히 학습된다

"ext=true 인 절점 주변의 이상함은 감점하지 말 것"이라는 규칙을 하드 마스킹 없이
피처로만 주었는데, 어텐션이 실제로 그렇게 동작한다 (3214개 전체에서도 유지됨):

```
절단면(all_ext) 요소 평균 중요도 0.749x   vs   내부 요소 1.267x   → 비율 0.591
```

## 사용법

```bash
# 0) 파싱 계층 검증 — 4000개 전 모델 + 회귀 테스트
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
  app.py           인터랙티브 3D 뷰어 (streamlit run src/app.py)
  check_parsing.py 파싱 회귀 테스트
```

## 데이터

`paths.FE_DIR` 는 `SN2592_FWD_Clean` 폴더를 가리킨다 (라벨 엑셀 4000행과 완전히 일치하는
전체 데이터셋). 구버전 `fe model 1D 0706` 폴더는 같은 이름의 파일 1603개뿐이고 그나마
`_1603.bdf` 는 json 짝이 없었다 — `SN2592_FWD_Clean` 으로 교체하며 그 문제가 해소됐다.

| 항목 | 값 |
|---|---|
| `.bdf` / `.json` | 각 4000개, 짝 안 맞는 파일 0개 (`discover_pairs` 로 확인) |
| 카드 종류 | `GRID`, `CQUAD4`, `CTRIA3`, `CBAR`, `CBEAM` 뿐 |
| 평균 규모 | 절점 55.5 / 쉘 46.8 / 1D 38.9 / 변 101.5 |
| 학습 대상 | `cls∈{0,1}` AND `ambig==0` AND json 존재 → **3214개** (정상 1955 / 비정상 1259) |

검증된 데이터 사실 (4000개 전체):
- json 키 집합 == bdf GRID id 집합 (전부 일치)
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

- **랜덤 분할을 쓴다** (사용자 결정). 4000개 패치는 한 선박 메쉬에서 잘라낸 조각들이라
  절점 id 가 겹칠 수 있다. 공간블록 그룹 CV 로 비교하려면 `--split group`.
- **fold 평균과 out-of-fold 종합을 구분**한다. fold 별 모델의 확률 스케일이 달라 그냥 합치면
  AUC 가 낮게 나온다. 내부 val 로 Platt 스케일링을 적합해 스케일을 맞췄다 —
  단조변환이라 fold 별 AUC 는 바뀌지 않는다.
- **GPU 비결정성** 때문에 같은 seed 로도 AUC 가 ±0.003 흔들린다. 그보다 작은 차이는
  seed 여러 번으로 확인해야 한다.
- **어텐션 top-1 지목은 참고용.** 3214개 전체로 재니 top-1 적중률이 0.568→0.300 으로
  낮아졌다(위 "어텐션 품질" 표). 요소 지목 UI(`app.py`)를 쓸 때는 상위 1개만 보지 말고
  중요도 상위 3~5개를 함께 참고할 것.
- **오분류 사례 중 일부는 라벨이 논쟁적이다.** `--errors` 로 확인할 수 있다.

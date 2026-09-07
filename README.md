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
| **final** | **0.9605 ± 0.0115** | **0.8970 ± 0.0199** | **0.8689 ± 0.0253** | 요소기반 + GATv2 + 집계피처 concat 제거 + **쉘/1D 인코더 분리** |

**최종 구성**: 요소기반 그래프 + GATv2 + mean/max/attention readout, 집계 피처 concat 없음,
**쉘 노드와 1D 노드를 각각 다른 인코더로 임베딩**(아래 "쉘/1D 인코더 분리" 참고).
기준선 대비 **AUC +0.018, ACC +0.032, F1 +0.042**.
혼동행렬(out-of-fold, 3214개): 오탐 170 / 미탐 161.

```bash
python src/train.py --graph elem --conv gatv2 --no-graph-features --separate-encoder --tag final
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

### 쉘/1D 인코더 분리 — 분류 성능은 그대로, 설명력이 크게 좋아짐

**문제 발견 경위**: `app.py`로 개별 사례를 보다가 `SN2592_FWD_Clean_3127`(실제 정상인데
p=0.985 로 비정상 오판)에서, 아무 이상 없는 CBEAM 하나가 같은 그래프의 어떤 쉘보다도
높은 어텐션(5.02x)을 받는 걸 발견했다. 그 CBEAM 은 길이·차수·ext 전부 정상이고, 인접한
쉘 2개도 완벽한 90°/90° 사각형이었다 — 실제 불량 요소(다른 구역의 CTRIA3 2개)와는
위상적으로 아무 연결도 없었다. **형상 근거가 전혀 없는데 튀는 어텐션**이었다.

**원인**: `graph_elem.py` 는 쉘 전용 피처(`s_*`)와 1D 전용 피처(`l_*`)를 한 벡터에 담고
해당 없는 칸을 0으로 채운다. 그런데 `models.py` 의 노드 인코더(`node_enc`)가 **쉘/1D 공용
`Linear(47, hidden)` 하나**였다 — 이 하나의 가중치 행렬이 "앞칸만 값 있고 뒤칸은 0"(쉘)
"뒤칸만 값 있고 앞칸은 0"(1D)이라는 서로 다른 0-패턴을 동시에 처리해야 했다. 그러다 보니
어텐션 게이트가 "형상이 나쁘다"가 아니라 "타입이 1D 다"라는 지름길에 반응했을 가능성이 있다.

**조치**: 쉘 노드와 1D 노드를 각각 다른 `Linear(47, hidden)` 로 임베딩하도록 분리했다
(`models.py: MeshGNN._encode_nodes`, `--separate-encoder`). 그래프 구조(노드/엣지/순서)는
전혀 안 바뀐다 — 노드를 처음 벡터로 바꾸는 단계만 노드 종류별로 나뉜다.

| 지표 | 분리 전 (공유 인코더) | 분리 후 | 변화 |
|---|---|---|---|
| 분류 AUC (fold 평균) | 0.9624 | 0.9605 | 잡음 범위 |
| 분류 ACC | 0.8989 | 0.8970 | 잡음 범위 |
| 그래프 내 불량요소 판별 AUC (좁은 정의) | 0.605 | **0.816** | **+0.21** |
| 불량 요소 / 나머지 평균 중요도 | 4.15배 | 5.22배 | +1.07배 |
| 비정상 예측 시 top-1 적중 | 0.300 | **0.496** | **+0.20** |
| 비정상 예측 시 top-3 적중 | 0.483 | **0.752** | **+0.27** |
| `SN2592_FWD_Clean_3127` 오탐 확률 | p=0.985 | p=0.759 | 여전히 오답이지만 확신도 급감 |
| 위 사례 top-1 쉘이 실제 불량 요소인가 | 아니오 (0개 후보 표시조차 안 됨) | **예** (CTRIA3 39849, 39852 둘 다 top-6 진입) | 개선 |

**분류 성능은 손해가 없는데(잡음 범위 안) 요소 지목 정확도가 크게 좋아졌다** — 그래서
이걸 새 `final` 기본값으로 삼았다. 참고로 3127번 사례를 분리 후 다시 보면, 여전히 오탐이지만
1위 1D 요소(CBAR)가 이번엔 **실제 불량 쉘에 바로 인접**해 있어 위상적으로 말이 된다
(이전엔 완전히 무관한 구역의 CBEAM 이었다).

```bash
python src/explain.py --run runs/final --name SN2592_FWD_Clean_3127
python src/explain.py --run runs/final --audit
```

**분류 확률은 항상 신뢰할 수 있었지만, 이제 top-1 요소 지목도 어느 정도 믿을 만해졌다.**
그래도 top-3까지 같이 보는 습관은 유지를 권장한다 (`app.py` 는 쉘/1D 표를 분리해서 보여준다).

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
python src/train.py --graph elem --conv gatv2 --no-graph-features --separate-encoder --tag final
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

## 기존 구현(`D:\choiwoojin\3_GNN\SHI_Git`) 대비 수정한 것

**주의 — 아래는 `git log` 로 실제 커밋 이력까지 확인하고 다시 쓴 내용이다.**
처음엔 SHI_Git 의 최초 커밋(`efb6c04 First_Commit`, 2026-07-13)만 보고 비교했는데,
그 **24분 뒤부터** 이미 1D 요소 추가·전역 정규화 같은 개선이 들어가 있었다. 즉 SHI_Git
현재 상태는 "손 안 댄 원본"이 아니라 이미 여러 차례 발전된 버전이다 — 아래 표는 그
현재 상태와 비교한 것이다. SHI_Git 안에도 **그래프 방식이 둘(Ver1=절점기반,
Ver2=요소기반)** 있고 개선 정도가 서로 달라서 나눠서 적는다.

| 항목 | Ver1(절점기반) | Ver2(요소기반) | 우리 프로젝트 |
|---|---|---|---|
| 그래프별(per-graph) 정규화 | 여전히 함 (`bdf2info.py:106-109`, 무조건) | 이미 고침 — train 통계로 global 정규화 저장까지 구현 (`dataset.py`) | 학습 **fold 안에서만** 적합 (fold 간 누출도 차단) |
| 절대 좌표 사용 | 여전히 씀 (`coords + angles_stat + j_stat`) | 안 씀 | 안 씀 |
| 1D(CBAR/CBEAM) 포함 | 여전히 없음 | 이미 포함 (`bdf2info_v2.py:203-218`) | 포함 + **쉘/1D 인코더 분리**(아래) |
| 꼭짓점만 공유하는 쉘-쉘 연결 | 해당없음 | 여전히 버림 (`_should_connect_elements` 가 `len(shared_nodes)>=2` 요구) | 포함 |
| CQUAD4 자코비안 좌표계 | 해당없음 | 여전히 전역 X,Y 사용 (이 데이터셋 쉘 상당수가 YZ/XZ 평면이라 퇴화 가능) | 요소 로컬 평면에 투영 후 계산 |
| 평가 프로토콜 | 랜덤 70/30 split 한 번, 그 val 로 early stopping **과** 최종 성능 보고를 겸함 | 좌동 | 5-fold, **fold 의 test 조각은 손대지 않고** 학습분에서 다시 뗀 내부 val 로만 early stopping |
| json(외부연결) 정보 | 안 씀 | 안 씀 | 씀 — "절단면 근처는 감점 말 것" 규칙이 실제로 학습됨 |
| 확실/애매 라벨 필터 | 라벨 포맷 자체가 다름(구버전 엑셀, 애매 개념 없음) | 좌동 | `확실(0)/애매(1)` 컬럼으로 애매 항목 학습 제외 |
| 요소별 설명(어텐션 지목) | 없음 | 없음 (mean+max pooling, free-edge 는 학습되는 어텐션이 아니라 규칙 기반 하드 게이트) | `GlobalAttention` + UI 로 "어느 요소 때문인지" 표시 |

**정확도 비교(참고용, 조건이 달라 직접 비교는 아님)**: SHI_Git 의 가장 발전된 버전
(Ver2, GATv2, `test_results/test_results.txt`)이 6529개 파일 랜덤 분할 테스트에서
**ACC 0.8917**을 냈다. 우리 최종 모델은 3214개, 5-fold 교차검증 fold 평균으로
**ACC 0.8970**. 큰 차이는 아니다 — **SHI_Git Ver2 는 이미 상당히 잘 만들어진
기준선이었다.** 우리 쪽이 실제로 더 보탠 건 정확도 몇 % 가 아니라:
1) 검증 방식이 더 엄격하다(5-fold, test 조각 완전 격리 — Ver1/Ver2 둘 다 val 로
   early stopping 과 최종 보고를 겸해서 성능이 약간 낙관적으로 나왔을 수 있다),
2) **쉘/1D 인코더 분리**로 설명력이 크게 개선됐다(둘 다 원래 없던 문제였다),
3) json 도메인 규칙과 애매 라벨 처리를 새로 반영했다,
4) "어느 요소가 문제인지" 보여주는 게 SHI_Git 에는 아예 없었다.

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

"""인수인계용 코드 투어 - 메쉬 한 개가 "정상/비정상" 판정까지 가는 길을 따라간다.

VSCode 에서 `# %%` 표시마다 블록 단위로 실행할 수 있다(셀 위의 "Run Cell" 클릭,
또는 Shift+Enter). 위에서부터 한 칸씩 내려가며 출력되는 숫자를 눈으로 확인하는 것이
목적이다. 한 번에 통째로 실행해도 된다:

    python src/walkthrough.py

각 단계가 실제로 쓰는 파일을 [코드] 로 적어 뒀다. 그 파일을 열어 같이 보면 좋다.

전체 흐름
    .bdf + .json  ->  MeshRep  ->  그래프  ->  정규화  ->  GNN  ->  확률 + 요소별 중요도
                      (연결관계)   (노드=요소)
"""

# %%
# ──────────────────────────────────────────────────────────────────────
# 0단계. 준비
# ──────────────────────────────────────────────────────────────────────
# [코드] src/paths.py - 경로는 전부 여기서만 정한다. 다른 파일에 경로를 하드코딩하지
#        않는 것이 이 프로젝트의 규칙이다.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import paths
from fasoo_drm import setting_fasoo

setting_fasoo()   # 사내 PC 는 엑셀에 암호가 걸린다. 파일을 하나라도 열기 전에 한 번.

print("데이터 폴더 :", paths.FE_DIR)
print("라벨 엑셀   :", os.path.basename(paths.LABEL_XLSX))
print("실행 폴더   :", paths.RUNS_DIR)

# %%
# ──────────────────────────────────────────────────────────────────────
# 1단계. 메쉬 하나 고르기
# ──────────────────────────────────────────────────────────────────────
# [코드] src/bdf_io.py
#
# 모델 하나 = 파일 두 개.
#   .bdf  : 절점 좌표와 요소 정의 (Nastran 형식)
#   .json : 절점마다 "모델 바깥과 연결되어 있는가" (true/false)
#
# json 이 왜 필요한가: 바깥과 이어진 자리는 잘린 단면이라 모양이 이상해 보이는 것이
# 정상이다. 그걸 불량으로 판정하면 안 되므로 모델에 그 사실을 알려준다.
from bdf_io import discover_pairs

paired, bdf_only, json_only = discover_pairs(paths.FE_DIR)
print(f"짝이 맞는 모델 {len(paired)}개  (json 없는 bdf {len(bdf_only)}, 고아 json {len(json_only)})")

NAME = "SN2592_FWD_Clean_0001"      # <- 여기를 바꿔가며 다른 메쉬도 보라
bdf_path, json_path = paired[NAME]
print("bdf :", os.path.basename(bdf_path))
print("json:", os.path.basename(json_path))

# %%
# ──────────────────────────────────────────────────────────────────────
# 2단계. 원시 파싱 - 파일에 실제로 뭐가 들어있나
# ──────────────────────────────────────────────────────────────────────
from bdf_io import load_raw_mesh, read_bdf

coords, shells_raw, lines_raw = read_bdf(bdf_path)
print(f"절점(GRID)       : {len(coords)}개")
print(f"쉘(CQUAD4/CTRIA3): {len(shells_raw)}개")
print(f"1D(CBAR/CBEAM)   : {len(lines_raw)}개")

k = list(coords)[0]
print(f"\n절점 예시: id={k}  좌표={coords[k]}")
print(f"쉘  예시: {shells_raw[0]}")
if lines_raw:
    print(f"1D  예시: {lines_raw[0]}")

raw = load_raw_mesh(bdf_path, json_path)
n_ext = sum(1 for v in raw.ext.values() if v)
print(f"\n외부연결 절점: {n_ext}/{len(raw.ext)}개 (json 에서 true 인 것)")

# %%
# ──────────────────────────────────────────────────────────────────────
# 3단계. MeshRep - 연결 관계를 미리 계산해 둔다
# ──────────────────────────────────────────────────────────────────────
# [코드] src/meshrep.py
#
# 원시 파싱 결과만으로는 "어느 요소가 어느 요소와 붙어 있나"를 알 수 없다.
# MeshRep 이 그 관계를 계산해 둔다. 이것이 곧 그래프의 뼈대가 된다.
from meshrep import MeshRep

rep = MeshRep(raw)
print(f"쉘 {len(rep.shells)}개 / 1D {len(rep.lines)}개 / 절점 {len(rep.node_ids)}개")

sh = rep.shells[0]
q = sh.quality
print(f"\n쉘 하나: eid={sh.eid} type={sh.etype} 절점={sh.nodes}")
print(f"  내각 {q.angle_min:.1f}~{q.angle_max:.1f}도, aspect {q.aspect:.2f}, skew {q.skew:.1f}도")

# 변(edge)마다 그 변을 공유하는 쉘 목록 - 그래프 연결의 핵심 재료
free = sum(1 for s in rep.edge_shells.values() if len(s) == 1)
manifold = sum(1 for s in rep.edge_shells.values() if len(s) == 2)
nonman = sum(1 for s in rep.edge_shells.values() if len(s) > 2)
print(f"\n변 {len(rep.edge_shells)}개 중  경계변(쉘 1개) {free} / 정상(2개) {manifold} / 이상(3개+) {nonman}")

# %%
# ──────────────────────────────────────────────────────────────────────
# 4단계. 품질 지표 - "이 요소가 이상한가"를 숫자로
# ──────────────────────────────────────────────────────────────────────
# [코드] src/features.py - 전부 순수 numpy 기하 계산이다.
#
# 좌표의 절대값은 쓰지 않고 각도와 비율만 쓴다. 배의 어느 위치에서 잘라낸 조각이든
# 같은 기준으로 보기 위해서다. 좌표를 그대로 넣으면 "앞쪽 조각은 비정상이 많더라"
# 같은 엉뚱한 규칙을 외워버린다.
import features as F

P = np.array([rep.raw.coords[n] for n in sh.nodes])
q2 = F.shell_quality(P)
print(f"직접 계산한 품질: 내각 {q2.angle_min:.1f}~{q2.angle_max:.1f}, aspect {q2.aspect:.2f}")
print(f"규칙 기반 불량 판정: {F.is_bad_shell(q2)}   (최소내각<30 또는 최대내각>150)")

bad = [s for s in rep.shells if F.is_bad_shell(s.quality)]
print(f"\n이 메쉬에서 규칙상 불량인 쉘: {len(bad)}개")
for s in bad[:5]:
    print(f"  {s.etype} {s.eid}: 내각 {s.quality.angle_min:.1f}~{s.quality.angle_max:.1f}")

# %%
# ──────────────────────────────────────────────────────────────────────
# 5단계. 그래프로 변환 - GNN 의 출발점
# ──────────────────────────────────────────────────────────────────────
# [코드] src/graph_elem.py (요소 기반), src/graph_base.py (자료구조)
#
# 가장 중요한 설계 선택: "노드 = 요소" 다. 절점이 아니라 요소를 노드로 둔다.
# 판정 대상이 요소(이 쉘이 찌그러졌나)이므로 요소에 피처를 붙이는 쪽이 자연스럽고,
# "어느 요소가 문제인지" 지목하기도 쉽다.
#
# 엣지(연결)는 4종류:
#   0 쉘-쉘 변 공유 / 1 쉘-쉘 꼭짓점만 공유 / 2 1D-쉘 / 3 1D-1D
import graph_elem
import graphs

ga = graphs.build(rep, "elem", corner_edges=True, include_lines=True)
print(f"그래프 노드 {ga.n_node}개 (쉘 {len(rep.shells)} + 1D {len(rep.lines)})")
print(f"그래프 엣지 {ga.n_edge}개 (양방향이라 실제 관계 수의 2배)")
print(f"노드 피처 {ga.x.shape[1]}차원 / 엣지 피처 {ga.edge_attr.shape[1]}차원")
print(f"그래프 전체 피처 {ga.graph_x.shape[0]}차원")

rel_names = ("rel_shell_edge", "rel_shell_corner", "rel_line_shell", "rel_line_line")
for nm, label in zip(rel_names, ("쉘-쉘 변", "쉘-쉘 꼭짓점", "1D-쉘", "1D-1D")):
    c = graph_elem.EDGE_SPEC.names.index(nm)
    print(f"  관계 {label:12s}: {int(ga.edge_attr[:, c].sum())}개")

# %%
# ──────────────────────────────────────────────────────────────────────
# 6단계. 피처에 뭐가 들어있나
# ──────────────────────────────────────────────────────────────────────
# s_ 로 시작 = 쉘에만 해당, l_ 로 시작 = 1D 에만 해당.
# 1D 요소에는 "면적"이 없다. 그런 칸은 x_valid 가 False 이고, 정규화 통계에서
# 빠진 뒤 0 으로 채워진다. 이 처리를 안 하면 1D 들의 0 이 쉘의 평균을 끌어내린다.
names = graph_elem.NODE_SPEC.names
print(f"노드 피처 {len(names)}개:")
for i in range(0, len(names), 6):
    print("   ", ", ".join(names[i:i + 6]))

print(f"\n0번 노드(쉘)의 유효 칸 수 : {int(ga.x_valid[0].sum())}/{len(names)}")
if rep.lines:
    li = len(rep.shells)
    print(f"{li}번 노드(1D)의 유효 칸 수: {int(ga.x_valid[li].sum())}/{len(names)}")

# %%
# ──────────────────────────────────────────────────────────────────────
# 7단계. 학습된 모델 불러오기
# ──────────────────────────────────────────────────────────────────────
# [코드] src/explain.py (불러오기), src/models.py (모델 구조)
#
# 체크포인트에는 가중치뿐 아니라 정규화 통계도 같이 들어있다. 학습 때와 똑같은
# 기준으로 피처를 변환해야 하기 때문이다. 이걸 빠뜨리면 에러 없이 조용히 틀린다.
import torch

from explain import attention_for, load_full_model, relative_importance, top_elements

RUN = os.path.join(paths.RUNS_DIR, "deploy")
device = torch.device("cpu")
model, norm, margs, platt = load_full_model(RUN, device)
print(f"실행 폴더 : {RUN}")
print(f"그래프={margs.graph} conv={margs.conv} hidden={margs.hidden} layers={margs.layers}")
print(f"파라미터  : {sum(p.numel() for p in model.parameters()):,}개")
print(f"확률 보정 : p = sigmoid({platt[0]:.3f} x logit + {platt[1]:.3f})")

# %%
# ──────────────────────────────────────────────────────────────────────
# 8단계. 예측해 보기
# ──────────────────────────────────────────────────────────────────────
# [코드] src/dataset.py 의 to_pyg - 정규화를 적용해 PyG 형식으로 바꾼다.
#
# logit 은 모델이 뱉는 원시 점수다(음수면 정상 쪽). 그대로는 확률이 아니라서
# sigmoid 와 Platt 보정을 거쳐 0~1 로 만든다.
logit, alpha = attention_for(ga, NAME, 0, model, norm, device)
prob = 1 / (1 + np.exp(-(platt[0] * logit + platt[1])))
print(f"원시 점수(logit) : {logit:.4f}")
print(f"비정상 확률      : {prob:.4f}  ->  {'비정상' if prob > 0.5 else '정상'}")

from labels import load_label_table

tbl = load_label_table(paths.LABEL_XLSX).set_index("name")
r = tbl.loc[NAME]
tag = {0: "정상", 1: "비정상", 2: "분류어려움"}[int(r.cls)]
print(f"엑셀 라벨        : {tag} ({'애매' if int(r.ambig) else '확실'})")

# %%
# ──────────────────────────────────────────────────────────────────────
# 9단계. "어느 요소 때문인가" - 설명력
# ──────────────────────────────────────────────────────────────────────
# 모델은 요소마다 가중치(alpha)를 매겨 합산한 뒤 전체를 판정한다. 그 가중치가 곧
# "이 요소를 얼마나 중요하게 봤는가"다. alpha 는 합이 1 이라 그래프 크기에 따라
# 값이 달라지므로, 노드 수를 곱해 "평균 대비 몇 배"로 바꿔 본다. 1.0 = 평균.
imp = relative_importance(alpha)
print(f"중요도 평균 {imp.mean():.2f} (정의상 1.0), 최대 {imp.max():.2f}배")
print("\n상위 쉘 요소:")
for line in top_elements(rep, ga, imp, k=5):
    print("  ", line)

# %%
# ──────────────────────────────────────────────────────────────────────
# 10단계. 라벨 - 무엇을 학습에 쓰는가
# ──────────────────────────────────────────────────────────────────────
# [코드] src/labels.py (엑셀 읽기), src/data_index.py (학습 대상 선정)
#
# 규칙: "확실"(애매 아님) 이면서 "분류어려움"이 아닌 것만 학습에 쓴다.
# 사람이 확신하지 못한 라벨을 정답이라고 가르치면 그 불확실함까지 배우기 때문이다.
t = load_label_table(paths.LABEL_XLSX)
print(t.pivot_table(index="ambig", columns="cls", aggfunc="size", fill_value=0))
print("\n행: 0=확실 1=애매 / 열: 0=정상 1=비정상 2=분류어려움")
print(f"학습 대상(확실 & 분류 가능): {int(t['trainable'].sum())}개")
print(f"제외: {len(t) - int(t['trainable'].sum())}개 (이들이 곧 test 집합이 된다)")

# %%
# ──────────────────────────────────────────────────────────────────────
# 11단계. 전체 데이터셋은 어떻게 도나
# ──────────────────────────────────────────────────────────────────────
# [코드] src/dataset.py (캐시/정규화), src/train.py (학습 루프)
#
# 4000개를 매번 파싱하고 변환하면 느리다. 그래서 그래프를 pickle 로 캐시한다.
# 주의: 캐시에는 '그래프만' 담는다. 라벨은 매번 엑셀에서 새로 읽는다. 예전에는
# 라벨까지 캐시에 넣었다가, 엑셀을 갈아끼워도 옛 라벨이 조용히 쓰이는 문제가 있었다.
#
# 아래는 수십 초 걸리므로 필요할 때만 주석을 풀어라.
#
# import dataset as DS
# names, labels, arrays = DS.build_cache("elem", opts={"corner_edges": True,
#                                                      "include_lines": True})
# print(len(names), "개 로드됨")

print("""
학습 (배포 모델과 같은 설정):
  python src/train.py --graph elem --conv gatv2 --no-graph-features
      --separate-encoder --full --val-frac 0 --epochs 44
      --platt-from runs/full0918/full.pt --tag deploy

예측 캐시 만들기 (앱이 오분류 목록을 즉시 띄우려면 필요):
  python src/predict_all.py --run runs/deploy

화면으로 보기:
  streamlit run src/app.py
""")

# %%
# ──────────────────────────────────────────────────────────────────────
# 직접 해보기
# ──────────────────────────────────────────────────────────────────────
# 1) 1단계의 NAME 을 비정상 메쉬로 바꿔 8~9단계를 다시 돌려보라.
#    확률과 상위 중요도 요소가 어떻게 달라지는가?
# 2) 5단계에서 corner_edges=False 로 하면 엣지가 몇 개 줄어드는가?
#    include_lines=False 면 노드 수는 어떻게 되는가?
# 3) 9단계의 상위 중요도 요소가 4단계의 "규칙상 불량" 요소와 겹치는가?
#    겹치지 않는다면 모델은 무엇을 보고 있는 것일까?
# 4) src/app.py 의 predict() 함수를 찾아, 이 흐름이 화면으로 어떻게 이어지는지 보라.

"""인터랙티브 3D 메쉬 뷰어 — bdf 파일을 불러와 회전/줌/패닝하며 요소별 어텐션을 본다.

    streamlit run src/app.py

기능
  - 샘플 폴더에서 모델 선택, 예측 사례(FP/FN/경계선 근처 정답) 목록에서 선택,
    애매/분류어려움(학습 제외) 목록에서 선택, 또는 .bdf(+.json) 직접 업로드
  - 사이드바에 현재 선택한 실행의 5-fold 성능(AUC/ACC/F1)을 상시 표시
  - 오분류 목록은 predict_all.py 가 배포 모델로 미리 계산해 둔 predictions.csv 를 써서
    재추론 없이 즉시 FP/FN 을 걸러내고, 확신도(0.5 에서 먼 정도) 순으로 정렬한다
  - Plotly 3D — 마우스로 자유롭게 회전·확대·이동, 요소에 마우스 올리면 상세정보
  - 초기 화면은 채색 없이 메쉬 형상만 명확히 표시. "중요도 강조"를 켜면
    임계값 이상인 요소만 단일 색 계열(빨강)로, 나머지는 중립 회색으로 표시
  - 학습된 GNN(runs/*)으로 정상/비정상 확률 + 요소별 어텐션(문제 요소 지목)
  - "그래프 구조" 탭 — 메쉬가 GNN 입력 그래프로 어떻게 변환됐는지
    (요소기반이면 요소 중심점=노드, 관계 종류별 색의 엣지) 시각화
  - 배포 모델(full.pt) 하나로 모든 메쉬를 예측한다. 예전처럼 파일 이름에 따라
    다른 모델을 고르지 않으므로 어느 PC 에서 열어도 같은 결과가 나온다
    (full.pt 가 없는 옛 5-fold 실행을 고르면 예전 방식으로 동작한다)
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import tempfile
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset as DS
import features as F
import graphs
import paths
from bdf_io import discover_pairs, load_raw_mesh, read_bdf
from explain import (
    attention_for,
    fold_assignment,
    load_fold_model,
    load_full_model,
    load_result,
    relative_importance,
)
from fasoo_drm import setting_fasoo
from labels import load_label_table
from meshrep import MeshRep

# 라벨 엑셀 등 fasoo 로 암호화된 파일을 읽어야 하므로, 파일을 하나라도 열기 전에
# 프로그램 시작 시 한 번만 해제한다. (setting_fasoo 안에 중복 호출 가드가 있어서
# Streamlit 이 위젯 조작마다 이 스크립트를 다시 실행해도 실제 해제는 한 번만 일어난다.)
setting_fasoo()

st.set_page_config(page_title="메쉬 정상/비정상 뷰어", layout="wide")

#: 실행마다 라벨 엑셀을 갈아끼우므로, 원래 기본값을 따로 기억해 둔다.
paths_label_default = paths.LABEL_XLSX

# ── 색상 상수 ────────────────────────────────────────────────────────────
NEUTRAL_SHELL = "#c7ced9"     # 강조 안 된 쉘 — 중립 회색, 형상이 잘 보이도록
NEUTRAL_LINE = "#1d4ed8"      # 강조 안 된 1D 요소 — 쉘과 구분되는 진한 파랑
RED_LIGHT = "#fcae91"         # 강조 요소 — 빨강 계열 (밝음=경계, 진함=핵심 원인)
RED_DARK = "#99000d"
BAD_OUTLINE = "#111827"       # 규칙 기반 불량각도 요소 테두리 — 검정 (빨강 강조와 구분)
EXT_MARKER = "#06b6d4"


# ──────────────────────────────────────────────────────────────────────────
# 캐시된 로딩
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def available_runs() -> list[str]:
    out = []
    # 배포용(runs/train) 만 보여준다. 검증용 5-fold 실행을 사용자가 고를 수 있으면
    # 또 "왜 결과가 다르냐" 가 반복된다.
    cur = os.path.basename(paths.LABEL_XLSX)
    for d in sorted(glob.glob(os.path.join(paths.RUNS_TRAIN, "*"))):
        name = os.path.basename(d)
        if name.startswith("calib"):
            continue   # 최적 epoch 와 보정 계수를 구하려고 만든 중간 산출물. 배포본이 아니다.
        rj = os.path.join(d, "result.json")
        if not os.path.isfile(rj):
            continue
        with open(rj, "r", encoding="utf-8") as fh:
            lab = json.load(fh).get("label_xlsx")
        if lab and lab != cur:
            continue   # 옛 라벨로 돌린 비교용 모델. 배포 후보가 아니다 (개발자 모드에서만)
        out.append(name)
    return out


@st.cache_data
def available_samples() -> list[str]:
    paired, _, _ = discover_pairs(paths.FE_DIR)
    return sorted(paired)


@st.cache_data
def validation_runs() -> dict[str, str]:
    """{표시이름: 경로} - runs/validate 와 그 안의 archive.

    개발자 모드에서만 쓴다. 여기 모델은 성능을 재려고 만든 것이라 배포본이 아니다.
    """
    out: dict[str, str] = {}
    shown = set(available_runs())
    for d in sorted(glob.glob(os.path.join(paths.RUNS_TRAIN, "*"))):
        name = os.path.basename(d)
        if name in shown or not os.path.isfile(os.path.join(d, "result.json")):
            continue
        out["train/" + name] = d          # 옛 라벨로 돌린 홀드아웃 모델, 보정용 실행 등
    for root, prefix in ((paths.RUNS_VALIDATE, ""), (paths.RUNS_ARCHIVE, "archive/")):
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "result.json")):
                out[prefix + os.path.basename(d)] = d
    return out


@st.cache_data
def validation_table() -> pd.DataFrame:
    """실행들의 성능 비교표.

    평가 방식이 두 가지라 열로 구분한다.
      홀드아웃   - 학습대상의 10%를 빼두고 거기서만 채점. 배포 모델이 쓰는 방식
      교차검증   - 5-fold. 모든 메쉬가 '자기를 학습에 안 쓴 fold' 에게 채점받는다
    둘은 평가 범위가 달라서 수치를 직접 비교하면 안 된다. 라벨 버전도 같이 본다.
    """
    rows = []
    srcs = dict(validation_runs())
    srcs.update({k: os.path.join(paths.RUNS_TRAIN, k) for k in available_runs()})

    for name, d in sorted(srcs.items()):
        if os.path.basename(d).startswith("calib"):
            continue   # 최적 epoch 를 찾으려고 돌린 중간 실행. 학습 방식이 달라 비교 대상이 아니다
        rj = os.path.join(d, "result.json")
        if not os.path.isfile(rj):
            continue
        with open(rj, "r", encoding="utf-8") as fh:
            res = json.load(fh)
        n = res.get("n_samples")
        lab = res.get("label_xlsx")
        if not lab:
            lab = {3223: "0918 (추정)", 3214: "0721 (추정)"}.get(n, "알 수 없음")
        lab = lab.replace("fe model 1D index ", "").replace(".xlsx", "")

        if res.get("mode") == "full":
            n_hold = res.get("n_holdout") or 0
            if not n_hold:
                continue            # 홀드아웃이 없으면 정직한 수치가 없다. 표에 올리지 않는다
            m, how, scope = res.get("metrics", {}), "홀드아웃", n_hold
        else:
            m, how, scope = res.get("fold_mean", {}), "교차검증", n

        a = res.get("args", {})
        opt = [t for t, on in (("인코더분리", a.get("separate_encoder")),
                               ("집계피처off", a.get("no_graph_features")),
                               ("final-epoch", a.get("final_epoch"))) if on]
        rows.append({
            "실행": name, "라벨": lab, "평가 방식": how, "평가 대상": scope,
            "AUC": round(m.get("auc", float("nan")), 4),
            "ACC": round(m.get("acc", float("nan")), 4),
            "F1": round(m.get("f1", float("nan")), 4),
            "설정": f"{a.get('graph','')}/{a.get('conv','')} " + ",".join(opt),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["라벨", "평가 방식", "AUC"], ascending=[False, True, False])
    return df


@st.cache_data
def label_lookup(label_path: str) -> dict[str, tuple[int, int]]:
    df = load_label_table(label_path)
    return {r.name: (int(r.cls), int(r.ambig)) for r in df.itertuples()}


LABEL_TAGS = {0: "정상", 1: "비정상", 2: "분류어려움"}


@st.cache_data
def run_label_path(run_dir: str) -> str:
    """그 실행이 평가 기준으로 삼은 라벨 엑셀의 경로.

    옛 라벨로 돌린 실행을 열었을 때 화면의 '실제 라벨' 이 현재 엑셀 기준이면,
    그 사이 분류가 바뀐 메쉬에서 "맞았다는데 왜 틀렸다고 나오지?" 가 된다.
    실행마다 그때의 기준을 따라간다.
    """
    rj = os.path.join(run_dir, "result.json")
    lab = None
    if os.path.isfile(rj):
        with open(rj, "r", encoding="utf-8") as fh:
            res = json.load(fh)
        lab = res.get("label_xlsx")
        if not lab:   # 옛 실행에는 기록이 없다. 학습대상 개수로 되짚는다.
            lab = {3223: "fe model 1D index 0918.xlsx",
                   3214: "fe model 1D index 0721.xlsx"}.get(res.get("n_samples"))
    if lab:
        cand = os.path.join(paths.ROOT, lab)
        if os.path.isfile(cand):
            return cand
    return paths.LABEL_XLSX


@st.cache_data
def split_samples(run_dir: str, label_path: str) -> tuple[list[str], list[str], list[str]]:
    """폴더의 모델을 세 묶음으로 나눈다.

    (학습에 쓴 것, 평가 전용 홀드아웃, 애매·분류어려움)

    - 학습에 쓴 것 : 모델이 정답을 본 메쉬. 잘 맞는 게 당연해서 성능 근거가 못 된다
    - 홀드아웃     : 확실한 라벨인데 일부러 빼둔 몫. **성능은 여기서 나온다**
    - 애매·분류어려움 : 사람이 확신 못 해 학습에서 제외한 것. 정답 자체가 불확실하다
    """
    paired = set(available_samples())
    hold = set(holdout_names(run_dir))
    tbl = load_label_table(label_path)

    # 5-fold 실행은 그 실행이 돌 때의 학습대상 명단(oof_names.txt)을 그대로 쓴다.
    # 지금 라벨 엑셀로 다시 뽑으면 개수가 안 맞는다 - 0721 로 돌린 실행은 3214개인데
    # 현재 엑셀(0918)로는 3223개가 나온다.
    own = run_sample_names(run_dir)

    fit, ho, amb = [], [], []
    for r in tbl.itertuples():
        if r.name not in paired:
            continue
        if own is not None:
            if r.name in own:
                fit.append(r.name)
            elif not r.trainable:
                amb.append(r.name)
            continue
        if not r.trainable:
            amb.append(r.name)
        elif r.name in hold:
            ho.append(r.name)
        else:
            fit.append(r.name)
    return sorted(fit), sorted(ho), sorted(amb)


@st.cache_data
def run_sample_names(run_dir: str) -> set[str] | None:
    """그 실행이 실제로 학습대상으로 삼았던 이름들. 교차검증 실행에만 있다."""
    path = os.path.join(run_dir, "oof_names.txt")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return set(fh.read().splitlines())


@st.cache_data
def holdout_names(run_dir: str) -> list[str]:
    """이 실행이 평가 전용으로 빼둔 메쉬 이름. 없으면 빈 목록(= 전부 학습에 씀)."""
    path = os.path.join(run_dir, "holdout_names.txt")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().splitlines()


def label_tag(name: str, label_path: str) -> str:
    """이름 옆에 괄호로 붙일 라벨 표시 - (정상) (비정상) (애매·비정상) (분류어려움)."""
    lab = label_lookup(label_path).get(name)
    if lab is None:
        return "(라벨 없음)"
    cls, ambig = lab
    if cls == 2:
        return "(분류어려움·애매)" if ambig else "(분류어려움)"
    return f"(애매·{LABEL_TAGS[cls]})" if ambig else f"({LABEL_TAGS[cls]})"


def pick_display(disp: list[str], names: list[str], key: str, title: str) -> str | None:
    """보여줄 문구(disp)로 고르게 하고, 대응하는 이름(names)을 돌려준다.

    목록마다 검색·선택을 따로 구현하면 화면마다 동작이 미묘하게 달라지므로 한 곳에 둔다.
    """
    query = st.text_input(
        "검색 (부분 문자열, 예: 0250)", key=key,
        help="입력한 글자를 이름 어디에든 포함하는 항목만 드롭다운에 남습니다.",
    )
    pairs = [(d, n) for d, n in zip(disp, names)
             if not query or query.lower() in d.lower()]
    if not pairs:
        st.warning(f"'{query}' 를 포함하는 항목이 없습니다.")
        return None
    pick = st.selectbox(f"{title} ({len(pairs)}개)", [d for d, _ in pairs], index=0)
    return dict(pairs)[pick]


def pick_by_label(names: list[str], key: str, title: str, label_path: str) -> str | None:
    """이름 옆에 라벨을 붙여 고르게 하고 고른 이름을 돌려준다."""
    return pick_display([f"{n}  {label_tag(n, label_path)}" for n in names],
                        names, key, title)


@st.cache_data
def load_run_metrics(run_dir: str) -> dict | None:
    """train.py 가 저장한 result.json 에서 fold 평균 성능을 읽는다."""
    path = os.path.join(run_dir, "result.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        res = json.load(fh)
    return {
        "mode": res.get("mode", "cv"),
        "fold_mean": res.get("fold_mean"),
        "n_samples": res.get("n_samples"),
        "n_fit": res.get("n_fit"),
        "n_holdout": res.get("n_holdout") or 0,
        "metrics": res.get("metrics") or {},
        "label_xlsx": res.get("label_xlsx"),
    }


@st.cache_resource(show_spinner=False)
def get_run_context(run_dir: str):
    """(args, names, labels, arrays, fold_of, opts) — 학습 캐시와 fold 배정을 한 번만 계산."""
    res = load_result(run_dir)
    args = SimpleNamespace(**res["args"])
    opts = {}
    if args.graph == "elem":
        opts = {"corner_edges": not args.no_corner_edges, "include_lines": not args.no_lines}
    names, labels, arrays = DS.build_cache(args.graph, opts=opts, verbose=False)
    # 배포용 단일 모델이 있으면 fold 배정을 아예 계산하지 않는다. 그 계산은 학습대상
    # 목록의 순서에 의존해서, 목록이 조금만 달라져도 담당 모델이 바뀌던 원인이었다.
    fold_of = None if has_full_model(run_dir) else fold_assignment(names, labels, args)
    return args, names, labels, arrays, fold_of, opts


def has_full_model(run_dir: str) -> bool:
    """전 학습대상으로 학습한 배포용 단일 모델이 있는가."""
    return os.path.isfile(os.path.join(run_dir, "full.pt"))


@st.cache_resource(show_spinner=False)
def get_fold_model(run_dir: str, fold: int, device_str: str):
    device = torch.device(device_str)
    return load_fold_model(run_dir, fold, device)


@st.cache_resource(show_spinner=False)
def get_full_model(run_dir: str, device_str: str):
    device = torch.device(device_str)
    return load_full_model(run_dir, device)


# ──────────────────────────────────────────────────────────────────────────
# 파일 로딩 (업로드 or 샘플 폴더)
# ──────────────────────────────────────────────────────────────────────────
def parse_from_bytes(bdf_bytes: bytes, json_bytes: bytes | None, name_hint: str) -> MeshRep:
    """업로드된 바이트를 임시 파일로 내려쓴 뒤 MeshRep 으로 파싱한다."""
    with tempfile.TemporaryDirectory() as td:
        bdf_path = os.path.join(td, f"{name_hint}.bdf")
        with open(bdf_path, "wb") as fh:
            fh.write(bdf_bytes)

        if json_bytes is not None:
            json_path = os.path.join(td, f"{name_hint}.json")
            with open(json_path, "wb") as fh:
                fh.write(json_bytes)
        else:
            # json 이 없으면 전 절점을 '외부연결 없음'으로 간주하고 알린다
            coords, _, _ = read_bdf(bdf_path)
            json_path = os.path.join(td, f"{name_hint}.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump({str(k): False for k in coords}, fh)

        return MeshRep(load_raw_mesh(bdf_path, json_path))


@st.cache_resource(show_spinner=False)
def parse_sample(name: str) -> MeshRep:
    bdf_path = os.path.join(paths.FE_DIR, f"{name}.bdf")
    json_path = os.path.join(paths.FE_DIR, f"{name}.json")
    return MeshRep(load_raw_mesh(bdf_path, json_path))


@st.cache_data(show_spinner=False)
def load_predictions(run_dir: str) -> pd.DataFrame:
    """배포 모델이 라벨 있는 전 메쉬를 예측해 둔 표(predict_all.py 가 만든다).

    화면 어디서나 같은 모델, 같은 확률을 쓴다. 예전에는 오분류 목록만 5-fold 의
    값을 썼는데, 그러면 한 메쉬에 점수가 두 개 생겨서 목록과 화면이 서로 다른
    판정을 내놓는 문제가 있었다.
    """
    path = os.path.join(run_dir, "predictions.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        # 5-fold 실행에는 predictions.csv 가 없다. 대신 oof.csv 가 있는데 이쪽이
        # 오히려 정직하다 - 모든 메쉬가 '자기를 학습에 쓰지 않은 fold' 에게 채점받은
        # 값이라, 10% 홀드아웃보다 평가 범위가 넓다.
        oof_p = os.path.join(run_dir, "oof.csv")
        names_p = os.path.join(run_dir, "oof_names.txt")
        if not (os.path.isfile(oof_p) and os.path.isfile(names_p)):
            return pd.DataFrame(
                columns=["name", "cls", "ambig", "trainable", "prob", "pred", "kind", "conf"]
            )
        arr = np.loadtxt(oof_p, delimiter=",", skiprows=1)
        with open(names_p, "r", encoding="utf-8") as fh:
            nm = fh.read().splitlines()
        df = pd.DataFrame({
            "name": nm,
            "cls": arr[:, 1].astype(int),
            "prob": arr[:, 2],
            "ambig": 0,
            "trainable": 1,
        })
    df["pred"] = (df["prob"] > 0.5).astype(int)
    df["kind"] = np.select(
        [(df["cls"] == 0) & (df["pred"] == 1), (df["cls"] == 1) & (df["pred"] == 0),
         df["cls"] == 2],
        ["FP", "FN", "분류어려움"], default="정답",
    )
    df["conf"] = (df["prob"] - 0.5).abs()
    return df


# ──────────────────────────────────────────────────────────────────────────
# 예측
# ──────────────────────────────────────────────────────────────────────────
def predict(rep: MeshRep, run_dir: str, device_str: str) -> dict:
    """예측 확률 + 요소별 중요도.

    배포 모델(full.pt)이 있으면 어떤 파일이든 그 모델 하나로만 예측한다 - 파일 이름이나
    학습셋 포함 여부에 따라 계산이 달라지지 않으므로 어느 PC 에서 열어도 같은 값이 나온다.
    """
    args, names, labels, arrays, fold_of, opts = get_run_context(run_dir)
    device = torch.device(device_str)

    ga = graphs.build(rep, args.graph, **opts)

    detail: dict = {}
    if has_full_model(run_dir):
        # 어떤 파일이든, 이름이 무엇이든 항상 같은 모델 하나로 판정한다.
        model, norm, _, platt = get_full_model(run_dir, device_str)
        logit, alpha = attention_for(ga, rep.name, 0, model, norm, device)
        prob = 1.0 / (1.0 + np.exp(-(platt[0] * logit + platt[1])))
        in_train = rep.name in names
        truth = labels[names.index(rep.name)] if in_train else None
        mode = "전체 학습 단일 모델 (어느 PC에서 열어도 동일한 결과)"
        detail = dict(kind="full", logit=float(logit), platt=platt, in_train=in_train)
    elif rep.name in names:
        idx = names.index(rep.name)
        fold = int(fold_of[idx])
        model, norm, _, platt = get_fold_model(run_dir, fold, device_str)
        logit, alpha = attention_for(ga, rep.name, labels[idx], model, norm, device)
        prob = 1.0 / (1.0 + np.exp(-(platt[0] * logit + platt[1])))
        mode = f"fold {fold} 모델 (이 요소를 학습에 쓰지 않음)"
        truth = labels[idx]
        detail = dict(kind="single", fold=fold, logit=float(logit), platt=platt)
    else:
        probs, alphas, logits = [], [], []
        for f in range(1, args.folds + 1):
            model, norm, _, platt = get_fold_model(run_dir, f, device_str)
            logit, alpha = attention_for(ga, rep.name, 0, model, norm, device)
            p = 1.0 / (1.0 + np.exp(-(platt[0] * logit + platt[1])))
            probs.append(p)
            logits.append(float(logit))
            alphas.append(alpha)
        prob = float(np.mean(probs))
        alpha = np.mean(np.stack(alphas), axis=0)
        mode = f"{args.folds}-fold 앙상블 평균 (학습셋에 없는 파일)"
        truth = None
        detail = dict(kind="ensemble", fold_probs=probs, logits=logits)

    imp = relative_importance(alpha)
    return dict(args=args, ga=ga, prob=prob, imp=imp, mode=mode, truth=truth, detail=detail)


def per_element_importance(rep: MeshRep, imp: np.ndarray, graph_kind: str) -> np.ndarray:
    """쉘 요소별 중요도(길이 n_shell). 절점기반 그래프면 인접 절점 중요도 평균으로 환산."""
    n_shell = len(rep.shells)
    if graph_kind == "elem":
        return imp[:n_shell]
    return np.array([
        np.mean([imp[rep.node_index[n]] for n in sh.nodes]) for sh in rep.shells
    ])


# ──────────────────────────────────────────────────────────────────────────
# 3D 메쉬 뷰
# ──────────────────────────────────────────────────────────────────────────
def build_mesh_figure(
    rep: MeshRep,
    ga,
    imp: np.ndarray,
    graph_kind: str,
    opacity: float,
    show_lines: bool,
    show_ext: bool,
    show_bad_outline: bool,
    show_shell_edges: bool,
    highlight: bool,
    threshold: float,
) -> go.Figure:
    """메쉬 3D 뷰.

    ``highlight=False`` 면 전 요소를 중립 회색으로 칠해 형상만 보여준다(초기 화면).
    ``highlight=True`` 면 중요도가 ``threshold`` 배 이상인 요소만 빨강 계열로,
    나머지는 회색 그대로 남겨 "이 요소 때문에 비정상"인 곳만 눈에 띄게 한다.
    색은 항상 빨강 한 계열(밝은→진한)만 쓴다 — 무지개색 컬러스케일은 쓰지 않는다.
    """
    fig = go.Figure()
    n_shell = len(rep.shells)
    shell_imp = per_element_importance(rep, imp, graph_kind)
    vmax = max(float(shell_imp.max()), threshold + 0.5, 1e-6)

    # ── 쉘 요소: 요소마다 정점을 따로 둬서 요소 단위로 균일하게 칠한다(faceted look) ──
    xs, ys, zs, ti, tj, tk, intensity, hover = [], [], [], [], [], [], [], []
    vtx = 0
    for si, sh in enumerate(rep.shells):
        pts = [rep.raw.coords[n] for n in sh.nodes]
        base = vtx
        for p in pts:
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        vtx += len(pts)

        tris = [(base, base + 1, base + 2)]
        if len(pts) == 4:
            tris.append((base, base + 2, base + 3))
        for a, b, c in tris:
            ti.append(a); tj.append(b); tk.append(c)

        val = float(shell_imp[si])
        q = sh.quality
        badge = "불량" if F.is_bad_shell(q) else "정상"
        txt = (
            f"{sh.etype} {sh.eid}<br>중요도 {val:.2f}x<br>"
            f"내각 {q.angle_min:.1f}°~{q.angle_max:.1f}° ({badge})<br>"
            f"aspect {q.aspect:.2f}  skew {q.skew:.1f}°"
        )
        for _ in pts:
            intensity.append(val if highlight else 0.0)
            hover.append(txt)

    mesh_kwargs = dict(
        x=xs, y=ys, z=zs, i=ti, j=tj, k=tk,
        flatshading=True, opacity=opacity,
        text=hover, hoverinfo="text", name="쉘 요소",
    )
    if highlight:
        gray_frac = float(np.clip(threshold / vmax, 0.02, 0.97))
        colorscale = [
            [0.0, NEUTRAL_SHELL], [gray_frac, NEUTRAL_SHELL],
            [min(gray_frac + 1e-3, 1.0), RED_LIGHT], [1.0, RED_DARK],
        ]
        fig.add_trace(go.Mesh3d(
            **mesh_kwargs, intensity=intensity, intensitymode="vertex",
            colorscale=colorscale, cmin=0, cmax=vmax, showscale=True,
            colorbar=dict(
                title="중요도(x평균)",
                tickvals=[0, threshold, vmax],
                ticktext=["평균 이하", f"{threshold:.1f}x (임계값)", f"{vmax:.1f}x"],
            ),
        ))
    else:
        fig.add_trace(go.Mesh3d(**mesh_kwargs, color=NEUTRAL_SHELL))

    # ── 쉘 경계선 — 옅은 회색. 인접한 쉘끼리 색이 비슷하면 뭉개져 보이므로,
    # 요소 하나하나의 경계를 항상 그어서 구분되게 한다 (진하게는 하지 않는다).
    if show_shell_edges:
        sx, sy, sz = [], [], []
        for e in rep.edge_shells:
            a, b = tuple(e)
            pa, pb = rep.raw.coords[a], rep.raw.coords[b]
            sx += [pa[0], pb[0], None]; sy += [pa[1], pb[1], None]; sz += [pa[2], pb[2], None]
        if sx:
            fig.add_trace(go.Scatter3d(
                x=sx, y=sy, z=sz, mode="lines",
                line=dict(color="#6b7280", width=1.5), opacity=0.55,
                name="쉘 경계", hoverinfo="skip",
            ))

    # ── 규칙 기반 불량각도 요소 테두리 (검정 — 빨강 강조와 구분되는 별도 신호) ──
    if show_bad_outline:
        ex, ey, ez = [], [], []
        for sh in rep.shells:
            if not F.is_bad_shell(sh.quality):
                continue
            pts = [rep.raw.coords[n] for n in sh.nodes] + [rep.raw.coords[sh.nodes[0]]]
            for p in pts:
                ex.append(p[0]); ey.append(p[1]); ez.append(p[2])
            ex.append(None); ey.append(None); ez.append(None)
        if ex:
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez, mode="lines",
                line=dict(color=BAD_OUTLINE, width=5),
                name="규칙 기반 불량각도 요소", hoverinfo="skip",
            ))

    # ── 1D 요소 (CBAR/CBEAM) — 선을 굵게, 끝점 마커를 찍어 연결을 명확히 ──
    if show_lines and rep.lines:
        has_line_imp = highlight and graph_kind == "elem" and ga.n_node > n_shell
        mx, my, mz, mcolor, mtext = [], [], [], [], []
        for li, ln in enumerate(rep.lines):
            a, b = rep.raw.coords[ln.nodes[0]], rep.raw.coords[ln.nodes[1]]
            if has_line_imp:
                v = float(imp[n_shell + li])
                if v >= threshold:
                    frac = float(np.clip((v - threshold) / max(vmax - threshold, 1e-6), 0, 1))
                    color = _blend(RED_LIGHT, RED_DARK, frac)
                    width = 5 + 4 * frac
                else:
                    color, width = NEUTRAL_LINE, 4
                txt = f"{ln.etype} {ln.eid}<br>길이 {ln.length:.1f}<br>중요도 {v:.2f}x"
            else:
                color, width = NEUTRAL_LINE, 4
                txt = f"{ln.etype} {ln.eid}<br>길이 {ln.length:.1f}"
            fig.add_trace(go.Scatter3d(
                x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
                mode="lines", line=dict(color=color, width=width),
                text=[txt, txt], hoverinfo="text", showlegend=False,
            ))
            for p in (a, b):
                mx.append(p[0]); my.append(p[1]); mz.append(p[2])
            mcolor += [color, color]
            mtext += [txt, txt]
        fig.add_trace(go.Scatter3d(
            x=mx, y=my, z=mz, mode="markers",
            marker=dict(size=3.5, color=mcolor),
            text=mtext, hoverinfo="text",
            name=f"1D 요소 (CBAR/CBEAM, {len(rep.lines)}개)",
        ))

    # ── 외부연결(ext) 절점 ────────────────────────────────────────────
    if show_ext:
        pts = [rep.raw.coords[n] for n in rep.node_ids if rep.ext.get(n, False)]
        if pts:
            P = np.array(pts)
            fig.add_trace(go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                marker=dict(size=4, color=EXT_MARKER, symbol="circle-open", line=dict(width=2)),
                name="외부연결 절점 (json ext=true)",
            ))

    _apply_layout(fig)
    return fig


def _blend(c1: str, c2: str, t: float) -> str:
    """두 hex 색을 t(0~1)로 선형 보간 — 강조 요소 안에서도 진하기 차등."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t); g = int(g1 + (g2 - g1) * t); b = int(b1 + (b2 - b1) * t)
    return f"rgb({r},{g},{b})"


def _apply_layout(fig: go.Figure) -> None:
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        height=720,
    )


def top_elements_table(
    rep: MeshRep, imp: np.ndarray, graph_kind: str, k: int = 10, only_shells: bool = False
):
    """중요도 상위 요소 표.

    ``only_shells=True`` 면 1D 요소(CBAR/CBEAM)를 순위에서 아예 뺀다. 1D 는 형상
    결함(각도/찌그러짐) 판정 대상이 아니라 보강재 위상 정보일 뿐이라 "판정" 칸이
    항상 "-"인데, 그런데도 어텐션 점수가 커서 종합 순위에서 1등을 차지하면
    "왜 비정상인지" 보여주려는 목적과 어긋난다. 쉘/1D 를 분리해서 보여주는 쪽을
    기본으로 쓴다 (아래 UI 에서 두 표로 나눠 호출).
    """
    n_shell = len(rep.shells)
    rows = []
    pool = np.arange(n_shell) if (graph_kind == "elem" and only_shells) else np.arange(len(imp))
    order = pool[np.argsort(-imp[pool])][:k]
    for i in order:
        if graph_kind == "elem" and i >= n_shell:
            ln = rep.lines[i - n_shell]
            rows.append({"중요도": round(float(imp[i]), 2), "종류": ln.etype,
                        "id": ln.eid, "내각/길이": f"길이 {ln.length:.1f}",
                        "판정": "-"})
        elif graph_kind == "elem":
            sh = rep.shells[i]
            q = sh.quality
            rows.append({"중요도": round(float(imp[i]), 2), "종류": sh.etype,
                        "id": sh.eid,
                        "내각/길이": f"{q.angle_min:.1f}°~{q.angle_max:.1f}°",
                        "판정": "불량" if F.is_bad_shell(q) else "정상"})
        else:
            nid = rep.node_ids[i]
            rows.append({"중요도": round(float(imp[i]), 2), "종류": "GRID",
                        "id": nid, "내각/길이": "-",
                        "판정": "ext" if rep.ext.get(nid, False) else "내부"})
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 그래프 구조 뷰 — 메쉬가 GNN 입력 그래프로 어떻게 바뀌었는지
# ──────────────────────────────────────────────────────────────────────────
def elem_graph_edges(rep: MeshRep) -> dict[str, list[tuple[int, int]]]:
    """요소기반 그래프의 관계 4종을 (그래프 노드 인덱스) 쌍 목록으로.

    graph_elem.build() 와 같은 인덱싱: 쉘은 0..n_shell-1, 1D 는 n_shell..n_shell+n_line-1.
    학습에 실제로 쓰이는 정규화된 피처와는 무관하게, 순수히 "무엇과 무엇이 연결됐는지"만
    다시 계산한다 — 시각화 전용이라 graph_elem 내부 구현에 의존하지 않는다.
    """
    n_shell = len(rep.shells)
    edges: dict[str, list[tuple[int, int]]] = {
        "shell_edge": [], "shell_corner": [], "line_shell": [], "line_line": [],
    }

    seen = set()
    for e, shells in rep.edge_shells.items():
        if len(shells) < 2:
            continue
        for a, b in combinations(sorted(set(shells)), 2):
            if (a, b) not in seen:
                seen.add((a, b))
                edges["shell_edge"].append((a, b))

    for nid, shells in rep.node_shells.items():
        for a, b in combinations(sorted(set(shells)), 2):
            if (a, b) not in seen:
                seen.add((a, b))
                edges["shell_corner"].append((a, b))

    for li, ln in enumerate(rep.lines):
        for si in rep.edge_shells.get(ln.edge, ()):
            edges["line_shell"].append((n_shell + li, si))

    seen_ll = set()
    for nid, lines in rep.node_lines.items():
        for a, b in combinations(sorted(set(lines)), 2):
            key = (n_shell + a, n_shell + b)
            if key not in seen_ll:
                seen_ll.add(key)
                edges["line_line"].append(key)

    return edges


REL_STYLE = {
    "shell_edge": ("쉘-쉘 변 공유", "#2563eb"),
    "shell_corner": ("쉘-쉘 꼭짓점 공유", "#9ca3af"),
    "line_shell": ("1D-쉘 연결", "#f97316"),
    "line_line": ("1D-1D 연결", "#dc2626"),
}


def build_graph_figure_elem(rep: MeshRep, show_rel: set[str]) -> go.Figure:
    """노드=요소(쉘/1D), 엣지=관계 4종. 원본 메쉬는 옅게 배경으로 깔아 대응관계를 보여준다."""
    fig = go.Figure()
    n_shell = len(rep.shells)

    # 배경: 원본 메쉬 윤곽 (옅게)
    ex, ey, ez = [], [], []
    for sh in rep.shells:
        pts = [rep.raw.coords[n] for n in sh.nodes] + [rep.raw.coords[sh.nodes[0]]]
        for p in pts:
            ex.append(p[0]); ey.append(p[1]); ez.append(p[2])
        ex.append(None); ey.append(None); ez.append(None)
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="#e5e7eb", width=1.5),
        name="원본 메쉬 (배경)", hoverinfo="skip",
    ))

    centroids = np.array(
        [s.centroid for s in rep.shells] + [l.centroid for l in rep.lines]
    ) if rep.lines else np.array([s.centroid for s in rep.shells])

    sc = centroids[:n_shell]
    fig.add_trace(go.Scatter3d(
        x=sc[:, 0], y=sc[:, 1], z=sc[:, 2], mode="markers",
        marker=dict(size=5, color="#1f2937"),
        text=[f"쉘 {s.etype} {s.eid}" for s in rep.shells], hoverinfo="text",
        name=f"그래프 노드 · 쉘 요소 ({n_shell}개)",
    ))
    if rep.lines:
        lc = centroids[n_shell:]
        fig.add_trace(go.Scatter3d(
            x=lc[:, 0], y=lc[:, 1], z=lc[:, 2], mode="markers",
            marker=dict(size=5, color="#f97316", symbol="diamond"),
            text=[f"1D {l.etype} {l.eid}" for l in rep.lines], hoverinfo="text",
            name=f"그래프 노드 · 1D 요소 ({len(rep.lines)}개)",
        ))

    edges = elem_graph_edges(rep)
    for key, (label, color) in REL_STYLE.items():
        if key not in show_rel:
            continue
        pairs = edges[key]
        if not pairs:
            continue
        lx, ly, lz = [], [], []
        for a, b in pairs:
            pa, pb = centroids[a], centroids[b]
            lx += [pa[0], pb[0], None]
            ly += [pa[1], pb[1], None]
            lz += [pa[2], pb[2], None]
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="lines",
            line=dict(color=color, width=2.5), opacity=0.75, hoverinfo="skip",
            name=f"엣지 · {label} ({len(pairs)}개)",
        ))

    _apply_layout(fig)
    return fig


def build_graph_figure_node(rep: MeshRep) -> go.Figure:
    """절점기반 그래프 — 노드=GRID 절점, 엣지=쉘 요소의 변."""
    fig = go.Figure()
    P = rep.points
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
        marker=dict(size=4, color="#1f2937"),
        text=[f"GRID {n}" for n in rep.node_ids], hoverinfo="text",
        name=f"그래프 노드 · 절점 ({len(P)}개)",
    ))
    ex, ey, ez = [], [], []
    for e in rep.edge_shells:
        a, b = tuple(e)
        pa, pb = rep.raw.coords[a], rep.raw.coords[b]
        ex += [pa[0], pb[0], None]; ey += [pa[1], pb[1], None]; ez += [pa[2], pb[2], None]
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="#2563eb", width=2.5), opacity=0.75, hoverinfo="skip",
        name=f"엣지 · 쉘 변 공유 ({len(rep.edge_shells)}개)",
    ))
    _apply_layout(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────
st.title("FE 메쉬 정상/비정상 뷰어")

runs = available_runs()
if not runs:
    st.error(
        f"배포용 모델이 없습니다. `python scripts/train_deploy.py` 로 먼저 학습하세요."
        f"\n\n찾은 위치: {paths.RUNS_TRAIN}"
    )
    st.stop()

with st.sidebar:
    st.header("설정")

    # 개발자 모드 - 성능 측정용(runs/validate) 실행까지 꺼내 본다. 기본은 꺼짐.
    # 배포본을 쓰는 사람에게 측정용 모델이 보이면 또 "왜 결과가 다르냐"가 생긴다.
    dev_mode = st.checkbox(
        "개발자 모드", value=False,
        help="검증용 5-fold 실행의 성능 비교표를 보고, 그 실행을 직접 불러올 수 있습니다. "
             "검증용 모델은 배포본이 아닙니다.",
    )

    # 배포용 단일 모델(full.pt)이 있는 실행을 우선 고른다 - 그래야 어느 PC 에서
    # 열어도 같은 판정이 나온다. 없으면 예전 5-fold 실행으로 되돌아간다.
    _full_runs = [r for r in runs if has_full_model(os.path.join(paths.RUNS_TRAIN, r))]
    if _full_runs:
        default_run = _full_runs[0]
    else:
        default_run = runs[0]

    run_paths = {r: os.path.join(paths.RUNS_TRAIN, r) for r in runs}
    if dev_mode:
        run_paths.update({f"[검증] {k}": v for k, v in validation_runs().items()})

    choices = list(run_paths)
    run_name = st.selectbox("모델 실행", choices, index=choices.index(default_run))
    run_dir = run_paths[run_name]

    # 화면 전체가 '그 실행이 평가 기준으로 삼은 라벨' 을 쓰도록 맞춘다. 아래에서
    # 읽는 그래프 캐시의 라벨(get_run_context)도 이 값을 따라간다.
    LBL = run_label_path(run_dir)
    paths.LABEL_XLSX = LBL
    if os.path.basename(LBL) != os.path.basename(paths_label_default):
        st.info(f"이 실행의 기준 라벨 **{os.path.basename(LBL)}** 로 화면을 표시합니다.")
    if run_name.startswith("[검증]"):
        if has_full_model(run_dir):
            st.error(
                "**배포본이 아닙니다.** 비교용으로 따로 학습한 단일 모델입니다 "
                "(현재 배포 라벨과 다른 기준으로 학습된 경우가 많습니다)."
            )
        else:
            st.error(
                "**검증용 실행입니다 - 배포본이 아닙니다.** 성능을 재려고 만든 fold "
                "모델이라, 메쉬마다 어느 fold 가 판정할지를 그때그때 계산합니다. 같은 "
                "메쉬라도 환경에 따라 다른 답이 나올 수 있습니다."
            )

    run_metrics = load_run_metrics(run_dir)
    if run_metrics and run_metrics["mode"] == "full":
        st.caption(
            (
                f"**단일 모델** · 학습 {run_metrics['n_fit']}개"
                + (f" / 평가 전용 {run_metrics['n_holdout']}개"
                   if run_metrics["n_holdout"] else "")
                + ". fold 분할이 없어 어느 PC에서 열어도 같은 결과가 나옵니다."
                + (f"  홀드아웃 성능 — AUC **{run_metrics['metrics'].get('auc', 0):.4f}** · "
                   f"ACC **{run_metrics['metrics'].get('acc', 0):.4f}** · "
                   f"F1 **{run_metrics['metrics'].get('f1', 0):.4f}**"
                   if run_metrics["n_holdout"] else "")
            )
        )
    elif run_metrics and run_metrics["fold_mean"]:
        fm = run_metrics["fold_mean"]
        st.caption(
            f"이 실행의 5-fold 성능 - AUC **{fm['auc']:.4f}** · "
            f"ACC **{fm['acc']:.4f}** · F1 **{fm['f1']:.4f}** "
            f"(학습대상 {run_metrics['n_samples']}개, fold 평균)"
        )
    else:
        st.caption("이 실행의 result.json 을 찾을 수 없습니다.")

    device_str = st.radio("연산 장치", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
                          index=0)

    st.divider()
    st.subheader("파일 선택")
    fit_names, hold_names, amb_names = split_samples(run_dir, LBL)
    # 5-fold 실행에서는 학습대상 전체가 이미 정직하게 평가되어 있다(OOF). 배포 모델처럼
    # "학습에 썼으니 못 믿는다"가 아니다 - 라벨 버전별 평가 결과를 볼 때 이 경로를 쓴다.
    is_cv_run = not has_full_model(run_dir)
    SRC_FIT = (f"학습대상 · 교차검증 평가 ({len(fit_names)}개)" if is_cv_run
               else f"학습에 쓴 메쉬 ({len(fit_names)}개)")
    SRC_HOLD = f"평가 전용 홀드아웃 ({len(hold_names)}개)"
    SRC_AMB = f"애매·분류어려움 ({len(amb_names)}개)"
    SRC_ERR = "오분류·경계선 사례 (FP/FN)"
    SRC_UPLOAD = "직접 업로드"

    sources = [SRC_FIT] + ([SRC_HOLD] if hold_names else []) + [SRC_AMB, SRC_ERR, SRC_UPLOAD]
    source = st.radio(
        "불러올 방식", sources,
        help="학습에 쓴 메쉬=모델이 정답을 이미 본 것. "
             "평가 전용 홀드아웃=확실한 라벨인데 일부러 빼둔 몫으로, 성능은 여기서 나온다. "
             "애매·분류어려움=사람이 확신 못 해 학습에서 뺀 것.",
    )

    rep: MeshRep | None = None
    if source == SRC_FIT:
        if is_cv_run:
            st.caption(
                "이 실행은 5-fold 교차검증이라 **모든 메쉬가 정직하게 평가**되어 있습니다. "
                "각 메쉬를 학습에 쓰지 않은 fold 가 채점했습니다."
            )
            stat_df = load_predictions(run_dir)
            if not stat_df.empty:
                acc = float((stat_df["pred"] == stat_df["cls"]).mean())
                st.caption(f"{len(stat_df)}개 기준 **일치율 {acc:.1%}**")
        else:
            st.caption(
                "모델이 **학습에 쓴** 메쉬입니다. 예측이 라벨과 잘 맞는 것은 당연하며, "
                "이 화면의 값은 성능 근거가 되지 못합니다."
            )
        nm = pick_by_label(fit_names, "search_fit", "모델 이름", LBL)
        if nm:
            rep = parse_sample(nm)

    elif source == SRC_HOLD:
        st.caption(
            "학습에서 **일부러 빼둔** 메쉬입니다. 가중치 학습에도, 조기종료 판단에도 "
            "쓰이지 않았습니다. **이 모델의 성능은 이 묶음에서 나옵니다.**"
        )
        stat_df = load_predictions(run_dir)
        if not stat_df.empty:
            known = stat_df[stat_df["name"].isin(hold_names) & (stat_df["cls"] <= 1)]
            if len(known):
                acc = float((known["pred"] == known["cls"]).mean())
                st.caption(f"{len(known)}개 기준 **일치율 {acc:.1%}**")
        nm = pick_by_label(hold_names, "search_hold", "모델 이름", LBL)
        if nm:
            rep = parse_sample(nm)

    elif source == SRC_AMB:
        st.caption(
            "애매하거나 분류가 어려워 **학습에서 제외한** 메쉬입니다. 모델이 본 적은 "
            "없지만, 사람도 판단이 어려웠던 사례라 '정답' 자체가 불확실합니다."
        )
        kind = st.radio(
            "무엇을 볼지", ["둘 다", "애매만", "분류어려움만"],
            horizontal=True, key="amb_kind",
            help="애매=사람이 정상/비정상 판정에 확신이 없었던 것. "
                 "분류어려움=정상/비정상 자체를 정하지 못한 것.",
        )
        if kind == "애매만":
            shown = [n for n in amb_names if label_lookup(LBL).get(n, (0, 0))[0] != 2]
        elif kind == "분류어려움만":
            shown = [n for n in amb_names if label_lookup(LBL).get(n, (0, 0))[0] == 2]
        else:
            shown = amb_names

        stat_df = load_predictions(run_dir)
        if not stat_df.empty:
            known = stat_df[stat_df["name"].isin(shown) & (stat_df["cls"] <= 1)]
            if len(known):
                acc = float((known["pred"] == known["cls"]).mean())
                st.caption(
                    f"라벨이 정상/비정상인 {len(known)}개 기준 **일치율 {acc:.1%}** "
                    f"(참고용 - '정답'이 불확실한 집합입니다)"
                )
            else:
                st.caption("전부 분류어려움이라 일치율을 낼 수 없습니다.")

        if not shown:
            st.info("해당하는 메쉬가 없습니다.")
        else:
            nm = pick_by_label(shown, "search_amb", "모델 이름", LBL)
            if nm:
                rep = parse_sample(nm)

    elif source == SRC_ERR:
        pred_df = load_predictions(run_dir)
        if pred_df.empty:
            st.warning(
                "이 실행에는 predictions.csv 가 없습니다. "
                "`python scripts/train_deploy.py` 를 다시 돌리면 만들어집니다."
            )
        else:
            _hold = set(hold_names)
            opts = ["학습에 쓴 메쉬"] + (["홀드아웃"] if _hold else []) + ["애매·분류어려움", "전체"]
            scope = st.radio(
                "어느 집합에서", opts, horizontal=True, key="err_scope",
                help="홀드아웃=학습에서 빼둔 몫이라 여기 오분류가 실제 실력에 가깝다. "
                     "학습에 쓴 메쉬의 오분류는 '외우고도 틀린 것'이라 성격이 다르다.",
            )
            if scope == "학습에 쓴 메쉬":
                pool = pred_df[(pred_df["trainable"] == 1) & (~pred_df["name"].isin(_hold))]
            elif scope == "홀드아웃":
                pool = pred_df[pred_df["name"].isin(_hold)]
            elif scope == "애매·분류어려움":
                pool = pred_df[pred_df["trainable"] == 0]
            else:
                pool = pred_df
            pool = pool[pool["cls"] <= 1]   # 분류어려움은 정답이 없어 오분류를 따질 수 없다

            kind = st.radio(
                "무엇을 볼지",
                ["FP (정상을 비정상으로 오판)", "FN (비정상을 정상으로 오판)", "둘 다",
                 "경계선 근처 정답 (확신도 낮게 맞춘 사례)"],
                help="FP=실제 정상인데 모델이 비정상이라 한 것. FN=실제 비정상인데 정상이라 "
                     "한 것. 경계선 근처 정답=답은 맞혔지만 0.5 에 아슬아슬하게 가까운 확률로 "
                     "맞춘 사례.",
            )

            if kind == "경계선 근처 정답 (확신도 낮게 맞춘 사례)":
                max_conf = st.slider(
                    "확신도 상한 (0.5 에서 얼마나 떨어졌는지)", 0.0, 0.5, 0.15, 0.01,
                    help="0 에 가까울수록 아슬아슬하게 맞춘 사례. "
                         "예: 0.15 → 확률 0.35~0.65 사이로 맞춘 것만.",
                )
                sub = pool[(pool["kind"] == "정답") & (pool["conf"] <= max_conf)]
                sub = sub.sort_values("conf")
                st.caption(
                    f"{scope} 정답 {int((pool['kind'] == '정답').sum())}개 중 확신도 "
                    f"{max_conf:.2f} 이하 {len(sub)}개 (가장 아슬아슬한 것부터)"
                )
            else:
                want = {"FP (정상을 비정상으로 오판)": ["FP"],
                        "FN (비정상을 정상으로 오판)": ["FN"],
                        "둘 다": ["FP", "FN"]}[kind]
                sub = pool[pool["kind"].isin(want)].sort_values("conf", ascending=False)
                n_fp = int((pool["kind"] == "FP").sum())
                n_fn = int((pool["kind"] == "FN").sum())
                acc = float((pool["pred"] == pool["cls"]).mean())
                st.caption(
                    f"{scope} {len(pool)}개 중 FP {n_fp}개 / FN {n_fn}개 "
                    f"(일치율 {acc:.1%}, 현재 선택 {len(sub)}개 · 확신 있게 틀린 것부터)"
                )

            if sub.empty:
                st.info("해당하는 사례가 없습니다.")
            else:
                disp = [f"[{r.kind}] {r.name}  {label_tag(r.name, LBL)}  p={r.prob:.3f}"
                        for r in sub.itertuples()]
                nm = pick_display(disp, list(sub["name"]), "search_err", "사례")
                if nm:
                    rep = parse_sample(nm)

    elif source == SRC_UPLOAD:
        bdf_file = st.file_uploader(".bdf 파일", type=["bdf"])
        json_file = st.file_uploader(".json 파일 (선택 — 없으면 전 절점 외부연결=False)",
                                     type=["json"])
        if bdf_file is not None:
            name_hint = os.path.splitext(bdf_file.name)[0]
            rep = parse_from_bytes(
                bdf_file.getvalue(),
                json_file.getvalue() if json_file is not None else None,
                name_hint,
            )
            if json_file is None:
                st.warning("json 파일 없이 불러왔습니다 — 전 절점을 '외부연결 없음'으로 처리합니다.")

    st.divider()
    st.subheader("메쉬 표시")
    opacity = st.slider("쉘 투명도", 0.3, 1.0, 0.95)
    show_lines = st.checkbox("1D 요소(CBAR/CBEAM) 표시", value=True)
    show_ext = st.checkbox("외부연결 절점 표시", value=True)
    show_bad_outline = st.checkbox("규칙 기반 불량각도 요소 테두리", value=True)
    show_shell_edges = st.checkbox("쉘 경계선 표시 (요소끼리 구분)", value=True)

    st.divider()
    st.subheader("중요도 강조")
    highlight = st.checkbox("중요도 강조 켜기 (기본: 꺼짐 — 형상만 보기)", value=False)
    threshold = st.slider(
        "강조 임계값 (평균 대비 배수)", 1.0, 6.0, 2.0, 0.1,
        disabled=not highlight,
        help="이 배수 이상인 요소만 빨강으로 칠합니다. 나머지는 회색 그대로 둡니다.",
    )

if dev_mode:
    with st.expander("검증 실행 비교 (개발자용)", expanded=True):
        vt = validation_table()
        if vt.empty:
            st.info("runs/validate 에 결과가 없습니다. `python scripts/run_validation.py` 로 만듭니다.")
        else:
            # 설정이 같고 라벨 버전만 다른 쌍을 찾아 먼저 보여준다. 15개 표에서
            # 비교 가능한 두 줄을 사람이 눈으로 찾게 하면 엉뚱한 것끼리 비교하게 된다.
            # 같은 설정·같은 라벨인 실행이 여러 개면(시드만 다른 반복) 평균을 쓴다.
            # 그중 하나를 집으면 어느 것을 집느냐에 따라 결론이 흔들린다.
            vt2 = vt.copy()
            vt2["라벨키"] = vt2["라벨"].astype(str).str[:4]
            # 평가 방식까지 같아야 비교가 된다. 홀드아웃(10%)과 교차검증(전체)은
            # 채점 범위가 달라서 섞으면 안 된다.
            vt2["설정"] = vt2["설정"] + " · " + vt2["평가 방식"]
            agg = vt2.groupby(["설정", "라벨키"]).agg(
                AUC=("AUC", "mean"), ACC=("ACC", "mean"), F1=("F1", "mean"),
                n=("실행", "count"),
            ).reset_index()

            pairs = []
            for cfg, g in agg.groupby("설정"):
                labs = {r["라벨키"]: r for _, r in g.iterrows()}
                if "0721" in labs and "0918" in labs:
                    pairs.append((cfg, labs["0721"], labs["0918"]))
            if pairs:
                st.markdown("**같은 설정 · 라벨 버전만 다른 쌍** (같은 설정이 여러 번 돌았으면 평균)")
                for cfg, a_, b_ in pairs:
                    cols = st.columns(4)
                    cols[0].markdown(
                        f"`{cfg}`<br>0721 {int(a_['n'])}회 → 0918 {int(b_['n'])}회",
                        unsafe_allow_html=True,
                    )
                    for c, k in zip(cols[1:], ("AUC", "ACC", "F1")):
                        c.metric(k, f"{b_[k]:.4f}", f"{b_[k] - a_[k]:+.4f}")
                st.caption(
                    "위 화살표는 **성능 향상이 아니라 라벨 교체의 효과**다. 340개의 정답이 "
                    "바뀌었으므로 두 수치는 서로 다른 시험의 점수다."
                )
                st.divider()

            st.dataframe(vt, width="stretch", hide_index=True)
            st.warning(
                "**라벨 버전이 다른 실행끼리 수치를 비교하면 안 됩니다.** 0721 에서 0918 로 "
                "오면서 340개 메쉬의 정답이 바뀌었습니다. 즉 시험 문제 자체가 달라진 것이라, "
                "숫자가 올라간 것을 '모델이 좋아졌다'고 읽으면 틀립니다. "
                "**라벨이 정확해져서 채점 기준이 제대로 선 것**입니다."
            )
            st.caption(
                "같은 라벨 안에서는 비교할 수 있습니다. 다만 같은 설정을 다시 돌려도 "
                "AUC 가 ±0.002 정도는 움직입니다(초기 가중치·배치 순서). 그 이하 차이로 "
                "우열을 말하면 안 됩니다."
            )

if rep is None:
    st.info("왼쪽에서 모델을 선택하거나 .bdf 파일을 업로드하세요.")
    st.stop()

with st.spinner("예측 계산 중..."):
    result = predict(rep, run_dir, device_str)

col1, col2, col3 = st.columns(3)
col1.metric("예측", "비정상" if result["prob"] > 0.5 else "정상", f"p={result['prob']:.3f}")
if result["truth"] is not None:
    truth_str = "비정상" if result["truth"] == 1 else "정상"
    correct = (result["truth"] == 1) == (result["prob"] > 0.5)
    col2.metric("실제 라벨", truth_str, "일치" if correct else "불일치",
               delta_color="normal" if correct else "inverse")
else:
    lab = label_lookup(LBL).get(rep.name)
    if lab:
        cls, ambig = lab
        tag = {0: "정상", 1: "비정상", 2: "분류어려움"}[cls]
        col2.metric("라벨 엑셀 기록", tag, "애매" if ambig else "확실")
    else:
        col2.metric("실제 라벨", "정보 없음")
col3.metric("요소 수", f"{len(rep.shells)}쉘 + {len(rep.lines)}개 1D")

with st.expander("예측 상세 (정확한 확률·확신도)", expanded=False):
    p = result["prob"]
    margin = abs(p - 0.5)
    st.markdown(
        f"- **비정상 확률**: `{p:.6f}`  (반올림 없이)\n"
        f"- **정상 확률**: `{1 - p:.6f}`  (= 1 − 비정상 확률)\n"
        f"- **임계값(0.5) 대비 여유**: `{margin:.6f}`  "
        f"({'비정상' if p > 0.5 else '정상'} 쪽으로 {margin / 0.5 * 100:.1f}% 만큼 치우침)"
    )
    if result["truth"] is not None:
        st.markdown(
            f"- **실제 라벨과 일치 여부**: "
            f"{'✅ 일치' if (result['truth'] == 1) == (p > 0.5) else '❌ 불일치'}"
        )
    detail = result.get("detail", {})
    if detail.get("kind") == "full":
        a, b = detail["platt"]
        st.markdown(
            f"- **평가에 쓴 모델**: 전체 학습대상으로 학습한 **단일 모델** "
            f"(fold 분할 없음 - 어느 PC에서 열어도 같은 값이 나옵니다)\n"
            f"- **보정 전 원시 점수(logit)**: `{detail['logit']:.4f}`\n"
            f"- **확률 보정식**: `p = sigmoid({a:.3f} x logit + {b:.3f})`"
        )
        if detail.get("in_train"):
            st.warning(
                "이 메쉬는 **이 모델의 학습에 쓰인 메쉬**입니다. 모델이 이미 정답을 본 "
                "상태라 실제 라벨과 잘 맞는 것이 당연하며, 이 결과를 모델의 성능 근거로 "
                "삼으면 안 됩니다. 성능은 별도 5-fold 실행 결과로 보고합니다."
            )
    elif detail.get("kind") == "single":
        a, b = detail["platt"]
        st.markdown(
            f"- **평가에 쓴 모델**: fold {detail['fold']} (이 메쉬를 학습에 쓰지 않은 모델)\n"
            f"- **보정 전 원시 점수(logit)**: `{detail['logit']:.4f}`\n"
            f"- **확률 보정식**: `p = sigmoid({a:.3f} × logit + {b:.3f})` "
            f"(fold 간 확률 스케일을 맞추는 Platt 보정)"
        )
    elif detail.get("kind") == "ensemble":
        fp = np.array(detail["fold_probs"])
        st.markdown(
            f"- **평가 방식**: 5개 fold 모델의 확률을 평균 (학습셋에 없는 파일이라 "
            f"어느 fold가 정답인지 알 수 없어 전부 평균냄)\n"
            f"- **fold별 확률**: {', '.join(f'{v:.3f}' for v in fp)}\n"
            f"- **fold 간 표준편차**: `{fp.std():.4f}` "
            f"({'모델들 의견이 갈림 — 신뢰도 낮음' if fp.std() > 0.15 else '모델들 의견 일치 — 신뢰도 높음'})"
        )

st.caption(f"평가 방식: {result['mode']}")

tab_mesh, tab_graph = st.tabs(["3D 메쉬", "그래프 구조"])

with tab_mesh:
    fig = build_mesh_figure(
        rep, result["ga"], result["imp"], result["args"].graph,
        opacity, show_lines, show_ext, show_bad_outline, show_shell_edges,
        highlight, threshold,
    )
    st.plotly_chart(fig, width="stretch", key="mesh_fig")

    st.subheader("중요도 상위 요소")
    st.caption(
        "1D 요소(CBAR/CBEAM)는 형상 결함 판정 대상이 아니라 보강재 위상 정보라서 "
        "'판정' 칸이 항상 '-' 입니다. **왜 비정상인지는 아래 '쉘' 표를 먼저 보세요.**"
    )
    col_shell, col_line = st.columns(2)
    with col_shell:
        st.markdown("**쉘 요소 (형상 결함 후보)**")
        st.dataframe(
            top_elements_table(rep, result["imp"], result["args"].graph, k=8, only_shells=True),
            width="stretch", hide_index=True,
        )
    with col_line:
        st.markdown("**1D 요소 (참고용 — 결함 판정 아님)**")
        if result["args"].graph == "elem" and rep.lines:
            full_ranked = top_elements_table(
                rep, result["imp"], result["args"].graph, k=len(result["imp"])
            )
            line_rows = [r for r in full_ranked if r["종류"] in ("CBAR", "CBEAM")][:8]
            st.dataframe(line_rows, width="stretch", hide_index=True)
        else:
            st.caption("이 실행은 1D 요소를 그래프에 포함하지 않습니다.")

    n_bad = sum(1 for s in rep.shells if F.is_bad_shell(s.quality))
    st.caption(
        f"불량 각도 요소(최소내각<30° 또는 최대내각>150°): {n_bad}개  ·  "
        f"외부연결 절점: {sum(1 for v in rep.ext.values() if v)}/{len(rep.ext)}"
    )

with tab_graph:
    st.caption(
        "이 메쉬가 GNN 에 들어가기 전에 어떤 그래프로 바뀌는지 보여줍니다. "
        f"현재 실행({run_name})은 **{result['args'].graph}** 그래프를 씁니다."
    )
    if result["args"].graph == "elem":
        st.markdown(
            "노드 = 요소(쉘 검정 원 / 1D 주황 다이아몬드), "
            "엣지 = 요소 사이의 연결 관계. 관계 종류별로 표시 여부를 고를 수 있습니다."
        )
        cols = st.columns(4)
        show_rel = set()
        defaults = {"shell_edge": True, "shell_corner": False,
                   "line_shell": True, "line_line": True}
        for c, (key, (label, color)) in zip(cols, REL_STYLE.items()):
            if c.checkbox(label, value=defaults[key], key=f"rel_{key}"):
                show_rel.add(key)
        fig2 = build_graph_figure_elem(rep, show_rel)
    else:
        st.markdown("노드 = GRID 절점, 엣지 = 쉘 요소의 변 — 원본 메쉬와 거의 같은 구조입니다.")
        fig2 = build_graph_figure_node(rep)

    st.plotly_chart(fig2, width="stretch", key="graph_fig")

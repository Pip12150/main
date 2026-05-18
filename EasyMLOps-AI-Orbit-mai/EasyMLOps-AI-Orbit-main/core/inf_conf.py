# core/inf_conf.py
import glob
import os
import re
import cv2
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from ultralytics import YOLO

import pandas as pd

from core.file_browser import IMAGE_EXTS

DATE_DIR_RE = re.compile(r"^\d{8}$")  # YYYYMMDD

@dataclass(frozen=True)
class DailySummary:
    """
        [데이터 구조] 일자별 confidence 요약 정보를 담기 위한 간단한 DTO.

        Attributes:
            date: 날짜 문자열(YYYYMMDD)
            count: 해당 날짜에 수집된 confidence 개수(객체 단위)
            mean: 해당 날짜 confidence 평균
    """
    date: str
    count: int
    mean: float


def list_date_dirs(inf_root: str) -> list[str]:
    """
        [날짜 디렉토리 목록 조회]
        inf_root 하위에서 폴더명이 YYYYMMDD(8자리 숫자)인 디렉토리만 추려
        날짜 목록으로 반환한다.

        입력:
            inf_root: inference 결과 루트 경로(예: /.../inference_results)

        출력:
            dates: ["20251201","20251202", ...] (오름차순 정렬)

        주의:
            - inf_root가 없거나 디렉토리가 아니면 빈 리스트 반환
            - DATE_DIR_RE(8자리 숫자)만 허용하므로 다른 폴더명은 무시된다.
    """
    if not inf_root or not os.path.isdir(inf_root):
        return []
    dates = []
    for name in os.listdir(inf_root):
        p = os.path.join(inf_root, name)
        if os.path.isdir(p) and DATE_DIR_RE.match(name):
            dates.append(name)
    dates.sort()
    return dates


def read_conf_from_polygon_txt(
    txt_path: str,
    class_filter: str = "",
    conf_min: Optional[float] = None,
) -> list[float]:
    """
        [polygon txt에서 conf 추출]
        txt 파일에서 confidence 값(2번째 토큰)을 추출하여 리스트로 반환한다.

        기대하는 라인 형식:
            cls conf x1 y1 x2 y2 ...

        필터 옵션:
            - class_filter: 값이 있으면 cls가 일치하는 라인만 사용
            - conf_min: 값이 있으면 conf >= conf_min 인 값만 사용

        입력:
            txt_path: 추론 결과 txt 경로
            class_filter: 특정 클래스만 집계할 때 사용(기본 "" = 전체)
            conf_min: 최소 conf threshold (기본 None = 적용 안함)

        출력:
            confs: [0.81, 0.55, ...] (라인 단위로 수집)

        주의:
            - 파일 읽기 실패/파싱 실패 라인은 스킵
            - 예외 발생 시 빈 리스트 반환
    """
    confs: list[float] = []
    try:
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                cls = parts[0]
                if class_filter and cls != class_filter:
                    continue
                try:
                    conf = float(parts[1])
                except Exception:
                    continue
                if conf_min is not None and conf < conf_min:
                    continue
                confs.append(conf)
    except Exception:
        return []
    return confs


def collect_daily_confidence(
    inf_root: str,
    class_filter: str = "",
    conf_min: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        [일자별 confidence 수집 + 요약]
        inf_root/YYYYMMDD/txt/*.txt 구조를 가정하고,
        모든 txt에서 conf를 모아 raw 데이터와 일자 요약 데이터를 만든다.

        데이터 수집 단위:
            - "객체 단위(conf)"로 수집
              (txt 한 줄 = 객체 하나라고 가정, 그 줄의 2번째 토큰이 conf)

        입력:
            inf_root: inference 결과 루트
            class_filter: 특정 클래스만 집계(기본 전체)
            conf_min: 최소 conf로 필터링(기본 None)

        출력:
            raw_df:
                columns=["date","confidence"]
                - date: YYYYMMDD
                - confidence: float
            summary_df:
                columns=["date","count","mean"]
                - date: YYYYMMDD
                - count: 해당 날짜의 conf 개수(객체 수)
                - mean: 해당 날짜 conf 평균

        주의:
            - 날짜 폴더에 txt 디렉토리가 없으면 스킵
            - rows가 비거나 numeric 변환 후 raw가 비면 (빈 DF, 빈 DF) 반환
            - docstring에는 median/p10/p90까지 써있지만 현재 코드는 mean/count만 생성함
              (필요하면 추가 계산 로직을 넣어 확장 가능)
    """
    dates = list_date_dirs(inf_root)
    rows: list[tuple[str, float]] = []

    for d in dates:
        txt_dir = os.path.join(inf_root, d, "txt")
        if not os.path.isdir(txt_dir):
            continue

        for fn in os.listdir(txt_dir):
            if not fn.lower().endswith(".txt"):
                continue
            p = os.path.join(txt_dir, fn)
            confs = read_conf_from_polygon_txt(p, class_filter=class_filter, conf_min=conf_min)
            for c in confs:
                rows.append((d, c))

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    raw = pd.DataFrame(rows, columns=["date", "confidence"])
    raw["date"] = raw["date"].astype(str)
    raw["confidence"] = pd.to_numeric(raw["confidence"], errors="coerce")
    raw = raw.dropna(subset=["confidence"])

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    g = raw.groupby("date")["confidence"]

    def q(s, p: float) -> float:
        return float(s.quantile(p)) if len(s) else math.nan

    summary = pd.DataFrame({
        "date": g.size().index,
        "count": g.size().values,
        "mean": g.mean().values
    }).sort_values("date")

    return raw, summary

def _result_to_summary_df(result) -> pd.DataFrame:
    """
        [Ultralytics Result -> 클래스별 conf 요약 DF]
        model.predict()[0]로 얻은 Result 객체에서
        boxes(cls/conf)를 이용해 클래스별 통계를 산출한다.

        산출 컬럼:
            - cls: 클래스 id(int)
            - count: 해당 클래스 탐지 개수
            - conf_mean: conf 평균
            - conf_min: conf 최소
            - conf_max: conf 최대

        입력:
            result: Ultralytics Result 객체 (단일 이미지)

        출력:
            summary_df: 위 컬럼으로 구성된 DataFrame
                        탐지가 없으면 빈 DF(컬럼은 유지)

        주의:
            - segmentation이어도 boxes가 존재하면 동일 로직으로 동작
            - boxes가 없거나 0개면 빈 DF 반환
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(columns=["cls", "count", "conf_mean", "conf_min", "conf_max"])

    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy().astype(float)

    df = pd.DataFrame({"cls": cls, "conf": conf})
    g = df.groupby("cls")["conf"]
    out = pd.DataFrame({
        "cls": g.mean().index.astype(int),
        "count": g.size().values.astype(int),
        "conf_mean": g.mean().values,
        "conf_min": g.min().values,
        "conf_max": g.max().values,
    }).sort_values("cls")

    return out

def _predict_one(model_path: str, img_bgr: np.ndarray, imgsz: int, conf: float, iou: float, device: str):
    """
        [단일 모델 추론 + 시각화 + 요약]
        주어진 모델(model_path)로 단일 이미지(img_bgr)를 추론하고,
        (1) 시각화 이미지(res.plot)와 (2) 클래스별 conf 요약 DF를 반환한다.

        입력:
            model_path: YOLO weight 경로(.pt)
            img_bgr: OpenCV로 로드한 BGR ndarray (원본 스케일 유지)
            imgsz: YOLO 입력 사이즈 (Ultralytics 내부에서 letterbox 처리)
            conf: conf threshold
            iou: iou threshold
            device: device 문자열

        출력:
            vis: res.plot() 결과(BGR ndarray) - 박스/마스크가 그려진 이미지
            summary: _result_to_summary_df() 결과 DF

        주의:
            - 현재 구현은 매 호출마다 YOLO(model_path)를 새로 로드한다.
              (두 모델 비교 UI에서 호출 횟수가 많아지면 모델 캐싱을 고려할 수 있음)
            - “강제 resize 금지”: source에 원본 ndarray를 그대로 넣어
              plot 결과가 원본 스케일에 자연스럽게 맞도록 유지한다.
    """
    model = YOLO(model_path)

    # ✅ 원본 그대로 넣기 (강제 resize 금지)
    res = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False
    )[0]

    # ✅ plot은 원본 스케일에 맞춰 잘 나옴
    vis = res.plot(
        line_width=2,
        font_size=0.6,
    )  # BGR ndarray

    summary = _result_to_summary_df(res)
    return vis, summary, res

def compare_infer_two_models(
    img_path: str,
    old_model_path: str,
    new_model_path: str,
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    device: str = "cuda:1"
):
    """
        [두 모델 비교 추론]
        하나의 입력 이미지(img_path)에 대해
        old_model_path / new_model_path 두 모델로 각각 추론하고,
        결과 시각화 이미지와 클래스별 conf 요약 DF를 함께 반환한다.

        입력:
            img_path: 원본 이미지 파일 경로
            old_model_path: 비교 대상(기존) 모델 weight 경로
            new_model_path: 비교 대상(최신) 모델 weight 경로
            imgsz/conf_thres/iou_thres/device: 추론 파라미터

        출력(성공):
            old_vis: 기존 모델 시각화 결과(BGR ndarray)
            new_vis: 최신 모델 시각화 결과(BGR ndarray)
            old_df: 기존 모델 클래스별 요약 DF
            new_df: 최신 모델 클래스별 요약 DF
            msg: "[OK] 비교 추론 완료"

        출력(실패):
            (None, None, None, None, "[ERROR] ...") 형태로 에러 메시지 반환

        주의:
            - 이미지 로드는 cv2.imread로 수행(BGR)
            - _predict_one을 2회 호출(각 모델 1회씩)
    """
    if not img_path:
        return None, None, None, None, "[ERROR] 원본 이미지를 선택해줘"
    if not old_model_path:
        return None, None, None, None, "[ERROR] 기존 모델 경로를 선택해줘"
    if not new_model_path:
        return None, None, None, None, "[ERROR] 최신 모델 경로를 선택해줘"

    img = cv2.imread(img_path)
    if img is None:
        return None, None, None, None, f"[ERROR] 이미지 로드 실패: {img_path}"

    old_vis, old_df = _predict_one(old_model_path, img, imgsz, conf_thres, iou_thres, device)
    new_vis, new_df = _predict_one(new_model_path, img, imgsz, conf_thres, iou_thres, device )

    return old_vis, new_vis, old_df, new_df, "[OK] 비교 추론 완료"

def summarize_conf_for_dir(
    img_dir: str,
    model_path: str,
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    device: str = "cuda:1"
) -> pd.DataFrame:
    """
    폴더 내 모든 이미지에 대해 model로 추론 후,
    클래스별 conf 통계 + 전체(ALL) conf 통계를 반환한다.

    반환 컬럼:
      - cls: int (클래스 id), 전체 요약은 -1 (ALL)
      - count: int
      - conf_mean/conf_min/conf_max: float

    정렬:
      - 클래스(0,1,2,...) 오름차순
      - ALL(-1)은 항상 마지막
    """
    empty = pd.DataFrame(columns=["cls", "count", "conf_mean", "conf_min", "conf_max"])

    if not img_dir or not os.path.isdir(img_dir):
        return empty
    if not model_path:
        return empty

    # 이미지 수집 (폴더 직속 파일만)
    imgs: list[str] = []
    for ext in IMAGE_EXTS:
        imgs += glob.glob(os.path.join(img_dir, f"*{ext}"))
        imgs += glob.glob(os.path.join(img_dir, f"*{ext.upper()}"))
    imgs = sorted(set(imgs))

    print("[DIR]", img_dir)
    print("[N IMAGES]", len(imgs))
    print("[FIRST 5]", list(imgs)[:5])

    if not imgs:
        return empty

    # cls -> {count, conf_sum, conf_min, conf_max}
    acc: dict[int, dict[str, float]] = {}

    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue

        # _predict_one은 (vis, per_img_df) 반환
        _, per_img_df = _predict_one(model_path, img, imgsz, conf_thres, iou_thres, device)
        if per_img_df is None or per_img_df.empty:
            continue

        for _, r in per_img_df.iterrows():
            cls = int(r["cls"])
            cnt = int(r["count"])
            mean = float(r["conf_mean"])
            cmin = float(r["conf_min"])
            cmax = float(r["conf_max"])

            # mean만 있으니 sum으로 복원
            conf_sum = mean * cnt

            if cls not in acc:
                acc[cls] = {
                    "count": 0,
                    "conf_sum": 0.0,
                    "conf_min": cmin,
                    "conf_max": cmax,
                }

            acc[cls]["count"] += cnt
            acc[cls]["conf_sum"] += conf_sum
            acc[cls]["conf_min"] = min(acc[cls]["conf_min"], cmin)
            acc[cls]["conf_max"] = max(acc[cls]["conf_max"], cmax)

    if not acc:
        return empty

    # rows 생성 + 전체(ALL) 누적
    rows: list[dict] = []

    total_count = 0
    total_conf_sum = 0.0
    total_conf_min: Optional[float] = None
    total_conf_max: Optional[float] = None

    for cls, v in acc.items():
        cnt = int(v["count"])
        conf_sum = float(v["conf_sum"])
        cmin = float(v["conf_min"])
        cmax = float(v["conf_max"])

        rows.append({
            "cls": cls,
            "count": cnt,
            "conf_mean": (conf_sum / cnt) if cnt else 0.0,
            "conf_min": cmin,
            "conf_max": cmax,
        })

        total_count += cnt
        total_conf_sum += conf_sum
        total_conf_min = cmin if total_conf_min is None else min(total_conf_min, cmin)
        total_conf_max = cmax if total_conf_max is None else max(total_conf_max, cmax)

    # 전체(ALL) row 추가: cls = -1 (Gradio number 타입 유지)
    if total_count > 0 and total_conf_min is not None and total_conf_max is not None:
        rows.append({
            "cls": -1,  # ALL
            "count": total_count,
            "conf_mean": total_conf_sum / total_count,
            "conf_min": total_conf_min,
            "conf_max": total_conf_max,
        })

    df = pd.DataFrame(rows)

    # 정렬: 클래스 오름차순, ALL(-1)은 마지막 고정
    df_all = df[df["cls"] == -1]
    df_cls = df[df["cls"] != -1].sort_values("cls")
    df = pd.concat([df_cls, df_all], ignore_index=True)

    # 소수점 자리수 정리 (conf 관련 컬럼만)
    for col in ["conf_mean", "conf_min", "conf_max"]:
        if col in df.columns:
            df[col] = df[col].round(4)

    return df.reset_index(drop=True)


    """
    rows = []
    for cls, v in acc.items():
        cnt = v["count"]
        rows.append({
            "cls": cls,
            "count": cnt,
            "conf_mean": (v["conf_sum"] / cnt) if cnt else 0.0,
            "conf_min": v["conf_min"],
            "conf_max": v["conf_max"],
        })

    return pd.DataFrame(rows).sort_values("cls").reset_index(drop=True)
    """



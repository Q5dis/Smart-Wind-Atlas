#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASOS(종관기상관측) 풍속 데이터 수집 + Weibull 피팅 예제
- 출처: 공공데이터포털(data.go.kr) 기상청 ASOS 시간자료 API
- 사용 전: 아래 CONFIG에서 API_KEY, STATION_ID, 기간을 설정하세요.
- 의존성: requests, pandas, numpy, scipy, matplotlib
    pip install requests pandas numpy scipy matplotlib
"""

import sys
import math
import json
import time
from datetime import datetime
from typing import Dict, Any, List

import requests
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt


# -----------------------
# CONFIG
# -----------------------
API_KEY = "여기에_본인_API_키_붙여넣기"   # 공공데이터포털 발급 키 (URL 인코딩되지 않은 일반 키)
STATION_ID = "108"  # 서울=108, 부산=159 등 (아래 STATION_MAP 참고)
START_DATE = "20240101"  # YYYYMMDD
START_HH   = "00"        # 00~23
END_DATE   = "20241231"
END_HH     = "23"

# 여러 관측소 코드 참고용(일부)
STATION_MAP = {
    "서울": "108",
    "부산": "159",
    "인천": "112",
    "대구": "143",
    "광주": "156",
    "대전": "133",
    "울산": "152",
    "춘천": "101",
    "강릉": "105",
    "청주": "131",
    "전주": "146",
    "제주": "184",
}


def fetch_asos_hourly(api_key: str, stn_id: str, start_dt: str, start_hh: str, end_dt: str, end_hh: str) -> pd.DataFrame:
    """
    ASOS 시간자료 API 호출
    문서 예시 엔드포인트:
    http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList

    필수 파라미터(일부):
        serviceKey, dataType, dataCd=ASOS, dateCd=HR,
        startDt, startHh, endDt, endHh, stnIds

    반환: 풍속/풍향/기온/습도 등 컬럼을 포함한 DataFrame
    """
    base_url = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    params = {
        "serviceKey": api_key,
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "startDt": start_dt,
        "startHh": start_hh,
        "endDt": end_dt,
        "endHh": end_hh,
        "stnIds": stn_id,
    }
    print("[INFO] 요청:", base_url)
    r = requests.get(base_url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("response", {}).get("header", {}).get("resultCode") != "00":
        print("[WARN] API header:", j.get("response", {}).get("header"))
    items = j.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if not items:
        print("[ERROR] 수신된 데이터가 없습니다. 파라미터/기간/관측소 코드를 확인하세요.")
        return pd.DataFrame()

    df = pd.DataFrame(items)
    # 주요 컬럼 이름 참조(제공 스키마에 따라 다소 변경될 수 있음)
    # tm: 관측시각, wd: 풍향(도), ws: 풍속(m/s), ta: 기온(°C), hm: 상대습도(%)
    # 필요 시 컬럼명 확인:
    print("[INFO] 컬럼:", list(df.columns))

    # 시각을 pandas datetime으로 변환
    if "tm" in df.columns:
        df["tm"] = pd.to_datetime(df["tm"], errors="coerce")
        df = df.sort_values("tm").reset_index(drop=True)

    # 숫자 컬럼을 float으로 변환
    for col in ["ws", "wd", "ta", "hm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fit_weibull(ws: pd.Series):
    """
    풍속 데이터로 Weibull 분포(k, c) 피팅.
    - 음수/결측 제거
    - 위치(loc)=0으로 고정(풍속은 음수가 아님)
    반환: k(shape), c(scale)
    """
    ws = ws.dropna()
    ws = ws[ws >= 0]
    if len(ws) < 100:
        print("[WARN] 샘플 수가 적습니다(100 미만). 결과가 불안정할 수 있습니다.")
    # floc=0으로 위치를 0으로 고정
    k, loc, c = weibull_min.fit(ws.values, floc=0)
    return k, c


def plot_hist_with_weibull(ws: pd.Series, k: float, c: float, out_png: str):
    """
    히스토그램 + Weibull PDF를 함께 플로팅
    """
    ws = ws.dropna()
    ws = ws[ws >= 0]
    if ws.empty:
        print("[WARN] 그릴 데이터가 없습니다.")
        return

    # 히스토그램
    plt.figure(figsize=(8, 5))
    n, bins, _ = plt.hist(ws.values, bins=40, density=True, alpha=0.5, label="Observed wind speed")

    # 이론 PDF
    x = np.linspace(0, max(ws.max(), 15), 400)
    pdf = weibull_min.pdf(x, k, scale=c)

    plt.plot(x, pdf, linewidth=2, label=f"Weibull PDF (k={k:.2f}, c={c:.2f})")
    plt.title("Wind Speed Distribution & Fitted Weibull")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[INFO] 그래프 저장: {out_png}")


def main():
    # 1) 데이터 수집
    df = fetch_asos_hourly(API_KEY, STATION_ID, START_DATE, START_HH, END_DATE, END_HH)
    if df.empty:
        sys.exit(1)

    # 2) CSV 저장
    out_csv = f"asos_{STATION_ID}_{START_DATE}_{END_DATE}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] CSV 저장: {out_csv} (rows={len(df)})")

    # 3) 풍속 컬럼 이름 확인 및 피팅
    if "ws" not in df.columns:
        print("[ERROR] 'ws'(풍속) 컬럼이 없습니다. API 응답 스키마를 먼저 확인하세요.")
        print("       print(df.head())로 컬럼명을 확인한 뒤, 풍속 컬럼명을 코드에서 수정하십시오.")
        print(df.head())
        sys.exit(1)

    k, c = fit_weibull(df["ws"])
    print(f"[RESULT] Weibull params: shape k={k:.4f}, scale c={c:.4f}")

    # 4) 시각화
    out_png = f"asos_{STATION_ID}_{START_DATE}_{END_DATE}_weibull.png"
    plot_hist_with_weibull(df["ws"], k, c, out_png)

    # 5) 간단한 지표 출력(평균풍속, 컷인/컷아웃 대비 비율 등은 후속 확장)
    mean_ws = df["ws"].dropna().mean()
    print(f"[INFO] 기간 평균 풍속: {mean_ws:.3f} m/s")

    print("\n[다음 단계 아이디어]")
    print("- 관측소 다중 병합: 인근 ASOS/AWS와 결합해 신뢰도 향상")
    print("- 허브고도 보정: 로그법/파워법으로 10m→허브 높이(예: 100m) 보정")
    print("- 터빈 파워커브 적용: 빈도×파워커브 컨볼루션으로 AEP 계산")
    print("- 계절/시간대별 분포: 풍향장미, 계절성 분석")

if __name__ == "__main__":
    main()

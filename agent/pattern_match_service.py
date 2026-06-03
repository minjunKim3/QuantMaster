"""
pattern_match_service.py — CLI 진입점
AItr_simulator 원본 로직(DataLoader/PatternMatcher/Simulator)을 그대로 재사용한다.
Streamlit UI(main.py) 의존성 없이 ProcessBuilder 에서 1회 호출용으로 동작.

호출:
  python pattern_match_service.py '{"ticker":"005930","prediction_days":30}'
  python pattern_match_service.py /path/to/params.json
  echo {...} | python pattern_match_service.py     # stdin
"""
import sys
import os
import json
import math
import numpy as np

# AItr_simulator 패키지 경로 등록
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_THIS_DIR, "AItr_simulator")
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from src.data_loader import DataLoader
from src.pattern_matcher import PatternMatcher
from src.simulator import Simulator


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return v if math.isfinite(v) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _clean_array(arr):
    """numpy 배열/리스트의 NaN/Inf -> None 정규화"""
    out = []
    for v in arr:
        try:
            f = float(v)
            out.append(f if math.isfinite(f) else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _load_params():
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            with open(arg, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(arg)
    # stdin 폴백
    data = sys.stdin.read().strip()
    return json.loads(data) if data else {}


def run(params):
    ticker = str(params.get("ticker", "")).strip()
    if not ticker:
        return {"error": "ticker 가 비어 있습니다."}

    prediction_days = int(params.get("prediction_days", 20))
    scan_window = int(params.get("scan_window", 20))

    loader = DataLoader(ticker=ticker)
    df_all, df_current = loader.get_data(window_size=scan_window)
    if df_all.empty or df_current.empty:
        return {"error": "데이터 로드 실패 또는 부족"}

    matcher = PatternMatcher(df_all=df_all, df_current=df_current)
    top_matches = matcher.find_top_matches(
        window_size=scan_window, pred_days=prediction_days, top_n=5
    )
    if not top_matches:
        return {"error": "유사한 패턴을 찾지 못했습니다."}

    sim = Simulator(top_matches, amount=100, pred_days=prediction_days)
    sim_data = sim.calculate_scenario()

    best = top_matches[0]
    distance = float(best["distance"])
    similarity_pct = max(0.0, 100.0 - (distance * 15.0))

    # 시각화 호환: 현재 시작가 기준 scale_ratio 적용한 과거 패턴 close
    current_start_price = float(df_current["Close"].iloc[0])
    matched_df = best["matched_df"]
    past_start_price = float(matched_df["Close"].iloc[0])
    scale_ratio = (current_start_price / past_start_price) if past_start_price > 0 else 1.0
    matched_scaled = (matched_df["Close"].values * scale_ratio).tolist()

    bull = sim_data.get("bull_scenario", {})
    bear = sim_data.get("bear_scenario", {})

    def build_scenario(scn):
        """visualizer.py generate_chart 의 plot_scenario() 로직 그대로:
        과거 패턴 + 미래 투영을 모두 current_start_price 기준 scale_ratio 로 보정한다."""
        m = scn.get("match")
        future_df = scn.get("future_df")
        pct = float(scn.get("pct", 0.0))
        if m is None:
            return {
                "pct": round(pct, 4),
                "matched_start_date": None,
                "matched_pattern_close_array": [],
                "future_close_array": [],
            }
        m_matched = m["matched_df"]
        past_start = float(m_matched["Close"].iloc[0])
        ratio = (current_start_price / past_start) if past_start > 0 else 1.0
        matched_arr = (m_matched["Close"].values * ratio).tolist()

        future_arr = []
        if future_df is not None and not future_df.empty:
            # visualizer.py: y_future_raw = [matched_last] + future.Close → 모두 * ratio
            past_last = float(m_matched["Close"].iloc[-1])
            future_raw = [past_last] + future_df["Close"].values.tolist()
            future_arr = [float(v) * ratio for v in future_raw]
        return {
            "pct": round(pct, 4),
            "matched_start_date": m["start_date"].strftime("%Y-%m-%d"),
            "matched_pattern_close_array": _clean_array(matched_arr),
            "future_close_array": _clean_array(future_arr),
        }

    return {
        "ticker": ticker,
        "prediction_days": prediction_days,
        "similarity_pct": round(float(similarity_pct), 2),
        "matched_start_date": best["start_date"].strftime("%Y-%m-%d"),
        "matched_end_date": best["end_date"].strftime("%Y-%m-%d"),
        "bull_scenario": build_scenario(bull),
        "bear_scenario": build_scenario(bear),
        "df_current_close_array": _clean_array(df_current["Close"].values),
        "matched_pattern_close_array": _clean_array(matched_scaled),
    }


def main():
    try:
        params = _load_params()
        result = run(params)
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(result, cls=NpEncoder, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

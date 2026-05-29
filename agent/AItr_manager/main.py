import sys
import os
import time
# 한국어(CP949) 콘솔에서도 이모지/한글 출력이 깨지지 않도록 표준출력을 UTF-8로 설정
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 이 파일이 있는 폴더를 import 경로에 추가 (다른 작업 디렉토리에서 실행돼도 src 패키지를 찾도록)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine import StockEngine
from src.agent import StockAgent
import json
import math
import numpy as np


# lstm_service.py 등 기존 agent 스크립트와 동일한 numpy 직렬화 패턴
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _clean(obj):
    """NaN/Infinity 를 null 로 치환해 '표준 JSON'을 보장한다.

    엔진 스코어링에서 NaN 점수가 나올 수 있는데, 파이썬 json 은 이를 `NaN`(비표준)
    으로 출력해 프론트엔드의 JSON.parse 가 실패한다. 결과를 만들기 직전 정리한다.
    (엔진 로직 자체는 손대지 않음 — 직렬화 안전성만 보장)
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj

def main():
    print("="*60)
    print("AI Agent 주식 분석 시스템")
    print("="*60)
    
    engine = StockEngine()
    agent = StockAgent()
    
    user_input = input("\n분석 요청을 입력하세요: ")
    if not user_input.strip(): user_input = "3일동안 안정적인 우량주 추천"
    
    # 1. 의도 분석 및 가중치 확인
    params = agent.extract_params(user_input)
    
    # [시스템 출력] 추출된 가중치 표시
    print("\n" + "-"*20 + " [지표 가중치 매핑] " + "-"*20)
    print(json.dumps(params, indent=4, ensure_ascii=False))
    print("-" * 65 + "\n")
    
    # 2. 엔진 스크리닝 (50개)
    top_50 = engine.get_filtered_candidates(params)
    
    # [Error Fix] 리스트 분할 정의
    top_5 = top_50[:5]
    others_45 = top_50[5:] # NameError 방지
    
    # 3. 리포트 생성 및 재검토
    raw_report = agent.generate_report(user_input, top_5)
    final_report = agent.refine_report(raw_report)

    print("\n" + "-" * 50)
    print("AI 심층 분석")
    print("-" * 50)
    print(final_report)

    # 4. 나머지 45개 리스트 출력 (코드 실행)
    print("\n" + "-" * 20 + " [추가 후보 45개 리스트] " + "-" * 20)
    print(f"{'종목명(코드)':<20} | {'수익률':<8} | {'RSI':<6} | {'거래비율':<8} | {'점수':<6}")
    print("-" * 65)
    for s in others_45:
        name_code = f"{s['name']}({s['ticker']})"
        print(f"{name_code:<20} | {s['price_chg']:>7}% | {s['rsi']:>6.1f} | {s['vol_ratio']:>8.1f}배 | {s['score']:>6.1f}")
    print("-" * 65)

def run_cli(input_arg):
    """1회 호출(non-interactive) 모드.

    Java(Spring) ProcessBuilder가 호출. lstm_service.py 와 동일하게
    argv[1] 로 'JSON 파일 경로' 또는 'JSON 문자열' 을 받는다.
    파이프라인을 1회 실행하고, 마지막 줄에 결과 JSON 한 줄을 stdout 으로 출력한다.
    (Java 쪽은 '{' 로 시작하는 마지막 줄만 결과로 채택)
    """
    # 입력 파싱: 파일 경로면 읽고, 아니면 JSON 문자열로 간주
    try:
        if os.path.exists(input_arg):
            with open(input_arg, "r", encoding="utf-8") as f:
                req = json.load(f)
        else:
            req = json.loads(input_arg)
    except Exception:
        # 순수 텍스트가 넘어온 경우 query 로 취급
        req = {"query": input_arg}

    user_input = (req.get("query") or "").strip()
    if not user_input:
        user_input = "3일동안 안정적인 우량주 추천"

    t0 = time.time()
    elapsed = {}

    try:
        engine = StockEngine()
        agent = StockAgent()  # gguf 모델 없으면 여기서 FileNotFoundError

        # 1) 사용자 의도 → 지표 가중치
        t = time.time()
        params = agent.extract_params(user_input)
        elapsed["extract"] = round(time.time() - t, 1)

        # 2) 엔진 스코어링 (300종목 스캔 → 상위 50)
        t = time.time()
        top_50 = engine.get_filtered_candidates(params)
        elapsed["engine"] = round(time.time() - t, 1)
        top_5 = top_50[:5]

        # 3) 심층 리포트 생성 + 재검수
        t = time.time()
        raw_report = agent.generate_report(user_input, top_5)
        final_report = agent.refine_report(raw_report)
        elapsed["agent"] = round(time.time() - t, 1)

        elapsed["total"] = round(time.time() - t0, 1)

        result = _clean({
            "query": user_input,
            "extracted_params": params,
            "scored_stocks": top_50,
            "report": final_report,
            "elapsed": elapsed,
        })
        # 결과는 반드시 '{' 로 시작하는 '한 줄' JSON (개행은 \\n 으로 이스케이프됨)
        # allow_nan=False: 혹시 남은 NaN 이 있으면 비표준 JSON 대신 즉시 에러로 드러나게
        print(json.dumps(result, ensure_ascii=False, cls=NpEncoder, allow_nan=False))
        return 0
    except Exception as e:
        # 모델 파일 없음 등 실패 시 에러 JSON 한 줄 + 비정상 종료코드
        err = {"error": str(e), "elapsed": {"total": round(time.time() - t0, 1)}}
        print(json.dumps(err, ensure_ascii=False, cls=NpEncoder))
        return 1


if __name__ == "__main__":
    # 인자가 있으면 1회 호출 모드(Java 연동), 없으면 기존 대화형 모드
    if len(sys.argv) > 1:
        sys.exit(run_cli(sys.argv[1]))
    main()
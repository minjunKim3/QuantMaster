import sys
import json
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import os

sys.stdout.reconfigure(encoding='utf-8')

def extract_weights_from_llm(query):
    """사용자 요청을 LLM이 분석하여 6개 지표 가중치 자동 설정"""
    default_weights = {
        'price_chg': -1.0, 'vol_ratio': 1.0, 'rebound': 1.0,
        'stability': 0.0, 'trend': 0.0, 'volatility': 1.0
    }
    
    if not query or query == "시총 상위 종목 스캔":
        return default_weights
    
    try:
        import re
        from langchain_ollama import OllamaLLM
        
        llm = OllamaLLM(model="gemma3:4b")
        
        prompt = f"""사용자 요청을 분석하여 다음 지표 가중치(-1.0 ~ 1.0)를 JSON으로 설정하세요.

[지표별 키워드 매핑 가이드]
- price_chg: 급등(+), 상승(+), 하락(-), 급락(-)
- vol_ratio: 수급, 거래량 많은, 관심집중, 거래량 급증
- rebound: 저점, 반등, 저평가, RSI 낮음, BB 하단
- stability: 안정성, 우량주, 이격도 낮음, 꾸준한, 정배열
- trend: 상승 추세, MACD, 골든크로스, 상승세 유지
- volatility: 변동성, 단타, 위험, 기회

결과는 오직 JSON만 출력하세요. 다른 설명은 하지 마세요.
{{ "days": 정수, "weights": {{ "price_chg": 0.0, "vol_ratio": 0.0, "rebound": 0.0, "stability": 0.0, "trend": 0.0, "volatility": 0.0 }} }}
요청: {query}"""
        
        print(f"[Agent] LLM 가중치 추출 중...", file=sys.stderr)
        res = llm.invoke(prompt)
        
        match = re.search(r'\{.*\}', res, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if 'weights' in parsed:
                print(f"[Agent] LLM 가중치: {parsed['weights']}", file=sys.stderr)
                return parsed['weights']
        
        return default_weights
        
    except Exception as e:
        print(f"[Agent] LLM 연결 실패, 기본 가중치 사용: {e}", file=sys.stderr)
        return default_weights

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def calculate_indicators(df):
    """조원분 engine.py 방식 — 7개 지표 계산"""
    close = df['Close']
    
    # RSI (EMA 방식)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rsi = 100 - (100 / (1 + (ema_up / ema_down)))
    
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    ma20 = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    upper_bb = ma20 + (std * 2)
    lower_bb = ma20 - (std * 2)
    bb_pct = (close - lower_bb) / (upper_bb - lower_bb + 1e-10)
    
    # 이격도 (MA Gap)
    ma_gap = (close / ma20) - 1
    
    return {
        'rsi': round(float(rsi.iloc[-1]), 2),
        'macd': round(float(macd.iloc[-1]), 4),
        'macd_sig': round(float(signal.iloc[-1]), 4),
        'bb_pct': round(float(bb_pct.iloc[-1]), 3),
        'ma_gap': round(float(ma_gap.iloc[-1]), 4)
    }

def run_analysis(query="", days=7):
    """주식 분석 실행 — 조원분 engine.py 로직 반영"""
    
    print(f"[Agent] 분석 시작 ({days}일 기준)", file=sys.stderr)
    
    # LLM으로 가중치 동적 추출 시도
    weights = extract_weights_from_llm(query)
    
    # 종목 리스트 가져오기
    try:
        df_listing = fdr.StockListing('KRX')
        if 'Marcap' in df_listing.columns:
            df_listing = df_listing.nlargest(30, 'Marcap')
        else:
            df_listing = df_listing.head(30)
    except Exception as e:
        print(f"[Agent] 종목 리스트 에러: {e}", file=sys.stderr)
        return {"error": str(e)}
    
    start_date = (datetime.now() - timedelta(days=days + 60)).strftime('%Y-%m-%d')
    results = []
    
    print(f"[Agent] {len(df_listing)}개 종목 스캔 중...", file=sys.stderr)
    
    for _, row in df_listing.iterrows():
        ticker = row.get('Code', '')
        name = row.get('Name', '')
        
        if not ticker or len(ticker) != 6:
            continue
        
        try:
            df = fdr.DataReader(ticker, start_date)
            
            if df.empty or len(df) < 30:
                continue
            
            # 지표 계산 (조원분 방식)
            ind = calculate_indicators(df)
            
            close = df['Close']
            curr_price = int(close.iloc[-1])
            
            # 수익률
            if len(close) > days:
                price_chg = ((curr_price - close.iloc[-(days+1)]) / close.iloc[-(days+1)]) * 100
            else:
                price_chg = 0
            
            # 거래량 비율
            vol_avg = df['Volume'].iloc[-(days+3):-1].mean()
            vol_ratio = df['Volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1.0
            
            # 변동성
            volatility = close.pct_change().tail(days).std()
            
            # 동적 가중치 스코어링 (조원분 engine.py 로직 그대로)
            score = 50
            score += price_chg * weights.get('price_chg', 0)
            score += (vol_ratio - 1) * weights.get('vol_ratio', 0) * 10
            score += (30 - ind['rsi']) * weights.get('rebound', 0)
            
            # 볼린저밴드 하단이면 반등 보너스
            if ind['bb_pct'] < 0.2:
                score += weights.get('rebound', 0) * 20
            
            # 이격도 낮으면 안정성 보너스
            if abs(ind['ma_gap']) < 0.05:
                score += weights.get('stability', 0) * 30
            
            # MACD 골든크로스면 추세 보너스
            if ind['macd'] > ind['macd_sig']:
                score += weights.get('trend', 0) * 20
            
            # 변동성 가중치
            score += volatility * weights.get('volatility', 0) * 100
            
            # MACD 상태 판별
            macd_status = "상승" if ind['macd'] > ind['macd_sig'] else "조정"
            
            results.append({
                'code': ticker,
                'name': name,
                'price_chg': round(float(price_chg), 2),
                'rsi': ind['rsi'],
                'vol_ratio': round(float(vol_ratio), 2),
                'volatility': round(float(volatility * 100), 2),
                'score': round(float(score), 1),
                'current_price': curr_price,
                'macd_status': macd_status,
                'bb_pct': ind['bb_pct'],
                'ma_gap': round(float(ind['ma_gap'] * 100), 2)
            })
            
        except Exception as e:
            continue
    
    if not results:
        return {"error": "분석 가능한 종목이 없습니다"}
    
    # 점수 기준 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    
    top5 = results[:5]
    candidates = results[5:50]
    
    output = {
        'query': query,
        'days': days,
        'analysisDate': datetime.now().strftime('%Y-%m-%d'),
        'totalScanned': len(results),
        'top5': top5,
        'candidates': candidates,
        'weights': weights
    }
    
    return output

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error": "파라미터 필요"}, cls=NpEncoder))
        sys.exit(1)
    
    arg = sys.argv[1]
    if arg.endswith('.json'):
        with open(arg, 'r', encoding='utf-8') as f:
            params = json.load(f)
    else:
        params = json.loads(arg)
    
    query = params.get('query', '')
    days = params.get('days', 7)
    
    result = run_analysis(query, days)
    print(json.dumps(result, ensure_ascii=False, cls=NpEncoder))
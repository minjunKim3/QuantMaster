import sys
import json
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import os

def run_analysis(query="", days=7):
    
    print(f"[Agent] 분석 시작: {query}", file=sys.stderr)
    
    # 1. 코스피+코스닥 종목 리스트 가져오기
    try:
        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')
        all_stocks = pd.concat([kospi, kosdaq])
    except Exception as e:
        print(f"[Agent] 종목 리스트 에러: {e}", file=sys.stderr)
        return {"error": str(e)}
    
    # 시가총액 상위 300개만 (속도를 위해)
    if 'Marcap' in all_stocks.columns:
        all_stocks = all_stocks.nlargest(30, 'Marcap')
    else:
        all_stocks = all_stocks.head(30)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)
    
    results = []
    
    print(f"[Agent] {len(all_stocks)}개 종목 스캔 중...", file=sys.stderr)
    
    for idx, row in all_stocks.iterrows():
        code = row.get('Code', '')
        name = row.get('Name', '')
        
        if not code or len(code) != 6:
            continue
        
        try:
            df = fdr.DataReader(code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if df.empty or len(df) < days + 14:
                continue
            
            close = df['Close']
            volume = df['Volume']
            
            # 수익률 (최근 N일)
            price_chg = (close.iloc[-1] - close.iloc[-days]) / close.iloc[-days] * 100
            
            # RSI (14일)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain / (loss_val + 1e-10)))
            current_rsi = rsi.iloc[-1]
            
            # 거래량 비율 (최근 5일 평균 / 20일 평균)
            vol_recent = volume.iloc[-5:].mean()
            vol_avg = volume.iloc[-20:].mean()
            vol_ratio = vol_recent / (vol_avg + 1) 
            
            # 변동성 (최근 N일 수익률의 표준편차)
            returns = close.pct_change().iloc[-days:]
            volatility = returns.std() * 100
            
            # 반등 가능성 (RSI가 낮을수록 높음)
            rebound = max(0, (50 - current_rsi) / 50)
            
            # 종합 점수
            score = (
                abs(price_chg) * 1.0 +
                vol_ratio * 20.0 +
                rebound * 50.0 +
                volatility * 10.0
            )
            
            results.append({
                'code': code,
                'name': name,
                'price_chg': round(price_chg, 2),
                'rsi': round(current_rsi, 1),
                'vol_ratio': round(vol_ratio, 2),
                'volatility': round(volatility, 2),
                'score': round(score, 1),
                'current_price': int(close.iloc[-1])
            })
            
        except Exception as e:
            continue
    
    if not results:
        return {"error": "분석 가능한 종목이 없습니다"}
    
    # 점수 기준 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 상위 5개 + 나머지
    top5 = results[:5]
    candidates = results[5:50]
    
    output = {
        'query': query,
        'days': days,
        'analysisDate': datetime.now().strftime('%Y-%m-%d'),
        'totalScanned': len(results),
        'top5': top5,
        'candidates': candidates,
        'weights': {
            'price_chg': 1.0,
            'vol_ratio': 1.0,
            'rebound': 1.0,
            'stability': 0.0,
            'trend': 0.0,
            'volatility': 1.0
        }
    }
    
    return output

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error": "파라미터 필요"}))
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
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
import sys
import json
import pandas as pd

def fetch_stock(code, days=365):
    """종목 코드에 따라 실시간 데이터 가져오기"""
    
    # 한국 주식 판별
    is_kr = code in ['KS11', 'KQ11', 'KS200'] or (code.isdigit() and len(code) == 6)
    
    if is_kr:
        import FinanceDataReader as fdr
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=days)
        df = fdr.DataReader(code, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    else:
        import yfinance as yf
        ticker = yf.Ticker(code)
        df = ticker.history(period=f'{days}d')
    
    if df.empty:
        return {"error": f"데이터 없음: {code}"}
    
    result = []
    for date, row in df.iterrows():
        result.append({
            "date": str(date)[:10],
            "open": round(float(row['Open']), 2),
            "high": round(float(row['High']), 2),
            "low": round(float(row['Low']), 2),
            "close": round(float(row['Close']), 2),
            "volume": int(row['Volume'])
        })
    
    return {"code": code, "data": result}

if __name__ == '__main__':
    code = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    
    sys.stdout.reconfigure(encoding='utf-8')
    result = fetch_stock(code, days)
    print(json.dumps(result, ensure_ascii=False))
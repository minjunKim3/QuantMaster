import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

# ============================================
# 설정
# ============================================
JAVA_SERVER_URL = "http://localhost:8080/api/stock/save-bulk"
SYMBOL = "AAPL"                    # 종목코드
START_DATE = "2025-01-01"          # 시작일
END_DATE = "2025-06-01"            # 종료일
TIMEFRAME = "1d"                   # 봉 단위 (1d=일봉, 1h=1시간봉)

# ============================================
# 1단계: yfinance에서 데이터 가져오기
# ============================================
print(f"[1] {SYMBOL} 데이터 다운로드 중... ({START_DATE} ~ {END_DATE})")
df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, interval=TIMEFRAME)

print(f"    → {len(df)}건 다운로드 완료!")
print(df.head())    # 처음 5줄 확인용

# ============================================
# 2단계: Java 서버에 보낼 JSON 형태로 변환
# ============================================

# 멀티인덱스 해제 (최신 yfinance 대응)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

data_list = []
for index, row in df.iterrows():
    data_list.append({
        "code": SYMBOL,
        "open": round(float(row["Open"]), 2),
        "high": round(float(row["High"]), 2),
        "low": round(float(row["Low"]), 2),
        "close": round(float(row["Close"]), 2),
        "volume": int(row["Volume"]),
        "timeframe": TIMEFRAME,
        "tradeTime": index.strftime("%Y-%m-%dT%H:%M:%S")
    })
# ============================================
# 3단계: Java 서버로 전송
# ============================================
print(f"\n[3] Java 서버로 전송 중... ({JAVA_SERVER_URL})")
try:
    response = requests.post(JAVA_SERVER_URL, json=data_list)
    if response.status_code == 200:
        print(f"    → 성공! 서버 응답: {response.text}")
    else:
        print(f"    → 실패! 상태코드: {response.status_code}")
        print(f"    → 에러 내용: {response.text}")
except Exception as e:
    print(f"    → 연결 실패! Java 서버가 실행 중인지 확인하세요.")
    print(f"    → 에러: {e}")
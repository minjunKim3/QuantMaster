import requests

JAVA_SERVER_URL = "http://localhost:8080/api/backtest"

# ============================================
# 1단계: 더미 백테스트 결과 3건 저장
# ============================================
test_results = [
    {
        "code": "AAPL",
        "timeframe": "1d",
        "strategies": "EMA,RSI,BBB",
        "mainModel": "LSTM",
        "entryThreshold": 2,
        "startDate": "2025-01-01T00:00:00",
        "endDate": "2025-06-01T00:00:00",
        "totalTrades": 48,
        "wins": 30,
        "draws": 3,
        "losses": 15,
        "winRate": 62.5,
        "totalProfitPct": 18.5,
        "maxDrawdownPct": -8.3
    },
    {
        "code": "AAPL",
        "timeframe": "1d",
        "strategies": "EMA,LSTM",
        "mainModel": "LSTM",
        "entryThreshold": 1,
        "startDate": "2025-01-01T00:00:00",
        "endDate": "2025-06-01T00:00:00",
        "totalTrades": 72,
        "wins": 35,
        "draws": 5,
        "losses": 32,
        "winRate": 48.6,
        "totalProfitPct": 5.2,
        "maxDrawdownPct": -15.1
    },
    {
        "code": "AAPL",
        "timeframe": "1d",
        "strategies": "RSI,BBB,TTM",
        "mainModel": "None",
        "entryThreshold": 2,
        "startDate": "2025-01-01T00:00:00",
        "endDate": "2025-06-01T00:00:00",
        "totalTrades": 31,
        "wins": 12,
        "draws": 2,
        "losses": 17,
        "winRate": 38.7,
        "totalProfitPct": -3.1,
        "maxDrawdownPct": -22.4
    }
]

print("=" * 50)
print("[1] 백테스트 결과 저장 테스트")
print("=" * 50)

for i, result in enumerate(test_results):
    response = requests.post(f"{JAVA_SERVER_URL}/save", json=result)
    print(f"  결과 {i+1}: {response.text}")

# ============================================
# 2단계: 저장된 결과 조회 테스트
# ============================================
print("\n" + "=" * 50)
print("[2] AAPL 수익률 TOP 조회")
print("=" * 50)

response = requests.get(f"{JAVA_SERVER_URL}/AAPL/top")
results = response.json()

for r in results:
    print(f"  전략: {r['strategies']:20s} | "
          f"수익률: {r['totalProfitPct']:>7.1f}% | "
          f"승률: {r['winRate']:>5.1f}% | "
          f"MDD: {r['maxDrawdownPct']:>7.1f}%")
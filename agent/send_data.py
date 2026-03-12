import requests
import yfinance as yf

url = "http://localhost:8080/api/save"

ticker_symbol = "005930.KS"



try:
    print(f"[Python] {ticker_symbol}의 현재 주가를 조회합니다...")
    stock = yf.Ticker(ticker_symbol)
    data = stock.history(period="1d")
    current_price = float(data['Close'].iloc[-1])

    print(f"[Market] 현재가 확인: {current_price}원")

    payload = {
    "code": "005930",
    "price": current_price
    }

    print(f"[Send] 자바 서버로 전송 중.. {payload}")
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print(f"성공! 자바의 응답: {response.text}")
    else:
        print(f"실패... 코드: {response.status_code}")

except Exception as e:
    print(f"서버 에러 발생: {e}")
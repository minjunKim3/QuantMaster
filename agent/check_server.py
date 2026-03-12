import requests

url = "http://localhost:8080/test"

try:
    print("여기는 파이썬, 자바에 신호 보내는중..")
    response = requests.get(url)

    if response.status_code == 200:
        print(f"[success] 응답 도착: {response.text}")
    else:
        print(f"[Error] 응답은 도착했으나 에러 발생.. 상태 코드. {response.status_code}")
except Exception as e:
    print(f"[Fail] 연결에 완전히 실패. 서버 ON 여부 확인 바람: {e}")
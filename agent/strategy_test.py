import requests
import json

URL = "http://localhost:8080/api/strategy"

# ============================================
# 1. Create: 전략 설정 2개 생성
# ============================================
print("=" * 55)
print("[1] 전략 설정 생성")
print("=" * 55)

config1 = {
    "name": "공격형 전략",
    "activeStrategies": "EMA,RSI,BBB,LSTM,DWTM",
    "mainModel": "LSTM",
    "entryThreshold": 1,
    "parameters": json.dumps({
        "EMA_fast_period": 10,
        "EMA_slow_period": 30,
        "RSI_period": 14
    })
}

config2 = {
    "name": "안정형 전략",
    "activeStrategies": "RSI,BBB",
    "mainModel": "None",
    "entryThreshold": 2,
    "parameters": json.dumps({
        "RSI_period": 23,
        "BBB_window": 30
    })
}


r1 = requests.post(URL, json=config1)
print(f"  상태코드: {r1.status_code}")    # ← 이거 추가
print(f"  응답내용: {r1.text}")            # ← 이거 추가
print(f"  공격형: id={r1.json()['id']}, 생성 완료!")  # 기존 줄

r2 = requests.post(URL, json=config2)
print(f"  안정형: id={r2.json()['id']}, 생성 완료!")

id1 = r1.json()["id"]
id2 = r2.json()["id"]

# ============================================
# 2. Read: 전체 목록 조회
# ============================================
print("\n" + "=" * 55)
print("[2] 전체 설정 목록")
print("=" * 55)

r = requests.get(URL)
for c in r.json():
    print(f"  id={c['id']} | {c['name']:10s} | "
          f"전략: {c['activeStrategies']:25s} | "
          f"활성: {c['isActive']}")

# ============================================
# 3. Update: 공격형 설정 수정
# ============================================
print("\n" + "=" * 55)
print("[3] 공격형 전략 수정 (TTM 추가, threshold 2로 변경)")
print("=" * 55)

update_data = {
    "activeStrategies": "EMA,RSI,BBB,TTM,LSTM,DWTM",
    "mainModel": "LSTM",
    "entryThreshold": 2,
    "parameters": json.dumps({
        "EMA_fast_period": 10,
        "EMA_slow_period": 30,
        "RSI_period": 14,
        "TTM_bb_length": 27
    })
}

r = requests.put(f"{URL}/{id1}", json=update_data)
print(f"  수정 완료! 전략: {r.json()['activeStrategies']}")

# ============================================
# 4. Activate: 공격형 활성화
# ============================================
print("\n" + "=" * 55)
print("[4] 공격형 전략 활성화")
print("=" * 55)

r = requests.put(f"{URL}/{id1}/activate")
print(f"  활성화 완료! isActive={r.json()['isActive']}")

# ============================================
# 5. Read: 현재 활성 설정 조회
# ============================================
print("\n" + "=" * 55)
print("[5] 현재 활성 설정 조회")
print("=" * 55)

r = requests.get(f"{URL}/active")
c = r.json()
print(f"  이름: {c['name']}")
print(f"  전략: {c['activeStrategies']}")
print(f"  메인: {c['mainModel']}")
print(f"  threshold: {c['entryThreshold']}")

# ============================================
# 6. Delete: 안정형 삭제
# ============================================
print("\n" + "=" * 55)
print("[6] 안정형 전략 삭제")
print("=" * 55)

r = requests.delete(f"{URL}/{id2}")
print(f"  {r.text}")

# ============================================
# 7. Read: 최종 목록 확인
# ============================================
print("\n" + "=" * 55)
print("[7] 최종 설정 목록")
print("=" * 55)

r = requests.get(URL)
for c in r.json():
    print(f"  id={c['id']} | {c['name']:10s} | "
          f"전략: {c['activeStrategies']:25s} | "
          f"활성: {c['isActive']}")

print("\n CRUD 테스트 완료!")
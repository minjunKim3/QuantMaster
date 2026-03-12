# QuantMaster

**퀀트 투자 백테스트 & 모의투자 시스템**

실제 주식 데이터(yfinance)를 기반으로 투자 전략을 적용해 백테스팅, 모의투자를 할 수 있는 웹 앱입니다.

---

## 주요 기능

### 백테스트

과거 주가 데이터로 투자 전략의 성능을 평가합니다.
- 수익률, 승률, 최대 낙폭(MDD) 계산
- 매매 포인트 차트 시각화
- 결과를 DB에 저장 및 전략 간 성능 비교

### 모의투자(미완)
초기 자금 설정 후 실제 투자 체험
- 매일 자산 변화 추이 차트
- 자산 변화 확인

### 앙상블(Threshold)
여러 전략 활성화 후, 설정한 Threshold 이상만큼 동의를 할때만 매매를 실행합니다.

---

## 기술 스택

| 영역      | 기술 |
|---------|------|
| Backend | Java 17, Spring Boot 3, JPA/Hibernate|
| Data Engine | Python 3, yfinance, pandas, numpy|
|Database| MySQL 8.0 |
|Frontend| HTML/CSS/JavaScript, Chart.js|
|연동 방식| java <-> python (ProcessBulider + JSON)|

### 투자 전략(현재는 8종)

| 전략 | 설명 |
|-----|------|
|EMA| EMA 크로스오버 (골든/데드 크로스)|
|EMA3| Triple EMA 정배열 전략|
|RSI| 과매도/과매수 반전 전략|
|MACD| MACD-시그널 교차 전략|
|BBB| 볼린저 밴드 이탈/복귀 전략|
|TTM| TTM squeeze 모멘텀 전략|
|SUT| SuperTrend 추세 추적 전략 |
|PSAR | Parabolic SAR 반전 전략 | 

---

## 프로젝트 구조
```
QuantMaster/
QuantMaster/
├── src/main/java/com/quant/server/
│   ├── controller/     — API 엔드포인트
│   ├── service/        — 비즈니스 로직
│   ├── repository/     — DB 접근
│   ├── domain/         — 엔티티
│   ├── dto/            — 응답 래퍼
│   └── exception/      — 전역 예외 처리
├── src/main/resources/
│   ├── static/index.html   — 대시보드 UI
│   └── application.properties
├── agent/
│   ├── backtest_runner.py  — 백테스트/모의투자 엔진
│   ├── stock_collector.py  — 주가 데이터 수집
│   └── requirements.txt
└── pom.xml
```

---

## 설치 및 실행

### 사전 준비 사항
- Java 17 이상
- Python 3.9 이상
- MySQL 8.0
- Maven

### 1. 프로젝트 클론
```bash
   git clone https://github.com/minjunKim3/QuantMaster.git
   cd QuantMaster
```

### DB 생성
```sql
   CREATE DATABASE quantmaster;
```

### 3. Python 환경 설정
```bash
   cd agent
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
```

### 4. application.properties 수정
```properties
spring.datasource.password=본인의비밀번호
collector.python-path=본인의경로/agent/venv/Scripts/python.exe
backtest.script-path=본인의경로/agent/backtest_runner.py
```

### 5. 서버 실행
```bash
   mvn spring-boot:run
```

### 6. 접속
브라우저에서 `http://localhost:8080` 접속

---

## API 엔드포인트

|Method| EndPoint                    | 설명|
|------|-----------------------------|----|
|GET | /api/stock/{code}           | 주식 데이터 조회|
|GET | /api/strategy               |전략 목록 조회|
|POST| /api/strategy               |새 전략 생성 |
|PUT | /api/strategy/{id}/activate | 전략 활성화 |
|DELETE| /api/strategy/{id}          | 전략 삭제|
|POST| /api/backtest/run           | 백테스트 실행|
|POST| /api/backtest/simulation | 모의투자 실행|
|GET| /api/backtest/{code}/top| 백테스트 결과 TOP 조회|

---

## 구성

| 역할 | 담당 |
|-----|-----|
| Backend (Java/Spring Boot) | API 서버, DB 설계, 프론트엔드, Java-Python 연동|
|Data/ML (Python) | 투자 전략 개발, 백테스트 엔진, AI/ML 모델 |

---
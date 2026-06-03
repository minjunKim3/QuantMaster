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

### 4. 환경 설정

20260602 정리: 모든 경로/비번이 `${VAR:default}` 패턴이라 **보통은 DB 비번만** 채우면 됩니다.
세 가지 방법 중 편한 걸 골라요:

**A. 환경변수 (가장 간단)**
```bash
# bash/zsh
export DB_PASSWORD=본인의비밀번호
# PowerShell
$env:DB_PASSWORD = "본인의비밀번호"
```
또는 `.env.example` 을 `.env` 로 복사 후 값 채우기 (gitignore 처리됨).

**B. `application-local.properties` (오버라이드 파일)**
```bash
cp src/main/resources/application-local.properties.example src/main/resources/application-local.properties
# 파일에 spring.datasource.password=본인의비밀번호 등을 채우기
```
이 파일은 .gitignore 등록되어 있어 커밋되지 않습니다.

**C. 절대경로 강제 (다른 디렉토리에서 실행할 때)**
```properties
# application-local.properties 안에
collector.python-path=/abs/path/agent/venv/Scripts/python.exe
backtest.script-path=/abs/path/agent/backtest_runner.py
manager.python-path=/abs/path/agent/AItr_manager/venv/Scripts/python.exe
```

> Linux/Mac venv 사용 시 PYTHON_PATH 를 `agent/venv/bin/python` 으로.

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
|POST| /api/backtest/predict | AI 예측 (modelId/forceTrain 옵션) |
|POST| /api/backtest/verify | 모델 성능 검증 (방향정확도) |
|GET| /api/backtest/has-model?code=XXX | 종목 V6 모델 존재 여부 |
|GET| /api/backtest/models?code=XXX | 종목별 학습 모델 리스트 |
|DELETE| /api/backtest/model?code=XXX&modelId=YYY | 특정 모델 삭제 |
|POST| /api/recommend/run | 자연어 종목 추천 (AItr_manager + gguf LLM) |

---

## AI 모델 아키텍처 (V6)

QuantMaster V6는 종목별 자동 학습 + 동적 게이팅 calibration을 지원합니다.

### 학습 파이프라인 (`agent/lstm_train_v6.py`)
1. **Foundation Models** — AutoGluon Chronos + DeepAR + ADIDA 앙상블 예측을 LSTM 입력 feature로 활용
2. **LSTM 본체** — 128 hidden × 2 layers, BatchNorm + GELU, time budget 90s + plateau 감지로 조기 종료
3. **Joint Calibration** — Entry/Exit 통과율 grid search (평균 ~62.5% 제약), 가중 DA 최대화
4. **종목당 다중 모델 저장** — `{code}_{YYYYMMDD_HHmmss}_*` 형식으로 timestamp 기반 modelId 관리

### 예측 메타 (meta.json)
- `gateEntryThrRaw`, `gateExitThrRaw` — 종목별 동적 게이팅 임계값
- `gateEntryTrainRatio`, `gateExitTrainRatio` — 학습 시 통과율 (UI의 게이팅 컷 = 1 - 통과율)
- `confidence_abs_max_dist`, `confidence_abs_min_dist` — percentile 자신감 계산용 학습 분포
- `gatedWeightedDaPct` — 게이팅 적용 통합 방향정확도

### UI (AI 예측 탭)
- 종목 입력 → KRX 2,880개 자동완성 + 모델 리스트 자동 fetch
- 라디오 선택 → "🚀 이 모델로 예측" (~5초) / "🆕 새 모델 생성 + 예측" (~3분)
- 결과 화면 — 방향 / 자신감 / **게이팅 컷 vs 자신감** 비교 (✓ 발동 또는 ✗ 죽은 신호)

---

## 구성

| 역할 | 담당 |
|-----|-----|
| Backend (Java/Spring Boot) | API 서버, DB 설계, 프론트엔드, Java-Python 연동|
|Data/ML (Python) | 투자 전략 개발, 백테스트 엔진, AI/ML 모델 |

---

## 배포

### 현재 상태
- ✅ Dockerfile 멀티스테이지 작성 완료 (Java 17 + Python 3.12)
- ❌ Railway 무료 티어 배포 보류 (메모리 한계)
- ✅ 로컬 실행: `mvn spring-boot:run`

### Railway 배포 미실행 사유
- 핵심 AI Agent 기능이 5GB gemma gguf 모델 요구
- 무료 티어 512MB로는 Java(300MB) + Python venv(800MB+) 수용 불가
- 유료 티어(8GB+) 전환 시 즉시 배포 가능 상태

### 로컬 실행 가이드
1. MySQL 8.0 설치 + `quantmaster` DB 생성
2. Python 3.12 venv 생성: `cd agent && python -m venv venv`
3. 의존성 설치: `pip install -r requirements.txt`
4. gguf 모델 다운로드 (별도): `gemma-4-E4B-it-Q4_1.gguf` → `agent/AItr_manager/`
5. `application.properties` 수정 (DB 비밀번호, 경로)
6. `mvn spring-boot:run`

---
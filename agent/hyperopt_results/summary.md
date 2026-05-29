# V5 LSTM Hyperopt — 5종목 일괄 결과

- 생성: 2026-05-14 14:02:04
- 종목 수: 5 / 요청 5
- 종목당 trial: 100
- 학습 SEED: 42 (고정) | Sampler seed: SEED + ticker_idx (종목별 분리)
- λ(penalty): 0.001 | min_trades/year: 10
- 총 소요: 633.4초 (10.56분)

## 전체 요약

| 종목 | Val DA | Val 신호 | Test DA | Test 신호 | Δ갭 | conf_exit | conf_mult | bb_high | bb_low | rsi_high | 시간(초) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KS11 | 76.62% | 77 | 48.36% | 122 | +28.26p | 0.4293 | 1817.3 | 0.5061 | -0.7274 | 63 | 146.4 |
| KQ11 | 80.00% | 20 | 0.00% | 0 | +80.00p | 0.0668 | 100.3 | 0.9668 | -0.5041 | 73 | 160.8 |
| 005930 | 66.67% | 3 | 41.38% | 29 | +25.29p | 0.6615 | 1427.8 | 0.8049 | -0.5561 | 56 | 108.5 |
| 000660 | 100.00% | 7 | 44.68% | 47 | +55.32p | 0.4886 | 208.1 | 0.6318 | -0.8328 | 55 | 109.0 |
| 035720 | 35.56% | 45 | 45.95% | 37 | -10.39p | 0.6681 | 1094.2 | 0.9922 | -0.7942 | 80 | 108.1 |

## 평균

- 5종목 **Val** DA 평균: **71.77%**
- 5종목 **Test** (hold-out) DA 평균: **36.07%**
- Val→Test 갭: **+35.70p** (>10p면 과적합 신호)

## 종목별 상세

### KS11

- Sampler seed: 42
- Best trial: #80 (score=76.6234)
- Systematic bias: -0.001047
- BB 임계값 (우리 단위): high=0.753, low=0.1363
- Val: DA 76.62% / 신호 77개 (min 11) / penalty 0
- Test (hold-out): DA 48.36% / 신호 122개 (min 11)
- 시간: 준비 146.0s + Optuna 0.4s = 146.4s

### KQ11

- Sampler seed: 43
- Best trial: #74 (score=80.0000)
- Systematic bias: -0.001912
- BB 임계값 (우리 단위): high=0.9834, low=0.2479
- Val: DA 80.00% / 신호 20개 (min 11) / penalty 0
- Test (hold-out): DA 0.00% / 신호 0개 (min 11)
- 시간: 준비 160.4s + Optuna 0.4s = 160.8s

### 005930

- Sampler seed: 44
- Best trial: #49 (score=66.6027)
- Systematic bias: +0.000414
- BB 임계값 (우리 단위): high=0.9025, low=0.222
- Val: DA 66.67% / 신호 3개 (min 11) / penalty 64
- Test (hold-out): DA 41.38% / 신호 29개 (min 11)
- 시간: 준비 108.1s + Optuna 0.4s = 108.5s

### 000660

- Sampler seed: 45
- Best trial: #93 (score=99.9840)
- Systematic bias: -0.000520
- BB 임계값 (우리 단위): high=0.8159, low=0.0836
- Val: DA 100.00% / 신호 7개 (min 11) / penalty 16
- Test (hold-out): DA 44.68% / 신호 47개 (min 11)
- 시간: 준비 108.6s + Optuna 0.4s = 109.0s

### 035720

- Sampler seed: 46
- Best trial: #84 (score=35.5556)
- Systematic bias: +0.001355
- BB 임계값 (우리 단위): high=0.9961, low=0.1029
- Val: DA 35.56% / 신호 45개 (min 11) / penalty 0
- Test (hold-out): DA 45.95% / 신호 37개 (min 11)
- 시간: 준비 107.7s + Optuna 0.4s = 108.1s

## 산출물

- `KS11_best_params.json` / `KS11_trials_log.csv`
- `KQ11_best_params.json` / `KQ11_trials_log.csv`
- `005930_best_params.json` / `005930_trials_log.csv`
- `000660_best_params.json` / `000660_trials_log.csv`
- `035720_best_params.json` / `035720_trials_log.csv`

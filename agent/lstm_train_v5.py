import sys
# 20260602: cp949 콘솔(Windows 한국어 기본)에서 ⏱·— 등 비-CP949 문자가
# print 시 'cp949' codec can't encode … 로 터지는 문제 차단.
# lstm_service.py 가 in-process 로 import 해서 train_model 을 호출할 때
# stdout 이 아직 cp949 상태라 PhaseTimer.report() 가 깨졌음.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import FinanceDataReader as fdr
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import pywt
import copy
import time
import os
import pickle
import yfinance as yf

# AutoGluon — Chronos, DeepAR 등을 한번에 쓸 수 있는 프레임워크
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# ============================================
# 재현성: 시드 고정 (random / numpy / torch / DataLoader worker)
# ============================================
SEED = 42

# ============================================
# 20260602: 구간별 소요시간 측정 — 새 종목 학습이 10분 가까이 걸리는 원인 추적용.
# 목표는 종목당 ≤2분. 어느 구간이 잡아먹는지 확인 → 핀포인트 튜닝.
# 사용법: pt = PhaseTimer(); with pt.phase("name"): ...  / pt.report()
# 중첩(sub-phase)도 지원: with pt.phase("autogluon", parent=None): / with pt.phase("rolling_chronos", parent="autogluon")
# ============================================
class PhaseTimer:
    def __init__(self):
        self.records = []  # [(name, parent, sec)]
        self._stack = []

    class _Ctx:
        def __init__(self, owner, name, parent):
            self.o = owner; self.n = name; self.p = parent; self.t0 = 0.0
        def __enter__(self):
            self.t0 = time.time(); self.o._stack.append(self.n); return self
        def __exit__(self, *a):
            self.o.records.append((self.n, self.p, time.time() - self.t0))
            self.o._stack.pop()

    def phase(self, name, parent=None):
        return PhaseTimer._Ctx(self, name, parent)

    def add(self, name, sec, parent=None):
        self.records.append((name, parent, sec))

    def report(self, header):
        total = sum(sec for n, p, sec in self.records if p is None)
        print(f"\n{'='*60}")
        print(f"  ⏱  TIMING BREAKDOWN — {header}")
        print(f"{'='*60}")
        for name, parent, sec in self.records:
            pct = (sec / total * 100) if (parent is None and total > 0) else None
            indent = "    " if parent else "  "
            tag = f"({pct:5.1f}%)" if pct is not None else "       "
            print(f"  {indent}{name:<28} {sec:>7.1f}s {tag}")
        print(f"  {'-'*55}")
        print(f"  TOTAL (top-level sum)         {total:>7.1f}s ({total/60:.1f}분)")
        print(f"{'='*60}")
        return total


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 모듈 로드 시 1차 고정
set_seed(SEED)

# ============================================
# 모델 정의 (V3와 동일)
# ============================================
class StockLSTM_V5(nn.Module):  # MODIFIED FROM V4: 클래스명 V4 → V5 (내부 구조 동일)
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 3)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d,)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

# MODIFIED [Multi-Filter]: 다중 필터 임계값 (ETH-튜닝값을 우리 BB 공식 [0,1]로 변환)
# 변환식: our_bb = (eth_bb + 1) / 2  — 우리 bb_position = (close - lower) / (upper - lower)
# eth -0.43 → 0.285 / eth 0.02 → 0.51 / eth 1.16 → 1.08 / eth -0.77 → 0.115
RSI_ENTRY_LOW = 38.0
BB_ENTRY_LOW = 0.285
RSI_ENTRY_HIGH = 31.0
BB_ENTRY_HIGH = 0.51
RSI_EXIT_HIGH = 65.0
BB_EXIT_HIGH = 1.08
BB_EXIT_LOW = 0.115


def huber_loss(pred, target, delta=0.1):
    error = pred - target
    is_small = torch.abs(error) <= delta
    small_loss = 0.5 * error**2
    large_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small, small_loss, large_loss).mean()

# MODIFIED FROM V4: 방향 가중 Huber 손실 함수 추가 (예측/실제 부호 불일치시 가중)
def directional_huber_loss(pred, target, delta=0.01, direction_weight=3.0):
    error = pred - target
    is_small = torch.abs(error) <= delta
    small_loss = 0.5 * error**2
    large_loss = delta * (torch.abs(error) - 0.5 * delta)
    huber = torch.where(is_small, small_loss, large_loss)

    pred_sign = torch.sign(pred)
    target_sign = torch.sign(target)
    direction_mismatch = (pred_sign != target_sign).float()

    weighted = huber * (1.0 + direction_mismatch * (direction_weight - 1.0))
    return weighted.mean()

# ============================================
# DWT 웨이블릿 (V3와 동일)
# ============================================
def add_dwt_features(close_prices, window_size=128):
    n = len(close_prices)
    dwt_trend = np.zeros(n)
    dwt_high = np.zeros(n)
    dwt_low = np.zeros(n)
    dwt_energy = np.zeros(n)
    for i in range(window_size, n):
        window = close_prices[i - window_size:i]
        try:
            coeffs = pywt.wavedec(window, 'db4', level=2)
            dwt_trend[i] = coeffs[0][-1]
            dwt_high[i] = coeffs[1][-1]
            dwt_low[i] = coeffs[2][-1]
            dwt_energy[i] = np.sum(coeffs[1]**2) + np.sum(coeffs[2]**2)
        except Exception:
            pass
    return dwt_trend, dwt_high, dwt_low, dwt_energy

# ============================================
# 외부 지표 (V3와 동일)
# ============================================
def get_external_data(start_date, end_date, timer=None):
    externals = {}
    symbols = {'vix': '^VIX', 'gold': 'GC=F', 'dxy': 'DX-Y.NYB'}
    for name, ticker in symbols.items():
        t0 = time.time()
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                externals[name] = data['Close']
        except Exception:
            pass
        if timer is not None:
            timer.add(f"yf_{name}", time.time() - t0, parent="external")
    return externals

# ============================================
# [핵심 추가] Foundation Model 예측 생성
#
# AutoGluon이 하는 일:
# 1. 데이터를 TimeSeriesDataFrame 형태로 변환
# 2. Chronos(Amazon) + DeepAR + ADIDA 모델을 학습
# 3. 각 모델이 "다음 1일 종가"를 예측
# 4. 이 예측값을 LSTM의 추가 특성으로 사용
#
# 조원분 코드에서는 이걸 rolling window로 매일 예측했는데
# 학습 시간이 엄청 오래 걸려 (수십 분~수 시간)
# 여기서는 간소화해서 일괄 예측 방식으로 처리
# ============================================
def generate_foundation_predictions(df, code, close_mean, model_dir, timer=None):
    """Chronos + DeepAR + ADIDA 예측 생성"""

    print("  [Foundation] AutoGluon 모델 학습 시작...")
    print("  이 과정은 5~15분 걸릴 수 있습니다.")

    close_normalized = df['Close'].values / close_mean

    # TimeSeriesDataFrame 생성
    ts_df = pd.DataFrame({
        'item_id': 'STOCK',
        'timestamp': df.index.tz_localize(None) if df.index.tz else df.index,
        'target': close_normalized
    })
    ts_df = TimeSeriesDataFrame(ts_df)

    # 모델 저장 경로
    safe_code = code.replace('^', '').replace('/', '_')
    ag_path = os.path.join(model_dir, f'{safe_code}_autogluon')

    # 학습 또는 로드
    t_loadfit = time.time()
    loaded_existing = False
    if os.path.exists(ag_path):
        print("  [Foundation] 기존 AutoGluon 모델 로드 중...")
        try:
            predictor = TimeSeriesPredictor.load(ag_path)
            loaded_existing = True
            print("  [Foundation] 기존 모델 로드 성공!")
        except Exception:
            print("  [Foundation] 로드 실패, 새로 학습...")
            predictor = None
    else:
        predictor = None

    if predictor is None:
        predictor = TimeSeriesPredictor(
            prediction_length=1,
            cache_predictions=False,
            log_to_file=False,
            path=ag_path,
            freq='D'
        )

        predictor.fit(
            ts_df,
            enable_ensemble=False,
            hyperparameters={
                'DeepAR': {},
                'Chronos': {"model_path": "amazon/chronos-bolt-small"},
                'ADIDA': {}
            },
            time_limit=600
        )
        predictor.save()
        print("  [Foundation] AutoGluon 학습 완료!")
    if timer is not None:
        label = "ag_load" if loaded_existing else "ag_fit"
        timer.add(label, time.time() - t_loadfit, parent="autogluon")

    # Rolling 예측 — 각 시점에서 다음 1일 예측
    MODEL_NAMES = {
        'DeepAR': 'deepar',
        'Chronos[amazon__chronos-bolt-small]': 'chronos',
        'ADIDA': 'adida'
    }

    # 결과 저장용
    predictions = {name: np.full(len(df), np.nan) for name in MODEL_NAMES.values()}

    context_len = 60  # 예측에 사용할 과거 데이터 길이
    batch_size = 256

    for model_name, col_name in MODEL_NAMES.items():
        print(f"  [Foundation] {model_name} 예측 중...")
        t_model = time.time()

        for batch_start in range(context_len, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))

            batch_ts_list = []
            for i in range(batch_start, batch_end):
                window_data = df.iloc[i - context_len:i]
                ts_window = pd.DataFrame({
                    'item_id': f'DATA_{i}',
                    'timestamp': window_data.index.tz_localize(None) if window_data.index.tz else window_data.index,
                    'target': window_data['Close'].values / close_mean
                })
                batch_ts_list.append(ts_window)

            batch_ts = pd.concat(batch_ts_list, ignore_index=True)

            try:
                ts_data = TimeSeriesDataFrame.from_data_frame(
                    batch_ts, id_column='item_id', timestamp_column='timestamp'
                )
                pred = predictor.predict(ts_data, model=model_name)

                for idx in range(batch_start, batch_end):
                    item_id = f'DATA_{idx}'
                    try:
                        predictions[col_name][idx] = pred.loc[item_id]['mean'].iloc[0]
                    except Exception:
                        pass
            except Exception as e:
                print(f"    배치 {batch_start}-{batch_end} 실패: {e}")
                continue

        if timer is not None:
            timer.add(f"rolling_{col_name}", time.time() - t_model, parent="autogluon")
        valid_count = np.isfinite(predictions[col_name]).sum()
        print(f"  [Foundation] {model_name} 완료: {valid_count}개 예측")

    return predictions

# ============================================
# 특성 생성 (V3 + Foundation 예측)
# ============================================
def create_features_v4(df, external_data=None, foundation_preds=None):
    close = df['Close'].values
    volume = df['Volume'].values

    features = pd.DataFrame(index=df.index)

    # 기본 특성 (V3와 동일)
    features['close'] = close
    features['ma5'] = pd.Series(close).rolling(5).mean().values
    features['ma20'] = pd.Series(close).rolling(20).mean().values

    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi'] = (100 - (100 / (1 + gain / (loss_val + 1e-10)))).values
    features['change'] = pd.Series(close).pct_change().values
    features['volume_norm'] = volume / (np.max(volume) + 1e-10)

    ma20_s = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()
    bb_upper = ma20_s + 2 * std20
    bb_lower = ma20_s - 2 * std20
    features['bb_position'] = ((close - bb_lower) / (bb_upper - bb_lower + 1e-8)).values

    exp12 = pd.Series(close).ewm(span=12).mean()
    exp26 = pd.Series(close).ewm(span=26).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9).mean()
    features['macd_hist'] = (macd - signal).values

    high = df['High'].values if 'High' in df.columns else close
    low = df['Low'].values if 'Low' in df.columns else close
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    features['atr'] = pd.Series(tr).rolling(14).mean().values

    # DWT 웨이블릿
    dwt_trend, dwt_high, dwt_low, dwt_energy = add_dwt_features(close, min(128, len(close) // 4))
    features['dwt_trend'] = dwt_trend
    features['dwt_high'] = dwt_high
    features['dwt_low'] = dwt_low
    features['dwt_energy'] = dwt_energy

    # 외부 지표
    if external_data:
        for name, series in external_data.items():
            aligned = series.reindex(df.index, method='ffill')
            features[f'{name}_norm'] = (aligned / aligned.mean()).values

    # [핵심 추가] Foundation Model 예측값
    if foundation_preds:
        for name, vals in foundation_preds.items():
            features[f'fm_{name}'] = vals

    features = features.ffill().bfill().fillna(0)
    return features

# ============================================
# 학습 함수
# ============================================
def train_model(code, start_date='2015-01-01', timer=None):
    set_seed(SEED)  # 재현성: 종목별 학습 시작마다 시드 재고정
    print(f"\n{'='*60}")
    print(f"LSTM V5 학습: {code} (Foundation Model 앙상블 + 수익률 타겟)")  # MODIFIED FROM V4: 로그 V4 → V5 + 수익률 표기
    print(f"{'='*60}")

    # 20260602: 종목별 구간 측정 — 인자로 받지 않으면 로컬에서 직접 생성
    own_timer = timer is None
    if own_timer:
        timer = PhaseTimer()

    # 1. 데이터
    print(f"\n[1] 데이터 다운로드: {code}")
    with timer.phase("data_download"):
        df = fdr.DataReader(code, start_date)
    if df.empty or len(df) < 200:
        print(f"  데이터 부족! ({len(df)}일)")
        return False
    print(f"  데이터: {len(df)}일")

    close_mean = df['Close'].mean()
    volume_mean = df['Volume'].mean()

    # 2. Foundation Model 예측
    print(f"\n[2] Foundation Model 예측 생성")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models_v5')  # MODIFIED FROM V4: models_v4 → models_v5
    os.makedirs(model_dir, exist_ok=True)

    with timer.phase("autogluon"):
        foundation_preds = generate_foundation_predictions(df, code, close_mean, model_dir, timer=timer)

    # 3. 외부 지표
    print(f"\n[3] 외부 지표 로드")
    with timer.phase("external"):
        external_data = get_external_data(start_date, df.index[-1].strftime('%Y-%m-%d'), timer=timer)

    # 4. 특성 생성
    print(f"\n[4] 특성 생성")
    with timer.phase("features"):
        features = create_features_v4(df, external_data, foundation_preds)
    print(f"  특성 수: {len(features.columns)}개")
    print(f"  특성 목록: {list(features.columns)}")

    # 5. 정규화
    with timer.phase("lstm_prep"):
        scaler = RobustScaler()
        target_scaler = RobustScaler()

        feature_values = features.values
        scaled = scaler.fit_transform(feature_values)

        # 6. 윈도우 + 타겟
        WINDOW = 60
        close_idx = list(features.columns).index('close')
        close_raw = df['Close'].values  # MODIFIED FROM V4: 정규화 전 raw close 사용 (수익률 계산용)

        X, y = [], []
        for i in range(WINDOW, len(scaled) - 3):
            X.append(scaled[i - WINDOW:i])
            # MODIFIED FROM V4: 타겟을 가격(정규화)에서 수익률((p_future - p_now) / p_now)로 변경
            p_now = close_raw[i]
            y.append([
                (close_raw[i + 1] - p_now) / p_now,
                (close_raw[i + 2] - p_now) / p_now,
                (close_raw[i + 3] - p_now) / p_now,
            ])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        y_scaled = target_scaler.fit_transform(y)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_scaled[:split], y_scaled[split:]
        y_test_raw = y[split:]

        print(f"  학습: {len(X_train)}개 | 테스트: {len(X_test)}개")

    # 7. 학습
    print(f"\n[5] 학습 시작")
    lstm_train_ctx = timer.phase("lstm_train")
    lstm_train_ctx.__enter__()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockLSTM_V5(input_size=X.shape[-1]).to(device)  # MODIFIED FROM V4: StockLSTM_V4 → StockLSTM_V5

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(device), torch.FloatTensor(y_train).to(device))
    # 재현성: shuffle 순서 고정용 generator + worker_init_fn
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        worker_init_fn=_seed_worker, generator=g
    )
    val_dataset = TensorDataset(torch.FloatTensor(X_test).to(device), torch.FloatTensor(y_test).to(device))
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        worker_init_fn=_seed_worker
    )

    EPOCHS = 100
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = directional_huber_loss(pred, batch_y, delta=0.01, direction_weight=3.0)  # MODIFIED FROM V4: huber_loss → directional_huber_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                loss = directional_huber_loss(pred, batch_y, delta=0.01, direction_weight=3.0)  # MODIFIED FROM V4: huber_loss → directional_huber_loss
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | {elapsed:.0f}초")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.time() - start_time
    print(f"  학습 완료! {elapsed:.0f}초 ({elapsed/60:.1f}분)")
    lstm_train_ctx.__exit__(None, None, None)

    # 8. 평가
    print(f"\n[6] 평가")
    lstm_eval_ctx = timer.phase("lstm_eval")
    lstm_eval_ctx.__enter__()
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    pred_raw = target_scaler.inverse_transform(pred_scaled)

    # MODIFIED [Systematic Bias]: 학습된 모델의 train set 예측 평균과 정답 평균 차이를 측정
    # → 모든 예측에 더해서 모델의 일관된 편향을 보정 (raw 수익률 단위, 스칼라 1개)
    # 조원분 코드: systematic_bias = y_train.mean() - model.predict(X_train).mean()
    with torch.no_grad():
        pred_train_scaled = model(torch.FloatTensor(X_train).to(device)).cpu().numpy()
    pred_train_raw = target_scaler.inverse_transform(pred_train_scaled)
    y_train_raw_full = target_scaler.inverse_transform(y_train)  # raw 수익률로 복원
    systematic_bias = float(y_train_raw_full.mean() - pred_train_raw.mean())
    print(f"  [Systematic Bias] mean(actual)={y_train_raw_full.mean():.6f} | mean(pred)={pred_train_raw.mean():.6f} | bias={systematic_bias:+.6f}")

    # MODIFIED [Systematic Bias]: bias 보정 후 예측값으로 모든 후속 메트릭 재계산
    pred_raw_uncorrected = pred_raw.copy()
    pred_raw = pred_raw + systematic_bias  # 모든 step에 동일하게 더함 (broadcast)

    for step in range(3):
        # MODIFIED FROM V4: 타겟이 수익률이므로 가격 복원시 (1+return)*close_mean이 아닌 수익률 자체로 평가
        actual = y_test_raw[:, step]
        predicted = pred_raw[:, step]
        # 수익률 → 가격으로 변환해 MAPE/RMSE 계산 (절대값 비교)
        actual_price = (1 + actual) * close_mean
        predicted_price = (1 + predicted) * close_mean
        mape = np.mean(np.abs((actual_price - predicted_price) / actual_price)) * 100
        rmse = np.sqrt(mean_squared_error(actual_price, predicted_price))
        print(f"  t+{step+1}: MAPE {mape:.2f}% | RMSE {rmse:.2f}")
        # MODIFIED FROM V4: Direction Accuracy 추가 (수익률 부호 일치율)
        direction_correct = np.sign(y_test_raw[:, step]) == np.sign(pred_raw[:, step])
        direction_acc = direction_correct.mean() * 100
        print(f"  t+{step+1}: Direction Accuracy {direction_acc:.2f}%")

    # MODIFIED [Max/Min Trick]: 3-step 예측을 max/min으로 압축한 신호의 방향정확도
    # Entry = sign(max(pred)) vs sign(max(actual)), Exit = sign(min(pred)) vs sign(min(actual))
    pred_mean = pred_raw.mean(axis=1)
    actual_mean = y_test_raw.mean(axis=1)
    da_mean_comp = (np.sign(pred_mean) == np.sign(actual_mean)).mean() * 100
    pred_max = pred_raw.max(axis=1)
    actual_max = y_test_raw.max(axis=1)
    da_entry = (np.sign(pred_max) == np.sign(actual_max)).mean() * 100
    pred_min = pred_raw.min(axis=1)
    actual_min = y_test_raw.min(axis=1)
    da_exit = (np.sign(pred_min) == np.sign(actual_min)).mean() * 100
    da_max_min_avg = (da_entry + da_exit) / 2
    print(f"  [BEFORE/mean comp] DA: {da_mean_comp:.2f}%")
    print(f"  [AFTER/max-min   ] Entry DA: {da_entry:.2f}% | Exit DA: {da_exit:.2f}% | 평균: {da_max_min_avg:.2f}%")

    # MODIFIED [Confidence Gating]: |return| * MULT 를 신뢰도로 변환, threshold 미달은 거래 신호 X
    # ETH 최적값(조원분 freqtrade): mult=20, entry_thr=0.21, exit_thr=0.32
    # 한국 주식은 일일 변동성이 ETH의 ~1/10 수준이라 MULT만 10배(200)로 스케일 보정
    GATE_MULT = 200.0
    GATE_ENTRY_THR = 0.21
    GATE_EXIT_THR = 0.32
    conf_entry = np.minimum(np.abs(pred_max) * GATE_MULT, 1.0)
    conf_exit = np.minimum(np.abs(pred_min) * GATE_MULT, 1.0)
    entry_mask = conf_entry > GATE_ENTRY_THR
    exit_mask = conf_exit > GATE_EXIT_THR
    n_test = len(pred_raw)
    n_entry_gated = int(entry_mask.sum())
    n_exit_gated = int(exit_mask.sum())
    da_entry_gated = (np.sign(pred_max[entry_mask]) == np.sign(actual_max[entry_mask])).mean() * 100 if n_entry_gated > 0 else None
    da_exit_gated = (np.sign(pred_min[exit_mask]) == np.sign(actual_min[exit_mask])).mean() * 100 if n_exit_gated > 0 else None
    if da_entry_gated is not None and da_exit_gated is not None:
        da_gated_avg = (da_entry_gated + da_exit_gated) / 2
    elif da_entry_gated is not None:
        da_gated_avg = da_entry_gated
    elif da_exit_gated is not None:
        da_gated_avg = da_exit_gated
    else:
        da_gated_avg = None
    entry_da_str = f"{da_entry_gated:.2f}%" if da_entry_gated is not None else "N/A"
    exit_da_str = f"{da_exit_gated:.2f}%" if da_exit_gated is not None else "N/A"
    avg_str = f"{da_gated_avg:.2f}%" if da_gated_avg is not None else "N/A"
    print(f"  [GATED/conf      ] Entry DA: {entry_da_str} (keep {n_entry_gated}/{n_test}={n_entry_gated/n_test*100:.1f}%) | "
          f"Exit DA: {exit_da_str} (keep {n_exit_gated}/{n_test}={n_exit_gated/n_test*100:.1f}%) | 평균: {avg_str}")

    # MODIFIED [Systematic Bias]: BEFORE(no bias) DA도 같이 출력해서 보정 효과 직접 비교
    pred_max_nb = pred_raw_uncorrected.max(axis=1)
    pred_min_nb = pred_raw_uncorrected.min(axis=1)
    da_entry_nb = (np.sign(pred_max_nb) == np.sign(actual_max)).mean() * 100
    da_exit_nb = (np.sign(pred_min_nb) == np.sign(actual_min)).mean() * 100
    da_avg_nb = (da_entry_nb + da_exit_nb) / 2
    print(f"  [BIAS BEFORE/no  ] Entry DA: {da_entry_nb:.2f}% | Exit DA: {da_exit_nb:.2f}% | 평균: {da_avg_nb:.2f}%")
    print(f"  [BIAS AFTER/+bias] Entry DA: {da_entry:.2f}% | Exit DA: {da_exit:.2f}% | 평균: {da_max_min_avg:.2f}% (Δ {da_max_min_avg - da_avg_nb:+.2f}p)")

    # MODIFIED [Multi-Filter]: Confidence Gating 위에 RSI/BB/MACD AND 패턴 적용
    # test 샘플 j의 "현재" 시점 글로벌 인덱스 = split + j + WINDOW (학습 split과 동일 매핑)
    test_now_idx = split + WINDOW
    rsi_test = features['rsi'].values[test_now_idx:test_now_idx + len(X_test)]
    bb_test = features['bb_position'].values[test_now_idx:test_now_idx + len(X_test)]
    macd_test = features['macd_hist'].values[test_now_idx:test_now_idx + len(X_test)]

    pat_rebound = (rsi_test < RSI_ENTRY_LOW) & (bb_test < BB_ENTRY_LOW)
    pat_breakout = (rsi_test > RSI_ENTRY_HIGH) & (bb_test > BB_ENTRY_HIGH) & (macd_test > 0)
    pat_overheat = (rsi_test > RSI_EXIT_HIGH) & (bb_test > BB_EXIT_HIGH)
    pat_breakdown = (bb_test < BB_EXIT_LOW) & (macd_test < 0)

    em_filt = entry_mask & (pat_rebound | pat_breakout)
    xm_filt = exit_mask & (pat_overheat | pat_breakdown)
    n_e_filt = int(em_filt.sum())
    n_x_filt = int(xm_filt.sum())
    da_e_filt = (np.sign(pred_max[em_filt]) == np.sign(actual_max[em_filt])).mean() * 100 if n_e_filt > 0 else None
    da_x_filt = (np.sign(pred_min[xm_filt]) == np.sign(actual_min[xm_filt])).mean() * 100 if n_x_filt > 0 else None
    if da_e_filt is not None and da_x_filt is not None:
        da_filt_avg = (da_e_filt + da_x_filt) / 2
    elif da_e_filt is not None:
        da_filt_avg = da_e_filt
    elif da_x_filt is not None:
        da_filt_avg = da_x_filt
    else:
        da_filt_avg = None
    e_filt_str = f"{da_e_filt:.2f}%" if da_e_filt is not None else "N/A"
    x_filt_str = f"{da_x_filt:.2f}%" if da_x_filt is not None else "N/A"
    avg_filt_str = f"{da_filt_avg:.2f}%" if da_filt_avg is not None else "N/A"
    delta_filt = (da_filt_avg - da_max_min_avg) if (da_filt_avg is not None) else None
    delta_filt_str = f"{delta_filt:+.2f}p" if delta_filt is not None else "N/A"
    print(f"  [FILTER/multi    ] Entry DA: {e_filt_str} (keep {n_e_filt}/{n_test}={n_e_filt/n_test*100:.1f}%) | "
          f"Exit DA: {x_filt_str} (keep {n_x_filt}/{n_test}={n_x_filt/n_test*100:.1f}%) | "
          f"평균: {avg_filt_str} (Δ vs gating {delta_filt_str})")

    lstm_eval_ctx.__exit__(None, None, None)

    # 9. 저장
    print(f"\n[7] 모델 저장")
    lstm_save_ctx = timer.phase("lstm_save")
    lstm_save_ctx.__enter__()
    safe_code = code.replace('^', '').replace('/', '_')

    torch.save(model.state_dict(), os.path.join(model_dir, f'{safe_code}_lstm_v5.pth'))  # MODIFIED FROM V4: *_lstm_v4.pth → *_lstm_v5.pth

    meta = {
        'input_size': X.shape[-1],
        'feature_columns': list(features.columns),
        'close_mean': close_mean,
        'volume_mean': volume_mean,
        'window': WINDOW,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'version': 'v5',  # MODIFIED FROM V4: 'v4' → 'v5'
        'target_type': 'return',  # MODIFIED FROM V4: 신규 — 타겟 타입 명시 (가격이 아닌 수익률)
        'direction_weight': 3.0,  # MODIFIED FROM V4: 신규 — 학습 시 사용한 방향 가중치
        'systematic_bias': systematic_bias,  # MODIFIED [Systematic Bias]: train set 측정 편향 (raw 수익률 단위, 예측에 더해서 보정)
        # MODIFIED [Multi-Filter]: 다중 필터 임계값 (서비스 추론에서 동일 적용)
        'filter_rsi_entry_low': RSI_ENTRY_LOW,
        'filter_bb_entry_low': BB_ENTRY_LOW,
        'filter_rsi_entry_high': RSI_ENTRY_HIGH,
        'filter_bb_entry_high': BB_ENTRY_HIGH,
        'filter_rsi_exit_high': RSI_EXIT_HIGH,
        'filter_bb_exit_high': BB_EXIT_HIGH,
        'filter_bb_exit_low': BB_EXIT_LOW,
        'foundation_models': ['deepar', 'chronos', 'adida']
    }

    with open(os.path.join(model_dir, f'{safe_code}_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, f'{safe_code}_target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_scaler, f)
    with open(os.path.join(model_dir, f'{safe_code}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"  저장 완료: {model_dir}/{safe_code}_*")
    lstm_save_ctx.__exit__(None, None, None)

    if own_timer:
        timer.report(f"{code}")
    return True

# ============================================
# 메인
# ============================================
if __name__ == '__main__':
    targets = {
        'KS11': '코스피 지수',  # MODIFIED FROM V4: KQ11(코스닥) → KS11(코스피)
        'KQ11': '코스닥 지수',
        '005930': '삼성전자',
        '000660': 'SK하이닉스',
        '035720': '카카오',
    }

    print("=" * 60)
    print("LSTM V5 — Foundation Model 앙상블 + 수익률 타겟 + 방향 가중 손실")  # MODIFIED FROM V4: 헤더 V4 → V5 + 변경 요약
    print(f"대상: {len(targets)}개 종목")
    print("=" * 60)

    total_start = time.time()
    success = 0
    per_ticker = []  # 20260602: 종목별 phase 비교 누적

    for code, name in targets.items():
        print(f"\n>>> {name} ({code})")
        timer = PhaseTimer()
        if train_model(code, timer=timer):
            success += 1
        total = timer.report(f"{name} ({code})")
        per_ticker.append((f"{name}({code})", timer, total))

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"전체 완료! {success}/{len(targets)}개 성공")
    print(f"총 소요시간: {total_elapsed/60:.1f}분")
    print(f"{'='*60}")

    # 20260602: 종목간 phase 비교표 — 어느 종목 어느 phase 가 튀는지 한눈에
    if per_ticker:
        phases = ["data_download", "autogluon", "external", "features",
                  "lstm_prep", "lstm_train", "lstm_eval", "lstm_save"]
        print(f"\n{'='*60}")
        print(f"  ⏱  종목 간 PHASE 비교 (단위: 초)")
        print(f"{'='*60}")
        header = f"  {'phase':<16}" + "".join(f"{n[:14]:>15}" for n, _, _ in per_ticker)
        print(header)
        print("  " + "-" * (16 + 15 * len(per_ticker)))
        for ph in phases:
            row = f"  {ph:<16}"
            for _, tm, _ in per_ticker:
                sec = sum(s for n, p, s in tm.records if n == ph and p is None)
                row += f"{sec:>14.1f}s"
            print(row)
        print("  " + "-" * (16 + 15 * len(per_ticker)))
        row = f"  {'TOTAL':<16}"
        for _, _, tot in per_ticker:
            row += f"{tot:>14.1f}s"
        print(row)
        print(f"{'='*60}")

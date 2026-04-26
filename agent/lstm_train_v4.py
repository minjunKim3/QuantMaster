import sys
import json
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
# 모델 정의 (V3와 동일)
# ============================================
class StockLSTM_V4(nn.Module):
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

def huber_loss(pred, target, delta=0.1):
    error = pred - target
    is_small = torch.abs(error) <= delta
    small_loss = 0.5 * error**2
    large_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small, small_loss, large_loss).mean()

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
        except:
            pass
    return dwt_trend, dwt_high, dwt_low, dwt_energy

# ============================================
# 외부 지표 (V3와 동일)
# ============================================
def get_external_data(start_date, end_date):
    externals = {}
    symbols = {'vix': '^VIX', 'gold': 'GC=F', 'dxy': 'DX-Y.NYB'}
    for name, ticker in symbols.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                externals[name] = data['Close']
        except:
            pass
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
def generate_foundation_predictions(df, code, close_mean, model_dir):
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
    if os.path.exists(ag_path):
        print("  [Foundation] 기존 AutoGluon 모델 로드 중...")
        try:
            predictor = TimeSeriesPredictor.load(ag_path)
            print("  [Foundation] 기존 모델 로드 성공!")
        except:
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
                    except:
                        pass
            except Exception as e:
                print(f"    배치 {batch_start}-{batch_end} 실패: {e}")
                continue
        
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
def train_model(code, start_date='2015-01-01'):
    print(f"\n{'='*60}")
    print(f"LSTM V4 학습: {code} (Foundation Model 앙상블)")
    print(f"{'='*60}")
    
    # 1. 데이터
    print(f"\n[1] 데이터 다운로드: {code}")
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
    model_dir = os.path.join(script_dir, 'models_v4')
    os.makedirs(model_dir, exist_ok=True)
    
    foundation_preds = generate_foundation_predictions(df, code, close_mean, model_dir)
    
    # 3. 외부 지표
    print(f"\n[3] 외부 지표 로드")
    external_data = get_external_data(start_date, df.index[-1].strftime('%Y-%m-%d'))
    
    # 4. 특성 생성
    print(f"\n[4] 특성 생성")
    features = create_features_v4(df, external_data, foundation_preds)
    print(f"  특성 수: {len(features.columns)}개")
    print(f"  특성 목록: {list(features.columns)}")
    
    # 5. 정규화
    scaler = RobustScaler()
    target_scaler = RobustScaler()
    
    feature_values = features.values
    scaled = scaler.fit_transform(feature_values)
    
    # 6. 윈도우 + 타겟
    WINDOW = 60
    close_idx = list(features.columns).index('close')
    close_normalized = feature_values[:, close_idx] / close_mean
    
    X, y = [], []
    for i in range(WINDOW, len(scaled) - 3):
        X.append(scaled[i - WINDOW:i])
        y.append([close_normalized[i + 1], close_normalized[i + 2], close_normalized[i + 3]])
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockLSTM_V4(input_size=X.shape[-1]).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(device), torch.FloatTensor(y_train).to(device))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = TensorDataset(torch.FloatTensor(X_test).to(device), torch.FloatTensor(y_test).to(device))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
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
            loss = huber_loss(pred, batch_y, delta=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                loss = huber_loss(pred, batch_y, delta=0.1)
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
    
    # 8. 평가
    print(f"\n[6] 평가")
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    pred_raw = target_scaler.inverse_transform(pred_scaled)
    
    for step in range(3):
        actual = y_test_raw[:, step] * close_mean
        predicted = pred_raw[:, step] * close_mean
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        print(f"  t+{step+1}: MAPE {mape:.2f}% | RMSE {rmse:.2f}")
    
    # 9. 저장
    print(f"\n[7] 모델 저장")
    safe_code = code.replace('^', '').replace('/', '_')
    
    torch.save(model.state_dict(), os.path.join(model_dir, f'{safe_code}_lstm_v4.pth'))
    
    meta = {
        'input_size': X.shape[-1],
        'feature_columns': list(features.columns),
        'close_mean': close_mean,
        'volume_mean': volume_mean,
        'window': WINDOW,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'version': 'v4',
        'foundation_models': ['deepar', 'chronos', 'adida']
    }
    
    with open(os.path.join(model_dir, f'{safe_code}_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, f'{safe_code}_target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_scaler, f)
    with open(os.path.join(model_dir, f'{safe_code}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"  저장 완료: {model_dir}/{safe_code}_*")
    return True

# ============================================
# 메인
# ============================================
if __name__ == '__main__':
    targets = {
        'KQ11': '코스닥 지수',
    }
    
    print("=" * 60)
    print("LSTM V4 — Foundation Model 앙상블 (Chronos + DeepAR + ADIDA)")
    print(f"대상: {len(targets)}개 종목")
    print("=" * 60)
    
    total_start = time.time()
    success = 0
    
    for code, name in targets.items():
        print(f"\n>>> {name} ({code})")
        if train_model(code):
            success += 1
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"전체 완료! {success}/{len(targets)}개 성공")
    print(f"총 소요시간: {total_elapsed/60:.1f}분")
    print(f"{'='*60}")
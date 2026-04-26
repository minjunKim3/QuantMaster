import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import FinanceDataReader as fdr
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pywt
import copy
import time
import os
import pickle
import yfinance as yf

# ============================================
# [개선점 1] 모델 구조 — 조원분 EnsembleLSTMPredictor 반영
# 
# 기존 너의 LSTM:
#   LSTM → Linear(32) → ReLU → Linear(1) → 1개 예측
# 
# 조원분 LSTM:
#   LSTM → BatchNorm → Linear → GELU → Dropout → Linear → GELU → Dropout → Linear(3) → 3개 예측
#
# 바뀐 점:
#   - BatchNorm: 학습 안정화 (배치마다 정규화)
#   - GELU: ReLU보다 부드러움 (음수 영역에서 약간의 값을 허용)
#   - 3개 출력: t+1(내일), t+2(모레), t+3(글피) 동시 예측
# ============================================
class StockLSTM_V3(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 출력 헤드 (조원분 구조 그대로)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 3)  # 3스텝 예측!
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]  # 마지막 시간 스텝
        return self.head(h_last)

# ============================================
# [개선점 2] Huber Loss
# 
# 기존: MSE Loss → 극단값(급등/급락)에 민감 → 전체 모델이 흔들림
# 개선: Huber Loss → 작은 오차는 MSE처럼, 큰 오차는 MAE처럼 처리
#       → 급등/급락에 덜 흔들림
# ============================================
def huber_loss(pred, target, delta=0.1):
    error = pred - target
    is_small = torch.abs(error) <= delta
    small_loss = 0.5 * error**2
    large_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small, small_loss, large_loss).mean()

# ============================================
# [개선점 3] DWT 웨이블릿 분석
# 
# 주가를 수학적으로 분해하는 기법:
#   원본 주가 = 추세(저주파) + 단기 변동(중주파) + 노이즈(고주파)
# 
# 비유: 음악에서 베이스(추세) + 보컬(단기변동) + 잡음(노이즈)을 분리하는 것
# LSTM에게 "지금 추세가 어떤지, 노이즈가 얼마나 큰지"를 알려줌
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
            dwt_trend[i] = coeffs[0][-1]      # 추세 (저주파)
            dwt_high[i] = coeffs[1][-1]        # 단기 변동 (중주파)
            dwt_low[i] = coeffs[2][-1]         # 노이즈 (고주파)
            dwt_energy[i] = np.sum(coeffs[1]**2) + np.sum(coeffs[2]**2)  # 에너지
        except:
            pass
    
    return dwt_trend, dwt_high, dwt_low, dwt_energy

# ============================================
# [개선점 4] 외부 지표 (VIX, 금, 달러)
# 
# 주가는 혼자 움직이지 않아:
#   - VIX 높으면 → 시장 공포 → 주가 하락 가능성
#   - 금 가격 상승 → 안전자산 선호 → 주식에서 자금 이탈
#   - 달러 강세 → 수출주에 불리
# ============================================
def get_external_data(start_date, end_date):
    externals = {}
    symbols = {'vix': '^VIX', 'gold': 'GC=F', 'dxy': 'DX-Y.NYB'}
    
    for name, ticker in symbols.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                externals[name] = data['Close']
                print(f"  외부지표 {name}: {len(data)}일")
        except Exception as e:
            print(f"  외부지표 {name} 실패: {e}")
    
    return externals

# ============================================
# 특성 생성 함수 — 기존 6개 → 15개+
# ============================================
def create_features(df, external_data=None):
    close = df['Close'].values
    volume = df['Volume'].values
    
    features = pd.DataFrame(index=df.index)
    
    # 기존 특성 (6개)
    features['close'] = close
    features['ma5'] = pd.Series(close).rolling(5).mean().values
    features['ma20'] = pd.Series(close).rolling(20).mean().values
    
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi'] = (100 - (100 / (1 + gain / (loss_val + 1e-10)))).values
    
    features['change'] = pd.Series(close).pct_change().values
    features['volume_norm'] = volume / (np.max(volume) + 1e-10)
    
    # [신규] 볼린저밴드 위치
    ma20_s = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()
    bb_upper = ma20_s + 2 * std20
    bb_lower = ma20_s - 2 * std20
    features['bb_position'] = ((close - bb_lower) / (bb_upper - bb_lower + 1e-8)).values
    
    # [신규] MACD 히스토그램
    exp12 = pd.Series(close).ewm(span=12).mean()
    exp26 = pd.Series(close).ewm(span=26).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9).mean()
    features['macd_hist'] = (macd - signal).values
    
    # [신규] ATR (변동성)
    high = df['High'].values if 'High' in df.columns else close
    low = df['Low'].values if 'Low' in df.columns else close
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    features['atr'] = pd.Series(tr).rolling(14).mean().values
    
    # [신규] DWT 웨이블릿
    print("  DWT 웨이블릿 계산 중...")
    dwt_trend, dwt_high, dwt_low, dwt_energy = add_dwt_features(close, min(128, len(close) // 4))
    features['dwt_trend'] = dwt_trend
    features['dwt_high'] = dwt_high
    features['dwt_low'] = dwt_low
    features['dwt_energy'] = dwt_energy
    
    # [신규] 외부 지표
    if external_data:
        for name, series in external_data.items():
            aligned = series.reindex(df.index, method='ffill')
            features[f'{name}_norm'] = (aligned / aligned.mean()).values

    # [추가 지표 1] Stochastic %K, %D
    low_14 = pd.Series(low).rolling(14).min()
    high_14 = pd.Series(high).rolling(14).max()
    stoch_k = ((close - low_14) / (high_14 - low_14 + 1e-10) * 100).values
    features['stoch_k'] = stoch_k
    features['stoch_d'] = pd.Series(stoch_k).rolling(3).mean().values
    
    # [추가 지표 2] Williams %R
    features['williams_r'] = (-(high_14 - close) / (high_14 - low_14 + 1e-10) * 100).values
    
    # [추가 지표 3] OBV (On Balance Volume)
    obv = np.zeros(len(close))
    for j in range(1, len(close)):
        if close[j] > close[j-1]:
            obv[j] = obv[j-1] + volume[j]
        elif close[j] < close[j-1]:
            obv[j] = obv[j-1] - volume[j]
        else:
            obv[j] = obv[j-1]
    features['obv'] = obv / (np.max(np.abs(obv)) + 1e-10)
    
    # [추가 지표 4] Price ROC (Rate of Change)
    features['roc_5'] = pd.Series(close).pct_change(5).values
    features['roc_20'] = pd.Series(close).pct_change(20).values
    
    # [추가 지표 5] 이격도 (MA Gap)
    features['ma_gap_5'] = ((close - pd.Series(close).rolling(5).mean()) / (pd.Series(close).rolling(5).mean() + 1e-10)).values
    features['ma_gap_20'] = ((close - pd.Series(close).rolling(20).mean()) / (pd.Series(close).rolling(20).mean() + 1e-10)).values
    
    # 결측치 처리
    features = features.ffill().bfill().fillna(0)
    
    return features

# ============================================
# 학습 함수
# ============================================
def train_model(code, start_date='2015-01-01'):
    print(f"\n{'='*60}")
    print(f"LSTM V3 학습: {code}")
    print(f"{'='*60}")
    
    # 1. 데이터 다운로드
    print(f"\n[1] 데이터 다운로드: {code}")
    df = fdr.DataReader(code, start_date)
    
    if df.empty or len(df) < 200:
        print(f"  데이터 부족! ({len(df)}일)")
        return False
    
    print(f"  데이터: {len(df)}일")
    
    # 정규화 기준값 저장 (조원분 방식)
    close_mean = df['Close'].mean()
    volume_mean = df['Volume'].mean()
    
    # 2. 외부 지표 가져오기
    print(f"\n[2] 외부 지표 로드")
    external_data = get_external_data(start_date, df.index[-1].strftime('%Y-%m-%d'))
    
    # 3. 특성 생성
    print(f"\n[3] 특성 생성")
    features = create_features(df, external_data)
    print(f"  특성 수: {len(features.columns)}개")
    print(f"  특성 목록: {list(features.columns)}")
    
    # 4. 정규화 (RobustScaler — 조원분 방식)
    scaler = RobustScaler()
    target_scaler = RobustScaler()
    
    feature_values = features.values
    scaled = scaler.fit_transform(feature_values)
    
# 5. 윈도우 + 3스텝 타겟 생성 (수익률 예측!)
    WINDOW = 60
    close_idx = list(features.columns).index('close')
    close_vals = feature_values[:, close_idx]
    
    X, y = [], []
    for i in range(WINDOW, len(scaled) - 3):
        X.append(scaled[i - WINDOW:i])
        # 수익률 타겟: 현재 대비 변화율 (예: +0.01 = 1% 상승)
        current = close_vals[i]
        y.append([
            (close_vals[i + 1] - current) / (current + 1e-10),
            (close_vals[i + 2] - current) / (current + 1e-10),
            (close_vals[i + 3] - current) / (current + 1e-10)
        ])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # 스케일링
    y_scaled = target_scaler.fit_transform(y)
    
    # Train/Test 분할 (시계열이라 shuffle=False)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]
    y_test_raw = y[split:]
    
    print(f"  학습: {len(X_train)}개 | 테스트: {len(X_test)}개")
    
    # 6. 모델 학습
    print(f"\n[4] 학습 시작")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = StockLSTM_V3(input_size=X.shape[-1]).to(device)
    
    # [개선점 6] CosineAnnealingWarmRestarts
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(y_train).to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_test).to(device),
        torch.FloatTensor(y_test).to(device)
    )
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
    
    # 7. 평가
    print(f"\n[5] 평가")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        pred_scaled = model(X_test_tensor).cpu().numpy()
    
    pred_raw = target_scaler.inverse_transform(pred_scaled)
    
    # 테스트 구간의 현재 가격 가져오기
    test_start_idx = WINDOW + split
    
    for step in range(3):
        actual_returns = y_test_raw[:, step]
        predicted_returns = pred_raw[:, step]
        
        # 수익률 → 가격 변환
        current_prices = close_vals[test_start_idx:test_start_idx + len(y_test_raw)]
        actual_prices = current_prices * (1 + actual_returns)
        predicted_prices = current_prices * (1 + predicted_returns)
        
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        print(f"  t+{step+1}: MAPE {mape:.2f}% | RMSE {rmse:.2f}")
    
    # [개선] Systematic Bias 보정
    print(f"\n[5.5] Systematic Bias 계산")
    model.eval()
    with torch.no_grad():
        all_pred_scaled = model(torch.FloatTensor(X).to(device)).cpu().numpy()
    all_pred_raw = target_scaler.inverse_transform(all_pred_scaled)
    
    systematic_bias = y.mean(axis=0) - all_pred_raw.mean(axis=0)
    print(f"  Bias: {systematic_bias}")
    
    # Bias 보정 후 재평가
    print(f"\n[6] 평가 (Bias 보정 후)")
    for step in range(3):
        actual_returns = y_test_raw[:, step]
        predicted_returns = pred_raw[:, step] + systematic_bias[step]
        
        current_prices = close_vals[test_start_idx:test_start_idx + len(y_test_raw)]
        actual_prices = current_prices * (1 + actual_returns)
        predicted_prices = current_prices * (1 + predicted_returns)
        
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        print(f"  t+{step+1}: MAPE {mape:.2f}% | RMSE {rmse:.2f}")
    
    # 8. 저장
    print(f"\n[6] 모델 저장")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models_v3')
    os.makedirs(model_dir, exist_ok=True)
    
    safe_code = code.replace('^', '').replace('/', '_')
    
    # 모델 가중치
    torch.save(model.state_dict(), os.path.join(model_dir, f'{safe_code}_lstm_v3.pth'))
    
    # 스케일러 + 메타 정보
    meta = {
        'input_size': X.shape[-1],
        'feature_columns': list(features.columns),
        'close_mean': close_mean,
        'volume_mean': volume_mean,
        'window': WINDOW,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'target_type': 'returns',
        'systematic_bias': systematic_bias.tolist(),
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
        'KS11': '코스피 지수',
        'KQ11': '코스닥 지수',
        '005930': '삼성전자',
    }
    
    print("=" * 60)
    print("LSTM V3 — 조원분 모델 개선점 통합")
    print(f"개선: DWT 웨이블릿 + RobustScaler + Huber Loss + 3스텝 예측")
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
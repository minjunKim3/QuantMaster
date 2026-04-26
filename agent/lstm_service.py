import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import FinanceDataReader as fdr
from sklearn.preprocessing import RobustScaler
import pywt
import pickle
import os
import yfinance as yf

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ============================================
# V3 모델 정의 (lstm_train_v3.py와 동일해야 함!)
# ============================================
class StockLSTM_V3(nn.Module):
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
        h_last = out[:, -1, :]
        return self.head(h_last)

# ============================================
# 기존 V2 모델 (폴백용)
# ============================================
class StockLSTM_V2(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze()

# ============================================
# DWT 웨이블릿
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
# 특성 생성 (V3)
# ============================================
def create_features_v3(df, external_data=None):
    close = df['Close'].values
    volume = df['Volume'].values
    
    features = pd.DataFrame(index=df.index)
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
    
    dwt_trend, dwt_high, dwt_low, dwt_energy = add_dwt_features(close, min(128, len(close) // 4))
    features['dwt_trend'] = dwt_trend
    features['dwt_high'] = dwt_high
    features['dwt_low'] = dwt_low
    features['dwt_energy'] = dwt_energy
    
    if external_data:
        for name, series in external_data.items():
            aligned = series.reindex(df.index, method='ffill')
            features[f'{name}_norm'] = (aligned / aligned.mean()).values
    
    features = features.ffill().bfill().fillna(0)
    return features

# ============================================
# 외부 지표
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
# 메인 실행
# ============================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error": "파라미터 필요"}, cls=NpEncoder))
        sys.exit(1)
    
    arg = sys.argv[1]
    if arg.endswith('.json'):
        with open(arg, 'r', encoding='utf-8') as f:
            params = json.load(f)
    else:
        params = json.loads(arg)
    
    code = params.get('code', 'KS11')
    days = params.get('days', 100)
    
    code = code.replace('.KS', '').replace('.KQ', '')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    safe_code = code.replace('^', '').replace('/', '_')
    
    # V3 모델 확인
    v3_dir = os.path.join(script_dir, 'models_v3')
    v3_model_path = os.path.join(v3_dir, f'{safe_code}_lstm_v3.pth')
    v3_meta_path = os.path.join(v3_dir, f'{safe_code}_meta.pkl')
    v3_scaler_path = os.path.join(v3_dir, f'{safe_code}_scaler.pkl')
    v3_target_scaler_path = os.path.join(v3_dir, f'{safe_code}_target_scaler.pkl')
    
    # V2 모델 확인 (폴백)
    v2_dir = os.path.join(script_dir, 'models')
    v2_model_path = os.path.join(v2_dir, f'{safe_code}_lstm.pth')
    
    use_v3 = os.path.exists(v3_model_path) and os.path.exists(v3_meta_path)
    use_v2 = os.path.exists(v2_model_path)
    
    if not use_v3 and not use_v2:
        print(json.dumps({"error": f"{code}의 LSTM 모델이 없습니다. 학습된 종목: KS11, KQ11, 005930, 000660, 035720"}, cls=NpEncoder))
        sys.exit(0)
    
    model_version = "V3" if use_v3 else "V2"
    print(f"[LSTM] 예측 시작: {code} ({model_version})", file=sys.stderr)
    
    try:
        # 데이터 다운로드
        df = fdr.DataReader(code, '2015-01-01')
        
        if use_v3:
            # ===== V3 모델 =====
            with open(v3_meta_path, 'rb') as f:
                meta = pickle.load(f)
            with open(v3_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(v3_target_scaler_path, 'rb') as f:
                target_scaler = pickle.load(f)
            
            close_mean = meta['close_mean']
            WINDOW = meta['window']
            
            # 외부 지표
            external_data = get_external_data('2015-01-01', df.index[-1].strftime('%Y-%m-%d'))
            
            # 특성 생성
            features = create_features_v3(df, external_data)
            
            # 학습 때와 동일한 특성만 사용
            expected_cols = meta['feature_columns']
            for col in expected_cols:
                if col not in features.columns:
                    features[col] = 0
            features = features[expected_cols]
            
            scaled = scaler.transform(features.values)
            
            # 모델 로드
            model = StockLSTM_V3(input_size=meta['input_size'],
                                  hidden_size=meta.get('hidden_size', 128),
                                  num_layers=meta.get('num_layers', 2),
                                  dropout=meta.get('dropout', 0.2))
            model.load_state_dict(torch.load(v3_model_path, map_location='cpu', weights_only=True))
            model.eval()
            
            # 예측
            close_normalized = features['close'].values / close_mean
            predictions_t1 = []
            actuals = []
            dates = []
            
            start_idx = max(WINDOW, len(scaled) - days)
            
            for i in range(start_idx, len(scaled)):
                window_data = scaled[i - WINDOW:i]
                X = torch.FloatTensor(window_data).unsqueeze(0)
                
                with torch.no_grad():
                    pred_scaled = model(X).numpy()
                
                pred_raw = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
                pred_price_t1 = pred_raw[0] * close_mean
                actual_price = df['Close'].iloc[i]
                date = df.index[i].strftime('%Y-%m-%d')
                
                predictions_t1.append(round(float(pred_price_t1), 2))
                actuals.append(round(float(actual_price), 2))
                dates.append(date)
            
        else:
            # ===== V2 모델 (폴백) =====
            close = df['Close']
            features_df = pd.DataFrame(index=df.index)
            features_df['Close'] = close
            features_df['MA5'] = close.rolling(5).mean()
            features_df['MA20'] = close.rolling(20).mean()
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
            features_df['RSI'] = 100 - (100 / (1 + gain / (loss_val + 1e-10)))
            features_df['Change'] = close.pct_change()
            features_df['Volume'] = df['Volume'] / (df['Volume'].max() + 1e-10)
            features_df.ffill(inplace=True)
            features_df.fillna(0, inplace=True)
            
            from sklearn.preprocessing import MinMaxScaler
            scaler_dict = {}
            scaled_df = pd.DataFrame(index=features_df.index)
            for col in features_df.columns:
                s = MinMaxScaler()
                vals = features_df[[col]]
                if vals.max().values[0] == vals.min().values[0]:
                    scaled_df[col] = 0.5
                else:
                    scaled_df[col] = s.fit_transform(vals).flatten()
                scaler_dict[col] = s
            
            close_scaler = scaler_dict['Close']
            scaled = scaled_df.values
            WINDOW = 60
            
            input_size = len(features_df.columns)
            model = StockLSTM_V2(input_size=input_size)
            model.load_state_dict(torch.load(v2_model_path, map_location='cpu', weights_only=True))
            model.eval()
            
            predictions_t1 = []
            actuals = []
            dates = []
            start_idx = max(WINDOW, len(scaled) - days)
            
            for i in range(start_idx, len(scaled)):
                window_data = scaled[i - WINDOW:i]
                X = torch.FloatTensor(window_data).unsqueeze(0)
                with torch.no_grad():
                    pred = model(X).item()
                pred_price = close_scaler.inverse_transform([[pred]])[0][0]
                actual_price = features_df['Close'].iloc[i]
                date = features_df.index[i].strftime('%Y-%m-%d')
                predictions_t1.append(round(float(pred_price), 2))
                actuals.append(round(float(actual_price), 2))
                dates.append(date)
        
        # 통계
        pred_arr = np.array(predictions_t1)
        actual_arr = np.array(actuals)
        mape = round(float(np.mean(np.abs((actual_arr - pred_arr) / actual_arr)) * 100), 2)
        
        pred_diff = np.diff(pred_arr)
        actual_diff = np.diff(actual_arr)
        valid = (pred_diff != 0) & (actual_diff != 0)
        direction = round(float(np.sum(np.sign(pred_diff[valid]) == np.sign(actual_diff[valid])) / valid.sum() * 100), 1) if valid.sum() > 0 else 0
        
        result = {
            'code': code,
            'modelVersion': model_version,
            'mape': mape,
            'directionAccuracy': direction,
            'latestActual': actuals[-1],
            'latestPredicted': predictions_t1[-1],
            'dates': dates,
            'actuals': actuals,
            'predictions': predictions_t1
        }
        
        sys.stdout.reconfigure(encoding='utf-8')
        print(json.dumps(result, cls=NpEncoder))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, cls=NpEncoder))
        sys.exit(1)
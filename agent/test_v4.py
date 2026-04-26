import numpy as np
import pandas as pd
import torch
import pickle
import os
import FinanceDataReader as fdr
from lstm_train_v4 import StockLSTM_V4, create_features_v4, get_external_data, generate_foundation_predictions

code = 'KQ11'
model_dir = 'models_v4'
safe_code = code

# 메타, 스케일러 로드
with open(f'{model_dir}/{safe_code}_meta.pkl', 'rb') as f:
    meta = pickle.load(f)
with open(f'{model_dir}/{safe_code}_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open(f'{model_dir}/{safe_code}_target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)

close_mean = meta['close_mean']
WINDOW = meta['window']

# 데이터
df = fdr.DataReader(code, '2015-01-01')
external_data = get_external_data('2015-01-01', df.index[-1].strftime('%Y-%m-%d'))

# Foundation 예측
print("Foundation 예측 생성 중 (기존 모델 로드)...")
foundation_preds = generate_foundation_predictions(df, code, close_mean, model_dir)

# 특성 생성
features = create_features_v4(df, external_data, foundation_preds)
expected_cols = meta['feature_columns']
for col in expected_cols:
    if col not in features.columns:
        features[col] = 0
features = features[expected_cols]

scaled = scaler.transform(features.values)
close_normalized = features['close'].values / close_mean

# 모델 로드
model = StockLSTM_V4(input_size=meta['input_size'])
model.load_state_dict(torch.load(f'{model_dir}/{safe_code}_lstm_v4.pth', map_location='cpu', weights_only=True))
model.eval()

# 마지막 100일 예측
days = 100
start_idx = max(WINDOW, len(scaled) - days)

actuals = []
preds = []

for i in range(start_idx, len(scaled)):
    window_data = scaled[i - WINDOW:i]
    X = torch.FloatTensor(window_data).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(X).numpy()
    pred_raw = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
    
    preds.append(pred_raw[0] * close_mean)
    actuals.append(df['Close'].iloc[i])

pred_arr = np.array(preds)
actual_arr = np.array(actuals)
mape = np.mean(np.abs((actual_arr - pred_arr) / actual_arr)) * 100

print(f"\n{'='*40}")
print(f"V4 KQ11 결과 (최근 {days}일)")
print(f"MAPE: {mape:.2f}%")
print(f"{'='*40}")
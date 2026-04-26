import torch, pickle, numpy as np
import FinanceDataReader as fdr
from lstm_train_v3 import StockLSTM_V3, create_features, get_external_data

code = 'KQ11'
with open('models_v3/KQ11_meta.pkl','rb') as f: meta = pickle.load(f)
with open('models_v3/KQ11_scaler.pkl','rb') as f: scaler = pickle.load(f)
with open('models_v3/KQ11_target_scaler.pkl','rb') as f: ts = pickle.load(f)

model = StockLSTM_V3(input_size=meta['input_size'])
model.load_state_dict(torch.load('models_v3/KQ11_lstm_v3.pth', map_location='cpu', weights_only=True))
model.eval()

df = fdr.DataReader(code, '2015-01-01')
ext = get_external_data('2015-01-01', df.index[-1].strftime('%Y-%m-%d'))
features = create_features(df, ext)
for col in meta['feature_columns']:
    if col not in features.columns: features[col] = 0
features = features[meta['feature_columns']]
scaled = scaler.transform(features.values)

print("=== 모델 Raw Output 확인 ===")
for i in range(-5, 0):
    idx = len(scaled) + i
    window = scaled[idx-60:idx]
    X = torch.FloatTensor(window).unsqueeze(0)
    with torch.no_grad():
        raw_output = model(X).numpy()[0]
    raw_returns = ts.inverse_transform(raw_output.reshape(1,-1))[0]
    date = df.index[idx].strftime("%Y-%m-%d")
    print(f"날짜: {date}")
    print(f"  모델 raw: {raw_output[0]:.6f}, {raw_output[1]:.6f}, {raw_output[2]:.6f}")
    print(f"  수익률:   {raw_returns[0]:.6f}, {raw_returns[1]:.6f}, {raw_returns[2]:.6f}")
    print()
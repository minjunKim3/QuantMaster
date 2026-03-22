import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
import os

# ============================================
# 모델 정의 (학습 때와 동일해야 함!)
# ============================================
class StockLSTM_V2(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(StockLSTM_V2, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze()

# ============================================
# 메인 실행
# ============================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error": "파라미터 필요"}))
        sys.exit(1)
    
    # 파라미터 읽기
    arg = sys.argv[1]
    if arg.endswith('.json'):
        with open(arg, 'r') as f:
            params = json.load(f)
    else:
        params = json.loads(arg)
    
    code = params.get('code', 'KS11')
    days = params.get('days', 100)
    
    # 모델 파일 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'kospi_lstm_v2_best.pth')
    
    print(f"[LSTM] 예측 시작: {code}", file=sys.stderr)
    
    try:
        # 1. 데이터 다운로드
        df = fdr.DataReader(code, '2015-01-01')
        
        # 2. 특성 생성 (학습 때와 동일!)
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
        features_df['Volume'] = df['Volume'] / df['Volume'].max()
        features_df.dropna(inplace=True)
        
        # 3. 정규화
        scaler_dict = {}
        scaled_df = pd.DataFrame(index=features_df.index)
        for col in features_df.columns:
            scaler = MinMaxScaler()
            scaled_df[col] = scaler.fit_transform(features_df[[col]]).flatten()
            scaler_dict[col] = scaler
        
        close_scaler = scaler_dict['Close']
        scaled = scaled_df.values
        
        # 4. 모델 로드
        input_size = len(features_df.columns)
        model = StockLSTM_V2(input_size=input_size)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # 5. 최근 데이터로 예측
        WINDOW = 60
        close_idx = list(features_df.columns).index('Close')
        
        predictions = []
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
            
            predictions.append(round(pred_price, 2))
            actuals.append(round(actual_price, 2))
            dates.append(date)
        
        # 6. 통계 계산
        pred_arr = np.array(predictions)
        actual_arr = np.array(actuals)
        mape = round(np.mean(np.abs((actual_arr - pred_arr) / actual_arr)) * 100, 2)
        
        pred_diff = np.diff(pred_arr)
        actual_diff = np.diff(actual_arr)
        valid = (pred_diff != 0) & (actual_diff != 0)
        if valid.sum() > 0:
            direction = round(np.sum(
                np.sign(pred_diff[valid]) == np.sign(actual_diff[valid])
            ) / valid.sum() * 100, 1)
        else:
            direction = 0
        
        # 7. JSON 출력
        result = {
            'code': code,
            'mape': mape,
            'directionAccuracy': direction,
            'latestActual': actuals[-1],
            'latestPredicted': predictions[-1],
            'dates': dates,
            'actuals': actuals,
            'predictions': predictions
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
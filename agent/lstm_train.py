import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import os

# ============================================
# 모델 정의
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
# 학습 함수
# ============================================
def train_model(code, start_date='2015-01-01'):
    print(f"\n{'='*60}")
    print(f"LSTM 학습: {code}")
    print(f"{'='*60}")
    
    # 1. 데이터 다운로드
    print(f"\n[1] 데이터 다운로드: {code}")
    df = fdr.DataReader(code, start_date)
    
    if df.empty or len(df) < 100:
        print(f"  데이터 부족! ({len(df)}일) - 건너뜀")
        return False
    
    # 2. 특성 생성
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
    
    print(f"  데이터: {len(features_df)}일, 특성: {len(features_df.columns)}개")
    
    # 3. 정규화
    scaler_dict = {}
    scaled_df = pd.DataFrame(index=features_df.index)
    for col in features_df.columns:
        scaler = MinMaxScaler()
        vals = features_df[[col]]
        if vals.max().values[0] == vals.min().values[0]:
            scaled_df[col] = 0.5
        else:
            scaled_df[col] = scaler.fit_transform(vals).flatten()
        scaler_dict[col] = scaler
    
    scaled = scaled_df.values
    
    # 4. 학습 데이터 생성
    WINDOW = 60
    close_idx = list(features_df.columns).index('Close')
    
    X, y = [], []
    for i in range(WINDOW, len(scaled)):
        X.append(scaled[i - WINDOW:i])
        y.append(scaled[i, close_idx])
    
    X = np.array(X)
    y = np.array(y)
    
    split = int(len(X) * 0.8)
    X_train = torch.FloatTensor(X[:split])
    y_train = torch.FloatTensor(y[:split])
    X_test = torch.FloatTensor(X[split:])
    y_test = torch.FloatTensor(y[split:])
    
    print(f"  학습: {len(X_train)}개 | 테스트: {len(X_test)}개")
    
    # 5. 학습
    input_size = X_train.shape[2]
    model = StockLSTM_V2(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    EPOCHS = 300
    BATCH_SIZE = 64
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    print(f"\n[2] 학습 시작 (최대 {EPOCHS} epochs)")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            batch_X = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(X_train) / BATCH_SIZE)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} | {elapsed:.0f}초")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 40:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    elapsed = time.time() - start_time
    print(f"  학습 완료! {elapsed:.0f}초 ({elapsed/60:.1f}분)")
    
    # 6. 평가
    model.eval()
    close_scaler = scaler_dict['Close']
    
    with torch.no_grad():
        predictions = model(X_test).numpy()
    
    pred_prices = close_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actual_prices = close_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
    
    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
    
    pred_diff = np.diff(pred_prices)
    actual_diff = np.diff(actual_prices)
    valid = (pred_diff != 0) & (actual_diff != 0)
    direction = np.sum(np.sign(pred_diff[valid]) == np.sign(actual_diff[valid])) / valid.sum() * 100 if valid.sum() > 0 else 0
    
    print(f"\n[3] 결과")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  방향 정확도: {direction:.1f}%")
    
    # 7. 모델 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    safe_code = code.replace('^', '').replace('/', '_')
    model_path = os.path.join(model_dir, f'{safe_code}_lstm.pth')
    torch.save(model.state_dict(), model_path)
    print(f"  모델 저장: {model_path}")
    
    # 8. 차트 저장
    plt.figure(figsize=(14, 5))
    plt.plot(actual_prices[-100:], label='Actual', color='blue', linewidth=1.5)
    plt.plot(pred_prices[-100:], label='Predicted', color='red', linewidth=1.5, linestyle='--')
    plt.title(f'{code} - LSTM Prediction (Last 100 Days)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(model_dir, f'{safe_code}_lstm_chart.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  차트 저장: {chart_path}")
    
    return True

# ============================================
# 메인
# ============================================
if __name__ == '__main__':
    targets = {
        'KS11': '코스피 지수',
        'KQ11': '코스닥 지수',
        '005930': '삼성전자',
        '000660': 'SK하이닉스',
        '035720': '카카오',
    }
    
    print("=" * 60)
    print("LSTM 종목별 일괄 학습")
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
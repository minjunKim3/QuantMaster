import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("LSTM 주가 예측 V2 - 수정 버전")
print("=" * 60)

# ============================================
# 1. 데이터 준비
# ============================================
print("\n[1단계] 데이터 다운로드 중...")
df = fdr.DataReader('KS11', '2015-01-01')

# 핵심 특성만 선택 (너무 많으면 오히려 혼란)
raw_close = df['Close'].values.reshape(-1, 1)

# 기술적 지표 계산
close = df['Close']
features_df = pd.DataFrame(index=df.index)
features_df['Close'] = close
features_df['MA5'] = close.rolling(5).mean()
features_df['MA20'] = close.rolling(20).mean()

# RSI
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
features_df['RSI'] = 100 - (100 / (1 + gain / (loss_val + 1e-10)))

# 변동률
features_df['Change'] = close.pct_change()

# 거래량 (정규화)
features_df['Volume'] = df['Volume'] / df['Volume'].max()

features_df.dropna(inplace=True)

print(f"  총 데이터: {len(features_df)}일")
print(f"  입력 특성: {len(features_df.columns)}개 {list(features_df.columns)}")

# ============================================
# 2. 정규화 — 각 특성별로 따로!
# ============================================
scaler_dict = {}
scaled_df = pd.DataFrame(index=features_df.index)

for col in features_df.columns:
    scaler = MinMaxScaler()
    scaled_df[col] = scaler.fit_transform(features_df[[col]]).flatten()
    scaler_dict[col] = scaler

scaled = scaled_df.values
close_scaler = scaler_dict['Close']  # 종가 복원용

print("  정규화: 특성별 개별 MinMaxScaler 적용")

# ============================================
# 3. 학습 데이터 생성
# ============================================
print("\n[2단계] 학습 데이터 생성 중...")

WINDOW = 60
close_idx = list(features_df.columns).index('Close')

X = []
y = []

for i in range(WINDOW, len(scaled)):
    X.append(scaled[i - WINDOW:i])
    y.append(scaled[i, close_idx])

X = np.array(X)
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

print(f"  입력 형태: {X_train.shape}")
print(f"  학습: {len(X_train)}개 | 테스트: {len(X_test)}개")

# ============================================
# 4. 모델 (적절한 크기)
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

input_size = X_train.shape[2]
model = StockLSTM_V2(input_size=input_size)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n[3단계] 모델 생성")
print(f"  구조: LSTM(64유닛, 2층) → FC(32) → 출력(1)")
print(f"  파라미터: {total_params:,}개")

# ============================================
# 5. 학습
# ============================================
print("\n[4단계] 학습 시작!")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

EPOCHS = 300
BATCH_SIZE = 64
start_time = time.time()
best_loss = float('inf')
patience_counter = 0

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
    
    if (epoch + 1) % 30 == 0:
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} | LR: {lr:.6f} | {elapsed:.0f}초")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'kospi_lstm_v2_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= 40:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

elapsed = time.time() - start_time
print(f"\n  학습 완료! {elapsed:.0f}초 ({elapsed/60:.1f}분)")

model.load_state_dict(torch.load('kospi_lstm_v2_best.pth'))

# ============================================
# 6. 예측
# ============================================
print("\n[5단계] 예측 중...")

model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()

# 정규화 복원 (종가 scaler 사용)
pred_prices = close_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
actual_prices = close_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

# 지표 계산
mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100

pred_diff = np.diff(pred_prices)
actual_diff = np.diff(actual_prices)
# 방향이 0인 경우 제외
valid = (pred_diff != 0) & (actual_diff != 0)
if valid.sum() > 0:
    direction_correct = np.sum(
        np.sign(pred_diff[valid]) == np.sign(actual_diff[valid])
    ) / valid.sum() * 100
else:
    direction_correct = 0

print(f"\n{'=' * 60}")
print(f"예측 결과 비교")
print(f"{'=' * 60}")
print(f"{'지표':<20} {'V1 (기본)':<15} {'V2 (수정)':<15}")
print(f"{'-' * 50}")
print(f"{'입력 특성':<20} {'종가 1개':<15} {f'{input_size}개':<15}")
print(f"{'LSTM 구조':<20} {'2층/64유닛':<15} {'2층/64유닛+FC':<15}")
print(f"{'오차율 (MAPE)':<20} {'2.97%':<15} {f'{mape:.2f}%':<15}")
print(f"{'방향 정확도':<20} {'55.1%':<15} {f'{direction_correct:.1f}%':<15}")
print(f"{'최근 실제 종가':<20} {'5,583':<15} {f'{actual_prices[-1]:,.0f}':<15}")
print(f"{'최근 예측 종가':<20} {'5,974':<15} {f'{pred_prices[-1]:,.0f}':<15}")
print(f"{'=' * 60}")

# ============================================
# 7. 차트
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(actual_prices, label='Actual', color='blue', linewidth=1)
axes[0].plot(pred_prices, label='Predicted', color='red', linewidth=1, alpha=0.7)
axes[0].set_title('KOSPI - Full Test Period')
axes[0].set_ylabel('KOSPI Index')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(actual_prices[-100:], label='Actual', color='blue', linewidth=1.5)
axes[1].plot(pred_prices[-100:], label='Predicted', color='red', linewidth=1.5, linestyle='--')
axes[1].set_title('KOSPI - Last 100 Days')
axes[1].set_xlabel('Days')
axes[1].set_ylabel('KOSPI Index')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kospi_lstm_v2_result.png', dpi=150)
print("\n차트: kospi_lstm_v2_result.png")
print("모델: kospi_lstm_v2_best.pth")
print("\n완료!")
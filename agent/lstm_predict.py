import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("LSTM 주가 예측 - 코스피/코스닥")
print("=" * 60)

print("\n[1단계] 데이터 다운로드 중...")
df = fdr.DataReader('KS11', '2015-01-01')
closes = df['Close'].values.reshape(-1,1)
print(f"  총 데이터: {len(closes)}일")
print(f"  기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

scaler = MinMaxScaler()
scaled = scaler.fit_transform(closes)

print("\n[2단계] 학습 데이터 생성 중...")

WINDOW = 30
FUTURE = 1

X = []
y = []

for i in range(WINDOW, len(scaled) - FUTURE):
    X.append(scaled[i - WINDOW:i, 0])
    y.append(scaled[i + FUTURE - 1, 0])

X = np.array(X)
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.FloatTensor(X_train).unsqueeze(-1)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_test = torch.FloatTensor(y_test)

print(f"  학습 데이터: {len(X_train)}개")
print(f"  테스트 데이터: {len(X_test)}개")


class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:,-1,:])
        return out.squeeze()

model = StockLSTM()
print(f"\n[3단계] 모델 생성 완료")
print(f"  구조: 입력(30일) -> LSTM(64유닛, 2층) -> 출력(1일 예측)")


print(f"\n[4단계] 학습 시작!")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
BATCH_SIZE = 64

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0


    for i in range(0, len(X_train), BATCH_SIZE):
        batch_X = X_train[i:i + BATCH_SIZE]
        batch_y = y_train[i:i + BATCH_SIZE]

        pred = model(batch_X)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / (len(X_train) / BATCH_SIZE)
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{EPOCHS}  |  Loss: {avg_loss:.6f}  |  경과: {elapsed:.0f}초")

elapsed = time.time() - start_time
print(f"\n 학습완료! 총 소요시간: {elapsed:.0f}초")

print("\n[5단계] 예측 실행 중...")

model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()

pred_prices = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
actual_prices = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
direction_correct = np.sum(np.sign(np.diff(pred_prices)) == np.sign(np.diff(actual_prices))) / len(np.diff(actual_prices)) * 100

print(f"\n{'=' * 60}")
print(f"예측 결과 (코스피 지수)")
print(f"{'=' * 60}")
print(f"  테스트 기간: 최근 {len(X_test)}일")
print(f"  평균 오차율(MAPE): {mape:.2f}%")
print(f"  방향 정확도: {direction_correct:.1f}% (오를 지 내릴 지 맞춘 비율)")
print(f"  최근 실제 종가: {actual_prices[-1]:,.0f}")
print(f"  최근 예측 종가: {pred_prices[-1]:,.0f}")
print(f" {'=' * 60}")

# 아래는 차트

plt.figure(figsize=(14, 6))
plt.plot(actual_prices[-100:], label='Actual (Real)', color='blue', linewidth=1.5)
plt.plot(pred_prices[-100:], label='Predicted (LSTM)', color='red', linewidth=1.5, linestyle='--')
plt.title('KOSPI Index - LSTM Prediction vs Actual (Last 100 days)')
plt.xlabel('Days')
plt.ylabel('KOSPI Index')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.savefig('kospi_lstm_result.png', dpi=150)
print("\n차트 저장: kospi_lstm_result.png")

torch.save(model.state_dict(), 'kospi_lstm_model.pth')
print("모델 저장: kospi_lstm_model.pth")
print("\n완료!")

import sys
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import FinanceDataReader as fdr

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

current_code = 'AAPL'

def is_korean_stock(code):
    """한국 종목인지 판별"""
    code = code.replace('.KS', '').replace('.KQ', '')
    if code in ['KS11', 'KQ11', 'KS200']:
        return True
    if code.isdigit() and len(code) == 6:
        return True
    return False

def clean_code(code):
    """종목 코드 정리: .KS, .KQ 제거"""
    return code.replace('.KS', '').replace('.KQ', '')

def fetch_stock_data(code, start_date, end_date):
    code = clean_code(code)
    """종목 코드에 따라 적절한 라이브러리로 데이터 가져오기"""
    if is_korean_stock(code):
        df = fdr.DataReader(code, start_date, end_date)
        # FinanceDataReader는 컬럼명이 이미 Open, High, Low, Close, Volume
        return df
    else:
        ticker = yf.Ticker(code)
        df = ticker.history(start=start_date, end=end_date)
        return df

def calc_ema(series, period):
    """EMA(지수이동평균) 계산"""
    return series.ewm(span=period, adjust=False).mean()

def strategy_ema(df, fast=10, slow=30):
    """EMA 크로스오버 전략: 빠른 EMA가 느린 EMA를 상향 돌파하면 매수"""
    ema_fast = calc_ema(df['Close'], fast)
    ema_slow = calc_ema(df['Close'], slow)
    
    buy = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    sell = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    return buy.astype(int), sell.astype(int)

def strategy_rsi(df, period=14, oversold=30, overbought=70):
    """RSI 전략: 과매도면 매수, 과매수면 매도"""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    buy = (rsi < oversold) & (rsi.shift(1) >= oversold)
    sell = (rsi > overbought) & (rsi.shift(1) <= overbought)
    return buy.astype(int), sell.astype(int)

def strategy_bbb(df, window=20, std=2.0):
    """볼린저 밴드 전략: 하한선 터치하면 매수, 중간선 터치하면 매도"""
    middle = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    upper = middle + (rolling_std * std)
    lower = middle - (rolling_std * std)
    
    buy = (df['Close'] <= lower).astype(int)
    sell = (df['Close'] >= middle).astype(int)
    return buy, sell

def strategy_ttm(df, bb_len=20, bb_std=1.5, kc_len=20, kc_mult=1.5):
    """TTM Squeeze 전략 (간략화): 스퀴즈 해제 + 모멘텀 상승이면 매수"""
    # 볼린저 밴드
    bb_mid = df['Close'].rolling(bb_len).mean()
    bb_std_val = df['Close'].rolling(bb_len).std()
    bb_upper = bb_mid + bb_std_val * bb_std
    bb_lower = bb_mid - bb_std_val * bb_std
    
    # 켈트너 채널 (ATR 기반)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(kc_len).mean()
    kc_upper = bb_mid + atr * kc_mult
    kc_lower = bb_mid - atr * kc_mult
    
    # 스퀴즈: BB가 KC 안에 있으면 스퀴즈 상태
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    squeeze_off = ~squeeze_on
    
    # 모멘텀
    momentum = df['Close'] - bb_mid
    mom_rising = momentum > momentum.shift(1)
    
    buy = (squeeze_off & squeeze_on.shift(1) & mom_rising).astype(int)
    sell = (squeeze_off & (momentum < 0)).astype(int)
    return buy, sell

def strategy_macd(df, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(df['Close'], fast)
    ema_slow = calc_ema(df['Close'], slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    buy = (
        (macd_line > signal_line) &
        (macd_line.shift(1) <= signal_line.shift(1))
    )

    sell = (
        (macd_line < signal_line) &
        (macd_line.shift(1) >= signal_line.shift(1))
    )

    return buy.astype(int), sell.astype(int)

def strategy_ema3(df, short=8, mid=21, long=50):
    ema_short = calc_ema(df['Close'], short)
    ema_mid = calc_ema(df['Close'], mid)
    ema_long = calc_ema(df['Close'], long)

    buy = (
        (ema_short > ema_mid) &
        (ema_short.shift(1) <= ema_mid.shift(1)) &
        (ema_mid > ema_long)
    )

    sell = (
        (ema_short < ema_mid) &
        (ema_short.shift(1) >= ema_mid.shift(1))
    )

    return buy.astype(int), sell.astype(int)

def strategy_supertrend(df, period=10, multiplier=3.0):
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    mid = (high + low) / 2
    upper_band = mid + (multiplier * atr)
    lower_band = mid - (multiplier * atr)

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(df)):
        if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
            lower_band.iloc[i] = lower_band.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i-1]
        
        if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
            upper_band.iloc[i] = upper_band.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i-1]

        
        if direction.iloc[i-1] == 1:
            if close.iloc[i] < lower_band.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]
        else:
            if close.iloc[i] > upper_band.iloc[i]:
                direction.iloc[i] = 1 
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                direction.iloc[i] = -1  
                supertrend.iloc[i] = upper_band.iloc[i]
    
    buy = ((direction == 1) & (direction.shift(1) == -1)).astype(int)
    sell = ((direction == -1) & (direction.shift(1) == 1)).astype(int)

    return buy, sell

def strategy_psar(df, af_start=0.02, af_max=0.2):
    high = df['High']
    low = df['Low']
    close = df['Close']

    n = len(df)
    sar = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    sar.iloc[0] = low.iloc[0]
    direction.iloc[0] = 1
    af = af_start
    ep = high.iloc[0]

    for i in range(1, n):
        prev_sar = sar.iloc[i-1]

        if direction.iloc[i-1] == 1:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = min(new_sar, low.iloc[i-1])
            if i >= 2:
                new_sar = min(new_sar, low.iloc[i-2])
            
            if close.iloc[i] < new_sar:
                direction.iloc[i] = -1
                sar.iloc[i] = ep
                af = af_start
                ep = low.iloc[i]
            else:
                direction.iloc[i] = 1
                sar.iloc[i] = new_sar
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_start, af_max)
        
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = max(new_sar, high.iloc[i-1])
            if i >= 2:
                new_sar = max(new_sar, high.iloc[i-2])
            
            if close.iloc[i] > new_sar:
                direction.iloc[i] = 1
                sar.iloc[i] = ep
                af = af_start
                ep = high.iloc[i]
            else:
                direction.iloc[i] = -1
                sar.iloc[i] = new_sar
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_start, af_max)
    
    buy = ((direction == 1) & (direction.shift(1) == -1)).astype(int)
    sell = ((direction == -1) & (direction.shift(1) == 1)).astype(int)

    return buy, sell

# ============================================
# LSTM 모델 정의 (학습 때와 동일해야 함)
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
# LSTM 전략 함수
# ============================================
def strategy_lstm(df):
    """LSTM AI 전략: 예측 종가가 현재보다 높으면 매수, 낮으면 매도"""
    
    global current_code
    
    # 모델 파일 경로 (backtest_runner.py와 같은 폴더에 .pth가 있어야 함)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 종목별 모델 찾기
    safe_code = current_code.replace('^', '').replace('/', '_')
    model_path = os.path.join(script_dir, 'models', f'{safe_code}_lstm.pth')

    
    # 모델 파일이 없으면 신호 없음으로 리턴
    if not os.path.exists(model_path):
        print("[LSTM] 모델 파일 없음, 신호 없이 진행", file=sys.stderr)
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index)
    
    # 1. 특성 생성 (학습 때와 동일하게!)
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
    
    # 결측치를 앞쪽 값으로 채우기
    features_df.ffill(inplace=True)
    features_df.fillna(0, inplace=True)
    
    # 2. 정규화 (특성별 개별 scaler)
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
    
    close_scaler = scaler_dict['Close']
    scaled = scaled_df.values
    
    # 3. 모델 로드
    input_size = len(features_df.columns)
    model = StockLSTM_V2(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    # 4. 매일 예측해서 매수/매도 신호 생성
    WINDOW = 60
    close_idx = list(features_df.columns).index('Close')
    
    buy_signals = pd.Series(0, index=df.index)
    sell_signals = pd.Series(0, index=df.index)
    
    for i in range(WINDOW, len(scaled)):
        # 과거 60일 데이터로 내일 종가 예측
        window_data = scaled[i - WINDOW:i]
        X = torch.FloatTensor(window_data).unsqueeze(0)
        
        with torch.no_grad():
            pred_scaled = model(X).item()
        
        # 예측 종가를 원래 스케일로 복원
        pred_price = close_scaler.inverse_transform([[pred_scaled]])[0][0]
        current_price = df['Close'].iloc[i]
        
        # 예측이 현재보다 0.5% 이상 높으면 매수
        # 예측이 현재보다 0.5% 이상 낮으면 매도
        # (0.5% 기준을 두는 이유: 너무 작은 차이에 반응하면 거짓 신호가 많아져서)
        change_pct = (pred_price - current_price) / current_price
        
        if change_pct > 0.005:
            buy_signals.iloc[i] = 1
        elif change_pct < -0.005:
            sell_signals.iloc[i] = 1
    
    print(f"[LSTM] 매수신호: {buy_signals.sum()}개, 매도신호: {sell_signals.sum()}개", file=sys.stderr)
    
    return buy_signals.astype(int), sell_signals.astype(int)

# ============================================
# 앙상블 투표 (MainStrategy 로직)
# ============================================

STRATEGIES = {
    'EMA': strategy_ema,
    'RSI': strategy_rsi,
    'BBB': strategy_bbb,
    'TTM': strategy_ttm,
    'MACD' : strategy_macd,
    'EMA3' : strategy_ema3,
    'SUT' : strategy_supertrend,
    'PSAR' : strategy_psar,
    'LSTM': strategy_lstm,
}

def run_ensemble(df, active_strategies, threshold=2):
    """활성 전략들의 신호를 합산해서 최종 매수/매도 결정"""
    buy_signals = pd.DataFrame(index=df.index)
    sell_signals = pd.DataFrame(index=df.index)
    
    for name in active_strategies:
        if name in STRATEGIES:
            buy, sell = STRATEGIES[name](df)
            buy_signals[name] = buy
            sell_signals[name] = sell
    
    # 투표: threshold 이상이면 매수/매도
    final_buy = (buy_signals.sum(axis=1) >= threshold).astype(int)
    final_sell = (sell_signals.sum(axis=1) >= threshold).astype(int)
    
    return final_buy, final_sell


# ============================================
# 백테스트 엔진
# ============================================

def run_backtest(code, start_date, end_date, active_strategies,
                 threshold=2, initial_cash=10000):
    global current_code
    code = clean_code(code)
    current_code = code
    
    df = fetch_stock_data(code, start_date, end_date)

    if df.empty:
        print(json.dumps({"error": f"데이터 없음: {code}"}))
        return None
    
    # 2. 앙상블 신호 생성
    buy_signal, sell_signal = run_ensemble(
        df, active_strategies, threshold)
    
    # 3. 가상 매매 시뮬레이션
    cash = initial_cash
    shares = 0
    trades = []
    position_open = False
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i].strftime('%Y-%m-%d')
        
        if buy_signal.iloc[i] == 1 and not position_open:
            # 매수: 전액 투자
            shares = cash / price
            cash = 0
            position_open = True
            trades.append({
                'type': 'BUY', 'date': date,
                'price': round(price, 2)
            })
            
        elif sell_signal.iloc[i] == 1 and position_open:
            # 매도: 전량 매도
            cash = shares * price
            shares = 0
            position_open = False
            trades.append({
                'type': 'SELL', 'date': date,
                'price': round(price, 2)
            })
    
    # 마지막에 포지션 열려있으면 종가로 청산
    if position_open:
        cash = shares * df['Close'].iloc[-1]
        shares = 0
    
    # 4. 결과 계산
    final_value = cash
    total_profit_pct = ((final_value - initial_cash) /
                        initial_cash) * 100
    
    # 승/패 계산
    wins = 0
    losses = 0
    draws = 0
    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades):
            buy_price = trades[i]['price']
            sell_price = trades[i + 1]['price']
            if sell_price > buy_price:
                wins += 1
            elif sell_price < buy_price:
                losses += 1
            else:
                draws += 1
    
    total_trades = wins + losses + draws
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # MDD (최대 낙폭) 계산
    portfolio_values = []
    temp_cash = initial_cash
    temp_shares = 0
    temp_open = False
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        if buy_signal.iloc[i] == 1 and not temp_open:
            temp_shares = temp_cash / price
            temp_cash = 0
            temp_open = True
        elif sell_signal.iloc[i] == 1 and temp_open:
            temp_cash = temp_shares * price
            temp_shares = 0
            temp_open = False
        
        value = temp_cash + (temp_shares * price)
        portfolio_values.append(value)
    
    portfolio = pd.Series(portfolio_values)
    peak = portfolio.cummax()
    drawdown = (portfolio - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    result = {
        'code': code,
        'timeframe': '1d',
        'strategies': ','.join(active_strategies),
        'mainModel': 'Ensemble',
        'entryThreshold': threshold,
        'startDate': start_date + "T00:00:00",
        'endDate': end_date + "T00:00:00",
        'totalTrades': total_trades,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'winRate': round(win_rate, 1),
        'totalProfitPct': round(total_profit_pct, 1),
        'maxDrawdownPct': round(max_drawdown, 1),
        'trades': trades
    }
    
    return result


def run_simulation(code, start_date, end_date, active_strategies,
                   threshold=2, initial_cash=10000):
    global current_code
    code = clean_code(code)
    current_code = code
    
    df = fetch_stock_data(code, start_date, end_date)
    
    if df.empty:
        return None
    
    buy_signal, sell_signal = run_ensemble(
        df, active_strategies, threshold)
    
    cash = initial_cash
    shares = 0
    position_open = False
    
    # 매일 자산 기록
    daily_portfolio = []
    trades = []
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i].strftime('%Y-%m-%d')
        
        if buy_signal.iloc[i] == 1 and not position_open:
            shares = cash / price
            cash = 0
            position_open = True
            trades.append({
                'type': 'BUY', 'date': date,
                'price': round(price, 2)
            })
            
        elif sell_signal.iloc[i] == 1 and position_open:
            cash = shares * price
            shares = 0
            position_open = False
            trades.append({
                'type': 'SELL', 'date': date,
                'price': round(price, 2)
            })
        
        # 오늘의 총 자산
        total_value = cash + (shares * price)
        daily_portfolio.append({
            'date': date,
            'value': round(total_value, 2),
            'price': round(price, 2)
        })
    
    # 마지막 포지션 정리
    if position_open:
        cash = shares * df['Close'].iloc[-1]
        shares = 0
        position_open = False
    
    final_value = cash
    total_return = ((final_value - initial_cash) /
                    initial_cash) * 100
    
    return {
        'code': code,
        'initialCash': initial_cash,
        'finalValue': round(final_value, 2),
        'totalReturn': round(total_return, 1),
        'strategies': ','.join(active_strategies),
        'threshold': threshold,
        'startDate': start_date,
        'endDate': end_date,
        'trades': trades,
        'portfolio': daily_portfolio
    }
# ============================================
# 메인 실행
# ============================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error": "파라미터가 필요합니다"}))
        sys.exit(1)
    
    arg = sys.argv[1]
    
    # 파일 경로면 파일에서 읽기, 아니면 JSON 문자열로 파싱
    if arg.endswith('.json'):
        with open(arg, 'r') as f:
            params = json.load(f)
    else:   
        params = json.loads(arg)
    
    code = params.get('code', 'AAPL')
    start_date = params.get('startDate', '2024-01-01')
    end_date = params.get('endDate', '2024-12-31')
    strategies = params.get('strategies', ['EMA', 'RSI'])
    threshold = params.get('threshold', 2)
    mode = params.get('mode', 'backtest')
    initial_cash = params.get('initial_cash', 10000)
    
    if mode == 'simulation':
        result = run_simulation(code, start_date, end_date, strategies, threshold, initial_cash)
        if result:
            print(json.dumps(result, cls=NpEncoder))
    else:
        result = run_backtest(
            code, start_date, end_date, strategies, threshold)
        if result:
            api_url = "http://localhost:8080/api/backtest/save"
            try:
                save_data = {k: v for k, v in result.items()
                            if k != 'trades'}
                response = requests.post(api_url, json=save_data)
                print(f"[저장 완료] {response.status_code}",
                    file=sys.stderr)
            except Exception as e:
                print(f"[저장 실패] {e}", file=sys.stderr)
            print(json.dumps(result, cls=NpEncoder))
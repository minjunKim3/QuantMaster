import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, ticker):
        self.ticker = ticker
        self.end_date = datetime.now()
        # 과거 패턴을 충분히 찾기 위해 약 1.5년(500일) 치 데이터 로드
        self.start_date = (self.end_date - timedelta(days=500)).strftime('%Y-%m-%d')

    def get_data(self, window_size=20):
        try:
            df = fdr.DataReader(self.ticker, self.start_date, self.end_date.strftime('%Y-%m-%d'))
            if df.empty or len(df) < window_size * 2:
                return pd.DataFrame(), pd.DataFrame()

            # 사용자 기존 지표 로직 적용 (추후 AI 모델 피처로 활용 가능)
            df = self._calculate_indicators(df)
            
            # 결측치 제거
            df = df.dropna()

            # 현재 비교할 기준이 되는 최근 window_size 만큼의 데이터
            df_current = df.tail(window_size).copy()
            
            # 과거 전체 데이터 (현재 윈도우 기간은 제외하여 자기 자신과 매칭되는 것 방지)
            df_all = df.iloc[:-window_size].copy()

            return df_all, df_current
        except Exception as e:
            print(f"데이터 로드 중 오류: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def _calculate_indicators(self, df):
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        df['rsi'] = 100 - (100 / (1 + (ema_up / ema_down)))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()

        ma20 = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        upper_bb = ma20 + (std * 2)
        lower_bb = ma20 - (std * 2)
        
        # 분모가 0이 되는 것 방지
        bb_range = upper_bb - lower_bb
        bb_range = bb_range.replace(0, np.nan)
        df['bb_pct'] = (df['Close'] - lower_bb) / bb_range
        df['ma_gap'] = (df['Close'] / ma20) - 1

        return df
import FinanceDataReader as fdr
from pykrx import stock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StockEngine:
    def __init__(self):
        self.target_date = self._get_latest_business_day()
        print(f"분석 기준 영업일: {self.target_date}")

    def _get_latest_business_day(self):
        curr = datetime.now()
        if curr.weekday() == 5: curr -= timedelta(days=1)
        elif curr.weekday() == 6: curr -= timedelta(days=2)
        for i in range(15):
            check_dt = curr - timedelta(days=i)
            if check_dt.weekday() >= 5: continue
            date_query = check_dt.strftime('%Y-%m-%d')
            try:
                df = fdr.DataReader('KS11', date_query, date_query)
                if not df.empty and df['Close'].iloc[0] > 0:
                    return date_query
            except: continue
        return curr.strftime('%Y-%m-%d')

    def _calculate_indicators(self, df):
        # RSI
        delta = df['Close'].diff()
        up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rsi = 100 - (100 / (1 + (ema_up / ema_down)))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        ma20 = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        upper_bb = ma20 + (std * 2); lower_bb = ma20 - (std * 2)
        bb_pct = (df['Close'] - lower_bb) / (upper_bb - lower_bb)

        # 이격도
        ma_gap = (df['Close'] / ma20) - 1

        return {
            'rsi': rsi.iloc[-1], 'macd': macd.iloc[-1], 'macd_sig': signal.iloc[-1],
            'bb_pct': bb_pct.iloc[-1], 'ma_gap': ma_gap.iloc[-1]
        }

    def get_filtered_candidates(self, params):
        days = params.get('days', 7)
        weights = params.get('weights', {})
        
        print(f"맞춤형 지표 가중치 적용 (300개 종목 스캔)...")
        df_listing = fdr.StockListing('KRX').head(300)
        results = []
        start_date = (datetime.now() - timedelta(days=days + 60)).strftime('%Y-%m-%d')

        for _, row in df_listing.iterrows():
            ticker, name = row['Code'], row['Name']
            try:
                df = fdr.DataReader(ticker, start_date)
                if len(df) < 30: continue
                
                ind = self._calculate_indicators(df)
                curr_price = int(df['Close'].iloc[-1])
                price_chg = ((curr_price - df['Close'].iloc[-(days+1)]) / df['Close'].iloc[-(days+1)]) * 100
                vol_ratio = df['Volume'].iloc[-1] / df['Volume'].iloc[-(days+3):-1].mean() if df['Volume'].iloc[-(days+3):-1].mean() > 0 else 1.0
                volatility = df['Close'].pct_change().tail(days).std()

                # 동적 가중치 스코어링 로직 (사용자 요청 반영)
                score = 50
                score += price_chg * weights.get('price_chg', 0)
                score += (vol_ratio - 1) * weights.get('vol_ratio', 0) * 10
                score += (30 - ind['rsi']) * weights.get('rebound', 0) 
                
                if ind['bb_pct'] < 0.2: score += weights.get('rebound', 0) * 20
                if abs(ind['ma_gap']) < 0.05: score += weights.get('stability', 0) * 30
                if ind['macd'] > ind['macd_sig']: score += weights.get('trend', 0) * 20
                # 변동성 키워드 대응 추가
                score += volatility * weights.get('volatility', 0) * 100

                results.append({
                    'ticker': ticker, 'name': name, 'score': round(score, 2),
                    'price_chg': round(price_chg, 2), 'rsi': round(ind['rsi'], 2),
                    'vol_ratio': round(vol_ratio, 2), 'macd_status': "상승" if ind['macd'] > ind['macd_sig'] else "조정",
                    'current_price': curr_price
                })
            except: continue
            
        return sorted(results, key=lambda x: x['score'], reverse=True)[:50]
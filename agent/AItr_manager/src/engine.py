import FinanceDataReader as fdr
from pykrx import stock
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta


def _nz(v, default=0.0):
    """NaN/Inf 안전 가드 — finite 아니면 default 반환."""
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default

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
                # 20260601: NaN 가드 — 데이터 부족·결측치로 인한 NaN이 점수까지 전파되어
                # UI에서 "점수 공란"으로 보이던 버그 수정. 모든 지표를 _nz로 감싼다.
                price_chg = _nz(((curr_price - df['Close'].iloc[-(days+1)]) / df['Close'].iloc[-(days+1)]) * 100)
                vol_mean = df['Volume'].iloc[-(days+3):-1].mean()
                vol_ratio = _nz(df['Volume'].iloc[-1] / vol_mean, default=1.0) if (vol_mean and vol_mean > 0) else 1.0
                volatility = _nz(df['Close'].pct_change().tail(days).std())

                rsi = _nz(ind['rsi'], default=50.0)
                bb_pct = _nz(ind['bb_pct'], default=0.5)
                ma_gap = _nz(ind['ma_gap'], default=0.0)
                macd = _nz(ind['macd'])
                macd_sig = _nz(ind['macd_sig'])

                # 동적 가중치 스코어링 로직 (사용자 요청 반영)
                # 20260602: 기존 if 이산 보너스 3개가 가중치 sparse 할 때 score 를 2~4개
                # 버킷으로 뭉치게 만들어 "25개 80점, 25개 74점" 양극화가 발생했음.
                # 같은 임계값에서 full bonus, 임계 밖에서 0 이 되도록 선형 ramp 로 교체 →
                # 종목별 미세 차이가 score 에 반영되어 88.9 / 88.5 같은 연속 분포로 회복.
                score = 50.0
                score += price_chg * weights.get('price_chg', 0)
                score += (vol_ratio - 1) * weights.get('vol_ratio', 0) * 10
                score += (30 - rsi) * weights.get('rebound', 0)

                score += max(0.0, (0.2 - bb_pct) / 0.2) * weights.get('rebound', 0) * 20
                score += max(0.0, (0.05 - abs(ma_gap)) / 0.05) * weights.get('stability', 0) * 30
                macd_diff = macd - macd_sig
                macd_denom = max(abs(macd_sig), abs(macd), 1e-6)
                score += max(0.0, min(1.0, macd_diff / macd_denom)) * weights.get('trend', 0) * 20
                score += volatility * weights.get('volatility', 0) * 100

                # 최종 안전망: score 자체가 어쩌다 NaN이 되면 기본값 50으로 폴백 (정렬도 안전해짐)
                score = _nz(score, default=50.0)

                results.append({
                    'ticker': ticker, 'name': name, 'score': round(score, 2),
                    'price_chg': round(price_chg, 2), 'rsi': round(rsi, 2),
                    'vol_ratio': round(vol_ratio, 2),
                    'macd_status': "상승" if macd > macd_sig else "조정",
                    'current_price': curr_price
                })
            except: continue
            
        return sorted(results, key=lambda x: x['score'], reverse=True)[:50]
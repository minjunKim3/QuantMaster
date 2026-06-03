import pandas as pd

class Simulator:
    # 단일 매칭이 아닌 상위 매칭 리스트(top_matches)를 받도록 수정됨
    def __init__(self, top_matches, amount, pred_days=10):
        self.top_matches = top_matches
        self.amount = amount
        self.pred_days = pred_days

    def calculate_scenario(self):
        if not self.top_matches:
            return self._default_empty_result()

        best_upside_pct = -float('inf')
        worst_downside_pct = float('inf')
        
        bull_match = None
        bull_future = None
        bear_match = None
        bear_future = None

        for match in self.top_matches:
            idx = match['best_idx']
            df_all = match['full_past_df']
            window_size = len(match['matched_df'])
            
            # 과거 패턴이 끝난 바로 다음 날부터 pred_days 만큼의 미래 데이터 추출
            future_start_idx = idx + window_size
            future_end_idx = future_start_idx + self.pred_days
            
            # 미래 데이터가 부족하면 있는 데까지만 사용 (에러 방지)
            if future_start_idx >= len(df_all):
                continue
                
            future_df = df_all.iloc[future_start_idx : future_end_idx]
            
            if future_df.empty:
                continue

            # 과거 패턴의 마지막 날 종가 (미래 변동률 계산의 기준점)
            base_price = match['matched_df']['Close'].iloc[-1]
            
            # 미래 구간에서의 최고가와 최저가
            max_price = future_df['High'].max()
            min_price = future_df['Low'].min()
            
            # [수정됨] 실제 주식 수익률 공식 정상 반영 (부호 오류 해결)
            up_pct = ((max_price - base_price) / base_price) * 100
            down_pct = ((min_price - base_price) / base_price) * 100
            
            # 가장 높이 올라간 긍정적 시나리오 갱신
            if up_pct > best_upside_pct:
                best_upside_pct = up_pct
                bull_match = match
                bull_future = future_df
                
            # 가장 깊게 떨어진 부정적 시나리오 갱신
            if down_pct < worst_downside_pct:
                worst_downside_pct = down_pct
                bear_match = match
                bear_future = future_df

        # 조건을 만족하는 시나리오가 하나도 없었다면 기본값 반환
        if bull_match is None or bear_match is None:
            return self._default_empty_result()

        return {
            'bull_scenario': {
                'match': bull_match,
                'future_df': bull_future,
                'pct': best_upside_pct
            },
            'bear_scenario': {
                'match': bear_match,
                'future_df': bear_future,
                'pct': worst_downside_pct
            },
            # 이전 main.py와의 변수 호환성을 위해 상위 레벨에도 pct 저장
            'max_upside_pct': best_upside_pct,
            'max_downside_pct': worst_downside_pct
        }

    def _default_empty_result(self):
        return {
            'bull_scenario': {'match': None, 'future_df': pd.DataFrame(), 'pct': 0.0},
            'bear_scenario': {'match': None, 'future_df': pd.DataFrame(), 'pct': 0.0},
            'max_upside_pct': 0.0,
            'max_downside_pct': 0.0
        }
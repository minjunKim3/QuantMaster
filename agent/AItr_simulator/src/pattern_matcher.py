import numpy as np
import pandas as pd
from fastdtw import fastdtw

class PatternMatcher:
    def __init__(self, df_all, df_current):
        self.df_all = df_all
        self.df_current = df_current

    def _min_max_scale(self, series):
        """형태 비교를 위해 가격을 0~1 사이로 비율 스케일링"""
        s_min = np.min(series)
        s_max = np.max(series)
        if s_max - s_min == 0:
            return np.zeros(len(series))
        return (series - s_min) / (s_max - s_min)

    def find_top_matches(self, window_size=20, pred_days=20, top_n=5):
        """상승/하락 다중 시나리오를 위해 겹치지 않는 가장 유사한 패턴 상위 N개를 찾음"""
        current_shape = self._min_max_scale(self.df_current['Close'].values)
        
        # 시뮬레이션을 돌리려면 과거 패턴 뒤에 '미래 데이터(pred_days)'가 존재해야 함
        search_limit = len(self.df_all) - window_size - pred_days
        
        if search_limit <= 0:
            print("스캔할 수 있는 과거 데이터가 부족합니다.")
            return []
            
        distances = []
        for i in range(search_limit):
            past_window = self.df_all['Close'].iloc[i : i + window_size].values
            past_shape = self._min_max_scale(past_window)
            
            # fastdtw 에러 방지를 위해 euclidean 거리 파라미터 제외
            dist, _ = fastdtw(current_shape, past_shape)
            distances.append({'idx': i, 'distance': dist})
            
        # 1. 거리(유사도) 순으로 오름차순 정렬 (거리가 짧을수록 유사함)
        distances = sorted(distances, key=lambda x: x['distance'])
        
        # 2. 기간이 겹치지 않는(Non-overlapping) 상위 N개 추출
        top_matches = []
        for d in distances:
            idx = d['idx']
            # 이미 선택된 패턴들과 기간이 겹치는지 확인 (window_size 기준)
            overlap = any(abs(idx - m['best_idx']) < window_size for m in top_matches)
            
            if not overlap:
                top_matches.append({
                    'best_idx': idx,
                    'distance': d['distance'],
                    'start_date': self.df_all.index[idx],
                    'end_date': self.df_all.index[idx + window_size - 1],
                    'matched_df': self.df_all.iloc[idx : idx + window_size].copy(),
                    'full_past_df': self.df_all 
                })
            
            # 원하는 개수(top_n)를 모두 찾으면 탐색 종료
            if len(top_matches) >= top_n:
                break

        return top_matches
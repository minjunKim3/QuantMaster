import plotly.graph_objects as go
import pandas as pd

class Visualizer:
    # main.py 버전에 따라 match_result가 넘어올 수도, 안 넘어올 수도 있으므로 기본값 추가
    def __init__(self, df_current, sim_data, match_result=None):
        self.df_current = df_current
        self.sim_data = sim_data
        self.match_result = match_result 

    def generate_chart(self):
        fig = go.Figure()

        if self.df_current.empty or not self.sim_data or 'bull_scenario' not in self.sim_data:
            fig.add_annotation(text="시각화할 데이터가 부족합니다.", showarrow=False)
            return fig

        # 1. 현재 주가 라인 추가
        x_current = list(range(len(self.df_current)))
        y_current = self.df_current['Close'].values
        current_start_price = y_current[0]
        
        fig.add_trace(go.Scatter(
            x=x_current, y=y_current,
            mode='lines', name='현재 주가 흐름',
            line=dict(color='blue', width=4)
        ))

        # 반복되는 그리기 로직을 헬퍼 함수로 통합하여 코드 중복 제거
        def plot_scenario(scenario_data, name, color):
            if scenario_data['match'] is None:
                return
                
            matched_df = scenario_data['match']['matched_df']
            future_df = scenario_data['future_df']
            past_start_price = matched_df['Close'].iloc[0]
            
            # 시작점을 기준으로 배율을 구해서 과거 차트를 위아래로 이동시킴 (스케일링)
            scale_ratio = current_start_price / past_start_price if past_start_price > 0 else 1
            y_matched_scaled = matched_df['Close'].values * scale_ratio
            past_date_str = scenario_data['match']['start_date'].strftime('%Y-%m-%d')
            
            # 2. 과거 유사 패턴 오버랩
            fig.add_trace(go.Scatter(
                x=x_current, y=y_matched_scaled,
                mode='lines', name=f'{name} 과거 패턴 ({past_date_str} ~)',
                line=dict(color=color, width=2, dash='dash'),
                opacity=0.4
            ))

            # 3. 과거 패턴 이후의 '미래' 투영 (예측 시뮬레이션 구간)
            if not future_df.empty:
                # 미래 구간 x축 좌표 (현재 차트 끝난 이후)
                x_future = list(range(len(self.df_current) - 1, len(self.df_current) + len(future_df)))
                
                # 이어지게 그리기 위해 마지막 날 종가를 첫 포인트로 삽입
                y_future_raw = [matched_df['Close'].iloc[-1]] + list(future_df['Close'].values)
                y_future_scaled = [val * scale_ratio for val in y_future_raw]
                
                fig.add_trace(go.Scatter(
                    x=x_future, y=y_future_scaled,
                    mode='lines', name=f'{name} 범위 추정',
                    line=dict(color=color, width=3, dash='dot')
                ))

        # 상승/하락 두 개의 시나리오를 각각 호출하여 투영
        plot_scenario(self.sim_data['bull_scenario'], "🟢 상승(Bull)", "green")
        plot_scenario(self.sim_data['bear_scenario'], "🔴 하락(Bear)", "red")

        # 레이아웃 정리
        fig.update_layout(
            title="현재 차트와 AI가 찾은 과거 상승/하락 상하단 바운더리 추정",
            xaxis_title="거래일 (0 = 패턴 시작점)",
            yaxis_title="스케일 조정된 주가 (원)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig
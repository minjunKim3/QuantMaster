import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
from src.data_loader import DataLoader
from src.pattern_matcher import PatternMatcher
from src.simulator import Simulator
from src.visualizer import Visualizer

@st.cache_data(ttl=86400) 
def load_stock_list():
    df = fdr.StockListing('KRX')
    return {f"{row['Name']} ({row['Code']})": row['Code'] for idx, row in df.iterrows()}

def main():
    st.set_page_config(page_title="주식 패턴 시뮬레이터", layout="wide")
    
    st.title("주식 패턴 매칭 시뮬레이터")
    st.write("현재 차트와 가장 유사한 과거 프랙탈 패턴을 분석합니다.")
    
    # 세션 상태 초기화
    if 'match_done' not in st.session_state:
        st.session_state.match_done = False
        st.session_state.sim_data_base = None
        st.session_state.df_current = None
        st.session_state.match_result = None

    # --- 사이드바 ---
    st.sidebar.header("1. 분석 설정")
    stock_dict = load_stock_list()
    selected_stock_name = st.sidebar.selectbox(
        "종목 검색", 
        options=list(stock_dict.keys()),
        index=None, 
        placeholder="종목명 또는 코드 입력"
    )
    
    if not selected_stock_name:
        st.info("좌측 사이드바에서 분석할 종목을 선택해 주세요.")
        return 

    ticker = stock_dict[selected_stock_name]
    scan_window = 20 
    prediction_days = st.sidebar.number_input("미래 예측 기간 (일)", min_value=5, max_value=60, value=20, step=5)
    
    if st.sidebar.button("패턴 분석 시작", type="primary"):
        with st.spinner("과거 데이터 스캔 중..."):
            try:
                loader = DataLoader(ticker=ticker)
                df_all, df_current = loader.get_data(window_size=scan_window)
                
                if df_all.empty:
                    st.error("데이터 로드 실패.")
                    return

                matcher = PatternMatcher(df_all=df_all, df_current=df_current)
                top_matches = matcher.find_top_matches(window_size=scan_window, pred_days=prediction_days, top_n=5)
                
                if top_matches:
                    base_simulator = Simulator(top_matches, amount=100, pred_days=prediction_days)
                    sim_data_base = base_simulator.calculate_scenario()
                    
                    distance = top_matches[0]['distance']
                    st.session_state.similarity_pct = max(0.0, 100.0 - (distance * 15)) 
                    
                    st.session_state.match_result = top_matches[0] # 시각화를 위해 첫 번째 매칭 저장
                    st.session_state.sim_data_base = sim_data_base
                    st.session_state.df_current = df_current
                    st.session_state.prediction_days = prediction_days
                    st.session_state.match_done = True
                else:
                    st.warning("유사한 패턴을 찾지 못했습니다.")
            except Exception as e:
                st.error(f"오류 발생: {e}")

    # --- 메인 화면 ---
    if st.session_state.match_done:
        m_result = st.session_state.match_result
        s_base = st.session_state.sim_data_base
        p_days = st.session_state.prediction_days
        
        st.success(f"분석 완료! 가장 유사한 과거 시점: {m_result['start_date'].strftime('%Y-%m-%d')} (일치도: {st.session_state.similarity_pct:.1f}%)")
        
        st.subheader("2. 모의 투자 시뮬레이션")
        investment_amount = st.number_input(
            f"향후 {p_days}일간의 시나리오에 따른 투자 금액(원)", 
            min_value=0, value=1000000, step=50000, format="%d"
        )
        
        bull_pct = s_base['bull_scenario']['pct']
        bear_pct = s_base['bear_scenario']['pct']
        
        calc_upside = int(investment_amount * (1 + (bull_pct / 100)))
        calc_downside = int(investment_amount * (1 + (bear_pct / 100))) 

        col1, col2 = st.columns(2)
        col1.metric("최대 기대 수익", f"{calc_upside:,} 원", f"{bull_pct:.2f}%")
        col2.metric("최대 예상 손실", f"{calc_downside:,} 원", f"{bear_pct:.2f}%")
            
        st.subheader("3. 패턴 차트 분석")
        visualizer = Visualizer(
            df_current=st.session_state.df_current, 
            sim_data=s_base
        )
        st.plotly_chart(visualizer.generate_chart(), use_container_width=True)

        st.info("AI 예측 모듈 연동 예정 공간: AI모델 추가로 모델 예측 곡선생성 가능.")

    else:
        st.info("사이드바에서 종목을 선택하고 분석을 시작하세요.")

if __name__ == "__main__":
    main()
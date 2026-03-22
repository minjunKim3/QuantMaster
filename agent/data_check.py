import FinanceDataReader as fdr
import pandas as pd

print("=" * 60)
print("코스피/코스닥 시계열 데이터 조사")
print("=" * 60)

print("\n[1] 코스피 지수 (KS11)")
kospi = fdr.DataReader('KS11', '2015-01-01')
print(f"  기간: {kospi.index[0].strftime('%Y-%m-%d')} ~ {kospi.index[-1].strftime('%Y-%m-%d')}")
print(f"  총 데이터 수: {len(kospi)}일")
print(f"  컬럼: {list(kospi.columns)}")
print(f"  결측치: {kospi.isnull().sum().sum()}개")
print(kospi.tail(3))

print("\n[2] 코스닥 지수 (KQ11)")
kosdaq = fdr.DataReader('KQ11', '2015-01-01')
print(f"  기간: {kosdaq.index[0].strftime('%Y-%m-%d')} ~ {kosdaq.index[-1].strftime('%Y-%m-%d')}")
print(f"  총 데이터 수: {len(kosdaq)}일")
print(f"  컬럼: {list(kosdaq.columns)}")
print(f"  결측치: {kosdaq.isnull().sum().sum()}개")
print(kosdaq.tail(3))

targets = {
    '005930': '삼성전자',
    '000660': 'SK하이닉스',
    '035720': '카카오',
    '068270': '셀트리온',
    '247540': '에코프로비엠',
}

print("\n[3] 개별 종목 데이터")
print("-" * 60)

for code, name in targets.items():
    try:
        df = fdr.DataReader(code, "2015-01-01")
        print(f"  기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  총 데이터 수: {len(df)}일")
        print(f"  컬럼: {list(df.columns)}")
        print(f"  최근 종가: {df['Close'].iloc[-1]:,.0f}원")
        print()
    except Exception as e:
        print(f"  {name}({code}): 에러 - {e}")
        print()


print("[4] 전체 종목 수")
kospi_list = fdr.StockListing('KOSPI')
kosdaq_list = fdr.StockListing('KOSDAQ')
print(f"  코스피 상장 종목: {len(kospi_list)}개")
print(f"  코스피 상장 종목: {len(kosdaq_list)}개")
print(f"  합계: {len(kospi_list) + len(kosdaq_list)}개")

print("\n[5] yfinance와 한국 주식 호환성 테스트")
import yfinance as yf
yf_targets = {
    '005930.KS': '삼성전자',
    '000660.KS': 'SK하이닉스',
    "^KS11" : '코스피지수',
}

for code, name in yf_targets.items():
    try:
        ticker = yf.Ticker(code)
        df = ticker.history(start='2020-01-01')
        print(f"  {name}({code})")
        print(f"    기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"    총 데이터 수: {len(df)}일")
        print()
    except Exception as e:
        print(f"  {name}({code}): 에러 - {e}")
        print()

print("=" * 60)
print("조사 완료!")
print("=" * 60)
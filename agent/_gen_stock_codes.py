"""KRX 전체 종목 리스트를 static/stock_codes.json 에 갱신.
실행: agent/venv/Scripts/python.exe agent/_gen_stock_codes.py
"""
import json
import os
from datetime import datetime

import FinanceDataReader as fdr

print("[KRX] StockListing 로드 중...")
df = fdr.StockListing('KRX')
print(f"  {len(df)}개 종목 발견")

# 주요 지수 (FDR StockListing 에 안 들어옴) — 수동 추가
indices = [
    {"code": "KS11",  "name": "코스피 지수",  "market": "INDEX"},
    {"code": "KQ11",  "name": "코스닥 지수",  "market": "INDEX"},
    {"code": "KS200", "name": "코스피200",     "market": "INDEX"},
]
stocks = list(indices)

cols = {c.lower(): c for c in df.columns}
code_col = cols.get('code') or cols.get('symbol') or 'Code'
name_col = cols.get('name') or 'Name'
market_col = cols.get('market') or 'Market'

for _, r in df.iterrows():
    code = str(r.get(code_col, '')).strip()
    name = str(r.get(name_col, '')).strip()
    market = str(r.get(market_col, '')).strip()
    if not code or not name or code.lower() == 'nan':
        continue
    stocks.append({"code": code, "name": name, "market": market})

output = {
    "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "description": f"KRX 전체 자동완성 ({len(stocks)}개 종목) — 지수 + KOSPI/KOSDAQ/KONEX",
    "stocks": stocks,
}

# 절대 경로 — 어디서 실행해도 동일 위치
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.normpath(os.path.join(
    script_dir, '..', 'src', 'main', 'resources', 'static', 'stock_codes.json'
))
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False)

size_kb = os.path.getsize(out_path) / 1024
print(f"[저장 완료] {out_path}")
print(f"  종목 수: {len(stocks)}개  |  파일 크기: {size_kb:.1f} KB")

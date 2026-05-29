import os
import json
import re
from llama_cpp import Llama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate

# gguf 모델 기본 경로: AItr_manager/gemma-4-E4B-it-Q4_1.gguf (이 파일의 상위 폴더)
_DEFAULT_GGUF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gemma-4-E4B-it-Q4_1.gguf",
)


class _GemmaLLM:
    """llama.cpp 기반 경량 래퍼.

    gemma "it"(instruction) 모델은 대화 형식이 필요하므로, gguf에 내장된
    chat 템플릿을 자동 적용하는 create_chat_completion 을 사용한다.
    invoke(prompt:str) -> str 인터페이스로 기존 호출부와 호환된다.
    """

    def __init__(self, model_path, n_ctx=8192, max_tokens=1024, temperature=0.3):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=os.cpu_count(),
            n_gpu_layers=0,
            verbose=False,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature

    def invoke(self, prompt: str) -> str:
        res = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return res["choices"][0]["message"]["content"]


class StockAgent:
    def __init__(self, model_path=None):
        # 환경변수 GGUF_MODEL_PATH 우선, 없으면 상대경로 gguf 사용 (절대경로 하드코딩 X)
        model_path = model_path or os.environ.get("GGUF_MODEL_PATH", _DEFAULT_GGUF)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"gguf 모델을 찾을 수 없습니다: {model_path}")
        print(f"🧠 [Agent] gguf 모델 로드 중: {model_path}")
        self.llm = _GemmaLLM(model_path)
        self.search = DuckDuckGoSearchRun()

    def extract_params(self, user_request):
        """사용자 요청에서 6가지 지표 가중치 추출 (매핑 가이드 포함)"""
        prompt = PromptTemplate.from_template("""
        사용자 요청을 분석하여 다음 지표 가중치(-1.0 ~ 1.0)를 JSON으로 설정하세요.
        
        [지표별 키워드 매핑 가이드]
        - price_chg: 급등(+), 상승(+), 하락(-), 급락(-)
        - vol_ratio: 수급, 거래량 많은, 관심집중, 거래량 급증
        - rebound: 저점, 반등, 저평가, RSI 낮음, BB 하단
        - stability: 안정성, 우량주, 이격도 낮음, 꾸준한, 정배열
        - trend: 상승 추세, MACD, 골든크로스, 상승세 유지
        - volatility: 변동성, 단타, 위험, 기회
        
        결과는 오직 JSON만 출력하세요. 다른 설명은 하지 마세요.
        {{ "days": 정수, "weights": {{ "price_chg": 0.0, "vol_ratio": 0.0, "rebound": 0.0, "stability": 0.0, "trend": 0.0, "volatility": 0.0 }} }}
        요청: {user_request}
        """)
        try:
            res = self.llm.invoke(prompt.format(user_request=user_request))
            match = re.search(r'\{.*\}', res, re.DOTALL)
            return json.loads(match.group())
        except:
            return {"days": 7, "weights": {"stability": 0.5, "trend": 0.5}}

    def generate_report(self, user_request, top_5):
        """상위 5개 종목에 대해 최근 근황을 포함한 정밀 분석 수행"""
        print(f"🔍 [Agent] 상위 5개 종목 최근 근황 및 이슈 분석 중...")
        
        search_results = []
        for s in top_5:
            try:
                # '최근 근황' 키워드 추가로 검색 퀄리티 향상
                info = self.search.run(f"주식 {s['name']} {s['ticker']} 최근 뉴스 호재 악재 및 기업 근황 전망")
                search_results.append(f"[{s['name']} 정보]: {info[:400]}")
            except: continue
        
        news_context = "\n".join(search_results) if search_results else "검색된 최신 뉴스가 없습니다."
        detailed_data = "\n".join([
            f"- {s['name']}({s['ticker']}): 수익률 {s['price_chg']}% | RSI {s['rsi']} | 거래 {s['vol_ratio']}배 | 점수 {s['score']}"
            for s in top_5
        ])

        prompt = PromptTemplate.from_template("""
        금융 분석가로서 상위 5개 종목에 대해 심층 리포트를 작성하세요.
        
        [지침]
        1. 각 종목별 섹션을 나누어 '지표 분석'과 '최근 근황 및 뉴스({news_context})'를 결합할 것.
        2. {user_request}의 의도에 비추어 이 종목이 왜 적합한지 설명할 것.
        3. 단순 수치 나열이 아닌 데이터의 의미(예: "RSI가 낮아 과매도 반등 국면")를 해석할 것.

        [후보 종목 데이터]:
        {detailed_data}
        """)
        
        return self.llm.invoke(prompt.format(
            user_request=user_request, news_context=news_context, detailed_data=detailed_data
        ))

    def refine_report(self, raw_report):
        """광고성 문구 필터링 및 리포트 재검토 로직"""
        
        prompt = PromptTemplate.from_template("""
        당신은 전문 금융 검수관입니다. 다음 주식 리포트에서 광고성 내용이나 부적절한 정보를 제거하세요.

        [필터링 규칙]
        1. 전화번호, 카톡 아이디, 유튜브 채널 홍보, 유료 서비스 가이드 삭제.
        2. 문맥에 맞지 않은 자극적이고 비전문적인 광고 용어 삭제.
        3. 지표 데이터와 텍스트 설명이 논리적으로 맞는지 확인하고 자연스럽게 교정.
        4. 전문적인 금융 리포트 어조를 유지할 것.

        [검수 대상 리포트]:
        {raw_report}
        """)
        return self.llm.invoke(prompt.format(raw_report=raw_report))
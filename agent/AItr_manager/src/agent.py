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

    # 20260602: max_tokens 를 호출별로 override 가능하게.
    # 종합 리포트(5종목)는 1024 캡에 걸려 뒷 종목이 1줄/빈칸으로 잘리던 문제 해결.
    def invoke(self, prompt: str, max_tokens: int = None) -> str:
        res = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=self.temperature,
        )
        return res["choices"][0]["message"]["content"]


_DAYS_KEYWORD_MAP = [
    ("하루", 1), ("1일", 1), ("오늘", 1),
    ("3일", 3),
    ("일주일", 7), ("1주", 7), ("주간", 7), ("한주", 7), ("7일", 7),
    ("2주", 14), ("이주일", 14), ("14일", 14),
    ("한달", 30), ("1개월", 30), ("월간", 30), ("30일", 30),
    ("두달", 60), ("2개월", 60),
    ("분기", 90), ("3개월", 90),
]


def _coerce_days(raw_value, user_request: str) -> int:
    """LLM이 days를 빠뜨리거나 잘못된 값을 줘도 합리적인 정수로 보정."""
    try:
        v = int(raw_value)
        if 1 <= v <= 365:
            return v
    except (TypeError, ValueError):
        pass
    txt = (user_request or "")
    for kw, val in _DAYS_KEYWORD_MAP:
        if kw in txt:
            return val
    return 7


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
        """사용자 요청에서 6가지 지표 가중치 + 분석 기간(days) 추출"""
        prompt = PromptTemplate.from_template("""
        사용자 요청을 분석하여 다음 지표 가중치(-1.0 ~ 1.0)와 분석 기간(days)을 JSON으로 설정하세요.

        [기간(days) 추출 가이드 — 사용자가 언급한 기간을 일(day) 단위 정수로 변환]
        - "하루", "1일", "오늘" → 1
        - "3일", "단기" → 3
        - "일주일", "1주", "주간", "단기 트레이딩" → 7
        - "2주", "이주일" → 14
        - "한달", "1개월", "월간", "중기" → 30
        - "두달", "2개월" → 60
        - "분기", "3개월" → 90
        - 기간 언급 없음 → 7 (기본값)

        [지표별 키워드 매핑 가이드 (-1.0 ~ 1.0)]
        - price_chg: 급등(+), 상승(+), 하락(-), 급락(-)
        - vol_ratio: 수급, 거래량 많은, 관심집중, 거래량 급증
        - rebound: 저점, 반등, 저평가, RSI 낮음, BB 하단
        - stability: 안정성, 우량주, 이격도 낮음, 꾸준한, 정배열, 대형주
        - trend: 상승 추세, MACD, 골든크로스, 상승세 유지
        - volatility: 변동성, 단타, 위험, 기회

        결과는 오직 JSON만 출력하세요. 다른 설명은 하지 마세요.
        {{ "days": 정수, "weights": {{ "price_chg": 0.0, "vol_ratio": 0.0, "rebound": 0.0, "stability": 0.0, "trend": 0.0, "volatility": 0.0 }} }}
        요청: {user_request}
        """)
        try:
            res = self.llm.invoke(prompt.format(user_request=user_request))
            match = re.search(r'\{.*\}', res, re.DOTALL)
            params = json.loads(match.group())
        except:
            params = {"days": 7, "weights": {"stability": 0.5, "trend": 0.5}}

        # 안전망: LLM 출력이 부실해도 user_request 텍스트로 기간 폴백 추정
        params["days"] = _coerce_days(params.get("days"), user_request)
        return params

    def generate_report(self, user_request, top_5):
        """상위 5개 종목에 대해 최근 근황을 포함한 정밀 분석 수행"""
        print(f"🔍 [Agent] 상위 5개 종목 최근 근황 및 이슈 분석 중...")

        search_results = []
        for s in top_5:
            try:
                # '최근 근황' 키워드 추가로 검색 퀄리티 향상
                info = self.search.run(f"주식 {s['name']} {s['ticker']} 최근 뉴스 호재 악재 및 기업 근황 전망")
                search_results.append(f"[{s['name']} 정보]: {info[:400]}")
            except Exception:
                continue

        news_context = "\n".join(search_results) if search_results else "검색된 최신 뉴스가 없습니다."
        detailed_data = "\n".join([
            f"- {s['name']}({s['ticker']}): 수익률 {s['price_chg']}% | RSI {s['rsi']} | 거래 {s['vol_ratio']}배 | 점수 {s['score']}"
            for s in top_5
        ])

        # 20260602: 사용자가 보여준 농심 예시 그대로 — 3섹션 양식 강제.
        #   ① 한 줄 요약 헤더 ② [지표 분석] 항목별 해석 ③ [최근 근황 및 분석] ④ [투자 판단 근거]
        #   추가 잔존 섹션("최종 TOP3", "종합 추천" 등)은 절대 출력 금지 명시.
        prompt = PromptTemplate.from_template("""
        당신은 전문 금융 분석가입니다. 아래 5개 종목 각각에 대해, **반드시 동일한 양식**으로 풍부한 심층 분석을 작성하세요.

        [출력 양식 — 종목당 반드시 이 형태]
        ## 🥇 N순위: 종목명(종목코드) - 한 줄 요약(15자 이내 강점 요약)

        **[지표 분석]**
        *   **수익률 (X%)**: 수익률 수치를 해석하세요. (예: "최근 시장 대비 우수한 흐름", "약세장 속에서도 견조한 흐름" 등)
        *   **RSI (X)**: RSI 수치의 의미를 해석하세요. (50 미만=조정, 50~70=상승 안정, 70 이상=과열, 30 이하=과매도 반등 가능)
        *   **거래량 (X배)**: 평균 대비 거래량 비율이 시사하는 바를 해석하세요.

        **[최근 근황 및 분석]**
        제공된 뉴스 컨텍스트를 활용해 종목 별 구체적 호재/악재/구조적 동향을 3~5문장으로 풀어쓰세요.
        (예: "2026년 1분기 매출 전망", "해외법인 호조", "신사업 진출", "원가 부담 완화" 등 구체적 사실 인용)

        **[투자 판단 근거]**
        사용자 요청({user_request})의 의도에 비추어, 왜 이 종목이 그 의도에 적합한지 명시하세요.
        지표 + 근황을 종합해 "어떤 투자자에게 적합한지"까지 2~3문장으로 구체적으로 작성하세요.

        [필수 규칙 — 위반 금지]
        1. 5개 종목 **전부** 위 양식 그대로 작성. 단 한 종목도 누락 X.
        2. 1순위 = 🥇, 2순위 = 🥈, 3순위 = 🥉, 4·5순위는 이모지 없이 "## 4순위:" / "## 5순위:" 형태.
        3. **5번째 종목 분석 직후 출력을 종료**. "최종 TOP3", "종합 추천", "총평", "결론" 등 추가 섹션 **절대 작성 금지**.
        4. 광고/홍보/유료 서비스/카톡/유튜브 채널 언급 금지.
        5. 단순 수치 나열 금지 — 반드시 의미를 해석.

        [뉴스 컨텍스트]:
        {news_context}

        [후보 종목 데이터]:
        {detailed_data}
        """)

        # 5종목 × (3섹션 풍부 작성) = ~3000 토큰. 기본 1024는 후반 종목 잘림.
        return self.llm.invoke(prompt.format(
            user_request=user_request, news_context=news_context, detailed_data=detailed_data
        ), max_tokens=3500)

    # 20260602: refine_report 를 LLM 호출 → Python 정규식 필터로 교체.
    #   기존엔 동일 3500토큰 분량을 LLM 한 번 더 돌려서 약 3~5분 추가 소요.
    #   광고/홍보성 텍스트는 패턴이 명확해서 regex 로 99% 잡힘. 4분대 단축.
    @staticmethod
    def _filter_ads(report: str) -> str:
        """광고성 / 유료서비스 / 연락처 패턴을 한 줄 단위로 제거."""
        if not report:
            return report
        ad_patterns = [
            re.compile(r"전화\s*[:：]?\s*0\d{1,2}[-\.\s]?\d{3,4}[-\.\s]?\d{4}"),
            re.compile(r"0\d{1,2}[-\.\s]?\d{3,4}[-\.\s]?\d{4}"),
            re.compile(r"(카톡|카카오톡|카카오\s*ID|오픈\s*채팅)"),
            re.compile(r"(유튜브|유튜버|구독|채널)\s*([:：]|바로가기|에서|에서는)"),
            re.compile(r"(유료\s*가이드|유료\s*멤버|VIP\s*회원|프리미엄\s*서비스|텔레그램\s*가입)"),
            re.compile(r"(자세한\s*문의|상담\s*문의|투자\s*상담|무료\s*상담)"),
            re.compile(r"http[s]?://\S+"),
        ]
        kept = []
        for ln in report.splitlines():
            if any(p.search(ln) for p in ad_patterns):
                continue
            kept.append(ln)
        return "\n".join(kept)

    # 20260602: 5번째 종목 섹션 직후 잔존 섹션("최종 TOP3", "종합 추천", "결론" 등) 절단.
    @staticmethod
    def _trim_after_fifth(report: str) -> str:
        """## N순위 헤더를 5개까지만 유지하고, 5번째 이후 추가 ## 또는
        '최종 TOP', '종합 추천', '총평', '결론' 등 잔존 키워드 시작 위치에서 절단."""
        if not report:
            return report
        # `## ` 헤더 위치 모두 찾기
        header_re = re.compile(r"^##\s+", re.MULTILINE)
        headers = list(header_re.finditer(report))
        # 6번째 ## 헤더가 있으면 그 직전에서 컷
        if len(headers) >= 6:
            report = report[:headers[5].start()].rstrip() + "\n"
        # 잔존 섹션 키워드(자유 텍스트) — 라인 시작에 등장하면 그 라인부터 컷
        cut_kw_re = re.compile(
            r"^(\s*[\*\-]?\s*)?\*?\*?(최종\s*TOP|종합\s*추천|총\s*평|결론|마무리|최종\s*결론|최종\s*추천)",
            re.MULTILINE,
        )
        m = cut_kw_re.search(report)
        if m:
            report = report[:m.start()].rstrip() + "\n"
        return report

    def refine_report(self, raw_report):
        """광고/홍보 패턴 정규식 필터 + 5번째 종목 이후 잔존 섹션 절단.

        20260602: LLM 검수 호출(3500 tokens, 약 3~5분)을 Python 정규식 후처리로 교체.
                  LLM 정확도 손실은 있을 수 있으나 광고 제거는 패턴이 명확해 충분.
                  총 추천 소요시간 약 12분 → 약 7~8분.
        """
        report = self._filter_ads(raw_report)
        report = self._trim_after_fifth(report)
        return report
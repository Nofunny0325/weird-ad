"""
판단엔진

흐름:
1. 추출된 전체 텍스트에서 금지어(banned) 매칭 → 하나라도 있으면 즉시 "부당광고" 판정 (AI 호출 없이 종료)
2. 금지어가 없으면 오인유발어(ambiguous) 매칭
3. 오인유발어가 있으면, 그 단어가 포함된 문장만 뽑아서 AI API에 넘겨 문맥 분석 요청
4. AI 분석 결과를 받아 최종 판정

단어 매칭은 형태소 분석(kiwipiepy) 기반으로 한다.
문장과 사전 단어를 모두 형태소 단위로 쪼갠 뒤, 사전 단어의 형태소 시퀀스가
문장의 형태소 시퀀스 안에 "연속으로" 등장하는지 확인한다.
예: "업계 최고의 품질" → 형태소: [업계, 최고, 의, 품질] → "최고"가 독립된 형태소로 잡힘.
반대로 "완전 치밀한 설계" → 형태소: [완전, 치밀, 하, ㄴ, 설계] → "완치"라는 형태소가
존재하지 않으므로 매칭되지 않음 (단순 substring 매칭이었다면 "완(전)치(밀)"처럼
우연히 걸릴 일은 없지만, 더 복잡한 예시에서는 이런 식으로 오탐을 줄여준다).

kiwipiepy가 설치되어 있지 않으면 기존 substring 매칭으로 자동 폴백한다
(개발 중 임시 환경에서도 파이프라인이 멈추지 않도록).
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import os
import re

from data.wordlists_schema import WordLists, WordEntry
from pipeline.extractors import ExtractedText


class Verdict(str, Enum):
    UNFAIR = "부당광고"               # 금지어 발견 → 확정
    LIKELY_UNFAIR = "부당광고 추정"    # AI 분석 결과 오인 가능성 높음
    NOT_UNFAIR = "문제 없음"


@dataclass
class MatchedWord:
    entry: WordEntry
    matched_sentence: str   # 해당 단어가 등장한 문장(또는 주변 문맥)


@dataclass
class JudgeResult:
    verdict: Verdict
    banned_hits: list[MatchedWord] = field(default_factory=list)
    ambiguous_hits: list[MatchedWord] = field(default_factory=list)
    ai_analysis: str | None = None   # AI API 응답 원문/요약 (호출했을 경우만)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "banned_hits": [
                {"word": h.entry.word, "category": h.entry.category, "sentence": h.matched_sentence}
                for h in self.banned_hits
            ],
            "ambiguous_hits": [
                {"word": h.entry.word, "category": h.entry.category, "sentence": h.matched_sentence}
                for h in self.ambiguous_hits
            ],
            "ai_analysis": self.ai_analysis,
        }


def split_sentences(text: str) -> list[str]:
    """아주 단순한 문장 분리기. 추후 한국어 문장 분리 라이브러리로 교체 가능."""
    sentences = re.split(r"(?<=[.!?。])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


class MorphTokenizer:
    """
    kiwipiepy 기반 형태소 토크나이저.
    설치가 안 되어 있으면 None을 반환하도록 해서, 상위(KeywordMatcher)에서
    substring 매칭으로 폴백할지 판단하게 한다.
    """

    _kiwi = None
    _available: bool | None = None

    @classmethod
    def available(cls) -> bool:
        if cls._available is None:
            try:
                from kiwipiepy import Kiwi
                cls._kiwi = Kiwi()
                cls._available = True
            except ImportError:
                cls._available = False
        return cls._available

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """텍스트를 형태소 surface form 리스트로 변환. 예: '최고의' -> ['최고', '의']"""
        if not cls.available():
            raise RuntimeError("kiwipiepy가 설치되어 있지 않습니다.")
        tokens = cls._kiwi.tokenize(text)
        return [t.form for t in tokens]


class KeywordMatcher:
    def __init__(self, wordlists: WordLists):
        self.wordlists = wordlists
        self._use_morph = MorphTokenizer.available()
        if not self._use_morph:
            print(
                "[경고] kiwipiepy 미설치 — 형태소 매칭 대신 단순 substring 매칭으로 동작합니다. "
                "`pip install kiwipiepy` 로 설치하면 더 정확해집니다."
            )

    def find_matches(self, sentences: list[str], entries: list[WordEntry]) -> list[MatchedWord]:
        if self._use_morph:
            return self._find_matches_morph(sentences, entries)
        return self._find_matches_substring(sentences, entries)

    # --- 형태소 기반 매칭 ---
    def _find_matches_morph(self, sentences: list[str], entries: list[WordEntry]) -> list[MatchedWord]:
        hits = []
        # 사전 단어들도 미리 형태소로 쪼개둔다 (문장마다 반복 계산하지 않도록 캐싱)
        entry_tokens = [(entry, MorphTokenizer.tokenize(entry.word)) for entry in entries]

        for sentence in sentences:
            sentence_tokens = MorphTokenizer.tokenize(sentence)
            for entry, word_tokens in entry_tokens:
                if self._contains_subsequence(sentence_tokens, word_tokens):
                    hits.append(MatchedWord(entry=entry, matched_sentence=sentence))
        return hits

    @staticmethod
    def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
        """needle(사전 단어의 형태소 시퀀스)이 haystack(문장의 형태소 시퀀스) 안에
        연속으로 등장하는지 확인."""
        if not needle:
            return False
        n = len(needle)
        for i in range(len(haystack) - n + 1):
            if haystack[i:i + n] == needle:
                return True
        return False

    # --- 폴백: 단순 substring 매칭 ---
    def _find_matches_substring(self, sentences: list[str], entries: list[WordEntry]) -> list[MatchedWord]:
        hits = []
        for sentence in sentences:
            for entry in entries:
                if entry.word in sentence:
                    hits.append(MatchedWord(entry=entry, matched_sentence=sentence))
        return hits


AI_SYSTEM_PROMPT = """\
너는 표시·광고의 공정화에 관한 법률(표시광고법) 관점에서 광고 문구를 심사하는 보조 역할이다.
아래 문장들은 광고 영상에서 추출된 것으로, 각 문장에는 "오인 유발 가능 표현"으로 미리 분류된 단어가
포함되어 있다. 그 단어 자체는 금지어가 아니지만, 문맥에 따라 소비자가 상품의 효능·성능·品質을
사실과 다르게 오인할 수 있다.

각 문장에 대해 다음을 판단하라:
- 해당 단어가 과장되거나 근거 없는 주장과 결합되어 소비자를 오인시킬 가능성이 있는가?
- 단순 홍보성 수사(예: 주관적 감상)에 그쳐 문제가 없는가?

반드시 아래 JSON 형식으로만 답하라 (다른 설명 금지):
{
  "misleading": true/false,
  "reasoning": "전체 판단 근거 2~3문장",
  "per_sentence": [
    {"sentence": "...", "misleading": true/false, "reason": "..."}
  ]
}
"""


class AIContextAnalyzer:
    """
    오인유발어가 포함된 문장을 Claude API에 넘겨 실제로 소비자를 오인시킬
    문맥인지 판단 요청하는 래퍼.

    환경변수 ANTHROPIC_API_KEY 필요.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-5"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic 패키지가 설치되어 있지 않습니다. `pip install anthropic` 로 설치하세요."
                ) from e
            if not self.api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY가 설정되어 있지 않습니다. "
                    "환경변수로 설정하거나 AIContextAnalyzer(api_key=...)로 전달하세요."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def analyze(self, matched_sentences: list[MatchedWord]) -> tuple[bool, str]:
        """
        반환: (오인 가능성 있음 여부, 분석 근거 텍스트)
        """
        client = self._get_client()

        # 중복 문장 제거
        seen = set()
        unique_sentences = []
        for m in matched_sentences:
            if m.matched_sentence not in seen:
                seen.add(m.matched_sentence)
                unique_sentences.append(m.matched_sentence)

        user_content = "다음 문장들을 심사해줘:\n" + "\n".join(
            f"- {s}" for s in unique_sentences
        )

        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=AI_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

        try:
            parsed = json.loads(raw_text)
            is_misleading = bool(parsed.get("misleading", False))
            reasoning = parsed.get("reasoning", raw_text)
        except (json.JSONDecodeError, AttributeError):
            # AI가 JSON 형식을 안 지켰을 경우를 대비한 fallback
            is_misleading = "true" in raw_text.lower() and "misleading" in raw_text.lower()
            reasoning = raw_text

        return is_misleading, reasoning


class JudgeEngine:
    def __init__(self, wordlists: WordLists, ai_analyzer: AIContextAnalyzer | None = None):
        self.matcher = KeywordMatcher(wordlists)
        self.ai_analyzer = ai_analyzer or AIContextAnalyzer()

    def judge(self, extracted: ExtractedText) -> JudgeResult:
        full_text = extracted.full_text()
        sentences = split_sentences(full_text)

        # 1. 금지어 검사 — 발견 즉시 확정 판정, AI 호출 없이 종료
        banned_hits = self.matcher.find_matches(sentences, self.matcher.wordlists.banned)
        if banned_hits:
            return JudgeResult(verdict=Verdict.UNFAIR, banned_hits=banned_hits)

        # 2. 오인유발어 검사
        ambiguous_hits = self.matcher.find_matches(sentences, self.matcher.wordlists.ambiguous)
        if not ambiguous_hits:
            return JudgeResult(verdict=Verdict.NOT_UNFAIR)

        # 3. 오인유발어가 있으면 AI 문맥 분석 호출
        is_misleading, analysis_text = self.ai_analyzer.analyze(ambiguous_hits)

        verdict = Verdict.LIKELY_UNFAIR if is_misleading else Verdict.NOT_UNFAIR
        return JudgeResult(
            verdict=verdict,
            ambiguous_hits=ambiguous_hits,
            ai_analysis=analysis_text,
        )

"""Timed, local screening. Related phrasing is evidence to review, not a verdict."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from data.wordlists_schema import WordEntry, WordLists
from .extractors import ExtractedText
from .local_policy import NON_CLAIM, explicit_rule
from .semantic import EvidenceRetriever, compact, normalize

ROOT = Path(__file__).resolve().parents[1]


class Verdict(str, Enum):
    LIKELY_UNFAIR = '위반 의심'
    NEEDS_REVIEW = '확인 필요'


@dataclass
class MatchedWord:
    entry: WordEntry
    matched_sentence: str
    start_sec: float
    end_sec: float
    source: str


@dataclass
class Finding:
    kind: str
    rule_id: str
    title: str
    sentence: str
    start_sec: float
    end_sec: float
    source: str
    signal: str
    check: str
    comparison: str
    legal_basis: str
    source_title: str
    source_url: str
    references: list = field(default_factory=list)
    context: str = ''
    model_explanation: str = ''


@dataclass
class JudgeResult:
    verdict: Verdict = Verdict.NEEDS_REVIEW
    reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    extraction_status: dict[str, str] = field(default_factory=dict)
    banned_hits: list[MatchedWord] = field(default_factory=list)
    ambiguous_hits: list[MatchedWord] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    model_mode: str = '문자 유사도 + 규칙'
    product_status: str = '미확인'
    product: dict | None = None
    scanned_windows: int = 0
    review_scope: str = '추출된 자막·음성·입력 문장의 선별 검토'
    context_status: str = 'not_requested'
    context_model: str = ''
    context_scanned: int = 0
    context_total: int = 0
    extraction_providers: dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {**asdict(self), 'verdict': self.verdict.value}


def _split(text):
    # Split at punctuation or a contrast that may reverse a disclaimer; do not split decimals.
    return [piece.strip() for piece in re.split(
        r'(?<=[.!?。])\s+|[!?。](?!\s)|[\r\n]+|\s+(?=(?:하지만|그러나|반면)\s)',
        text) if piece.strip()]


def _windows(extracted):
    segments = [s for s in extracted.all_segments() if s.text.strip()]
    channels = {kind: [s for s in segments if s.source == kind] for kind in {s.source for s in segments}}
    channels = dict(sorted(channels.items()))
    for kind, channel in channels.items():
        for i, current in enumerate(channel):
            near = channel[max(0,i-1):i+2]
            # Attach only overlapping text from other channels; retain actual claim source.
            other = [s for k,group in channels.items() if k != kind for s in group
                     if s.start_sec < current.end_sec and s.end_sec > current.start_sec]
            context = '\n'.join(f'[{s.source} {s.start_sec:.1f}–{s.end_sec:.1f}] {s.text}'
                                for s in sorted(near+other,key=lambda s:s.start_sec))
            parts = _split(current.text)
            for j,part in enumerate(parts):
                local_context = '\n'.join(parts[max(0,j-1):j+2]) if kind=='text' else context
                yield (part,current.start_sec,current.end_sec,kind,local_context,())
            if kind=='ocr':
                for spoken in other:
                    if spoken.source!='stt':
                        continue
                    for left in parts:
                        for right in _split(spoken.text):
                            pair=(left,right)
                            joined=' '.join(pair)
                            if (compact(left)!=compact(right) and not any(explicit_rule(p) for p in pair)
                                    and explicit_rule(joined)):
                                yield (joined,min(current.start_sec,spoken.start_sec),
                                       max(current.end_sec,spoken.end_sec),'ocr+stt',context,pair)
            if i and parts:
                prev = channel[i-1]
                prev_parts = _split(prev.text)
                if (prev_parts and 0 <= current.start_sec-prev.end_sec <= 2 and
                        not re.search(r'[.!?。]$',prev_parts[-1])):
                    pieces = (prev_parts[-1],parts[0])
                    yield (' '.join(pieces),prev.start_sec,current.end_sec,kind,context,pieces)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _catalog(path):
    records = _read(path)['products']
    result = {}
    for item in records:
        if (not isinstance(item['id'], str) or not item['id'].strip() or item['id'] in result or
            not all(isinstance(item.get(k),str) and item[k].strip() for k in ['name','classification','source_url','as_of']) or
            not isinstance(item['functionalities'],list) or not item['functionalities'] or
            not all(isinstance(x,str) and x.strip() for x in item['functionalities'])):
            raise ValueError('제품 목록에 필요한 값이 없거나 식별번호가 중복됩니다.')
        url = urlsplit(item['source_url'])
        if url.scheme != 'https' or not url.hostname or url.username or url.password:
            raise ValueError('제품 근거 URL은 올바른 HTTPS 주소여야 합니다.')
        result[item['id']] = item
    return result


class JudgeEngine:
    def __init__(self, wordlists, rules=(), references=(), products=None, model_dir=None):
        self.words = wordlists
        self.rules = {r['id']:r for r in rules}
        self.references = {r['id']:r for r in references}
        self.products = products or {}
        self.keys = [(bucket,row,compact(row.word)) for bucket,items in
                     [('banned_hits',wordlists.banned),('ambiguous_hits',wordlists.ambiguous)]
                     for row in items if compact(row.word)]
        self.retriever = EvidenceRetriever(list(self.rules.values()),model_dir)

    @classmethod
    def from_local_data(cls, model_dir=None, catalog_path=None):
        data = _read(ROOT/'data'/'food_ad_review.json')
        entries = lambda group: [WordEntry(x['word'],x['category'],x['legal_basis']) for x in data[group]]
        return cls(WordLists(entries('banned'),entries('ambiguous')),data['rules'],data['references'],
                   _catalog(catalog_path or ROOT/'data'/'product_catalog.json'), model_dir)

    def _evidence(self, kind, rule_id, window, product=None):
        text,start,end,source,context,_=window
        if product:
            return Finding(kind,rule_id,'제품 인정 문구와 일치',text,start,end,source,
                '등록 자료에 있는 기능성 문구와 일치합니다.',
                '광고의 나머지 주장, 섭취 조건과 심의 자료는 별도 확인하세요.',
                '', '', product['name'],product['source_url'],context=context)
        rule=self.rules.get(rule_id,{})
        return Finding(kind,rule_id,rule.get('title','질병 효능·약 대체 표현'),text,start,end,source,
            rule.get('signal','광고에서 질병 효능 또는 약 대체를 주장하는 표현이 감지됐습니다.'),
            rule.get('check','해당 제품의 인정 기능성과 문맥을 확인하세요.'),
            rule.get('contrast',''),rule.get('legal_basis',''),rule.get('source_title',''),rule.get('source_url',''),
            [self.references[r] for r in rule.get('reference_ids',[])],context)

    def judge(self, extracted: ExtractedText, product_id: str = '', *, groq_client=None) -> JudgeResult:
        product=self.products.get(product_id.strip())
        result=JudgeResult(extraction_status=dict(extracted.channel_status), product=product,
            extraction_providers=dict(extracted.providers),
            product_status='등록 제품 대조' if product else '미확인', limitations=list(extracted.warnings)+[
                '판매페이지·전후 사진 등 비텍스트 연출, 성분 실측, 인체시험 및 자율심의는 별도 확인이 필요합니다.',
                '비교 문구 및 유사 사례는 검토 근거입니다. 제품 전체의 적법성을 인증하지 않습니다.'])
        if not product:
            result.limitations.append('제품의 인정 기능성과 성분·섭취 조건을 대조하지 못했습니다.')
        else:
            result.limitations.append('입력된 번호의 기록과 대조했습니다. 영상에 나온 제품과 같은지 확인이 필요합니다.')
        windows = list(_windows(extracted))
        result.scanned_windows = len(windows)
        candidates = []
        definite = set()
        evidence_seen = set()
        approved = {compact(x).rstrip('.!?。') for x in product['functionalities']} if product else set()
        def record(kind, rule, window, matched_product=None):
            text,start,end,source,_,_=window
            dedupe=(kind,rule,source,start,end,text)
            if dedupe not in evidence_seen:
                evidence_seen.add(dedupe)
                result.findings.append(self._evidence(kind,rule,window,matched_product))
        for window in windows:
            text,start,end,source,_,parts=window
            if source=='ocr+stt':
                record('combined',explicit_rule(text),window)
                continue
            folded=compact(text)
            for bucket,entry,key in self.keys:
                if key in folded and not any(key in compact(p) for p in parts):
                    getattr(result,bucket).append(MatchedWord(entry,text,start,end,source))
            hard=explicit_rule(text)
            if hard and not any(explicit_rule(p)==hard for p in parts):
                definite.add(hard)
                record('explicit',hard,window)
            elif not parts and product and product['classification']=='건강기능식품' and folded.rstrip('.!?。') in approved:
                record('recognized','PRODUCT',window,product)
            elif not hard and not NON_CLAIM.search(normalize(text)):
                candidates.append(window)
        related,used_e5 = self.retriever.retrieve([w[0] for w in candidates])
        if used_e5 or (self.retriever.dense is not None and not candidates):
            result.model_mode='로컬 E5 + 문자 유사도 + 규칙'
        elif candidates:
            result.limitations.append('E5 모델을 사용하지 못해 문자 유사도와 규칙으로 검토했습니다.')
        for window, rules in zip(candidates,related):
            for rule in rules:
                record('similar',rule['id'],window)
        order={'explicit':0,'combined':1,'similar':2,'recognized':3}
        result.findings.sort(key=lambda f:(order[f.kind],f.start_sec,f.end_sec,f.rule_id))
        if len(result.findings)>80:
            result.limitations.append(f'후보 {len(result.findings)}개 중 우선순위가 높은 80개를 표시했습니다.')
            result.findings=result.findings[:80]
        if not windows:
            result.reasons.append('분석할 텍스트가 없습니다. 음성·자막 추출 상태를 확인하세요.')
        elif definite:
            result.verdict = Verdict.LIKELY_UNFAIR
            result.reasons.append('제품 효능·약 대체 등에 관한 명시적 의심 표현이 발견됐습니다. 근거 장면을 확인하세요.')
        elif result.findings or result.banned_hits or result.ambiguous_hits:
            result.reasons.append('관련 표현 또는 비교 기준을 찾았습니다. 유사도나 단어 일치만으로 위반을 판정하지 않습니다.')
        else:
            result.reasons.append('추출 텍스트에서 주요 후보를 찾지 못했습니다. 사전에 없는 주장이나 미추출 구간은 별도 확인이 필요합니다.')
        if groq_client is not None:
            from .context_review import enhance_review
            return enhance_review(self, extracted, result, groq_client)
        return result


@lru_cache(maxsize=1)
def default_engine():
    model=os.environ.get('FOOD_AD_E5_DIR',str(ROOT/'storage'/'models'/'e5'))
    catalog=os.environ.get('FOOD_AD_PRODUCT_CATALOG')
    return JudgeEngine.from_local_data(Path(model),catalog)

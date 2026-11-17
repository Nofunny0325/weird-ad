"""Local retrieval: multilingual E5 embeddings and Korean character TF-IDF."""

from collections import Counter
import math
from pathlib import Path
import re
import unicodedata

# Additional examples cover spoken expressions with no dictionary match.
# These are retrieval examples, not regulatory findings or an evaluation set.
_RISK_VARIANTS = {
    'CTX-01': ['병원에서 받아 온 혈당약을 이제 챙기지 않아도 됩니다.'],
    'CTX-03': ['한 알 복용하면 잠 못 자던 증상이 없어져 수면제가 필요하지 않아요.'],
    'CTX-05': ['산만하던 아이를 진정시키고 주의력결핍을 해결해 줍니다.'],
    'CTX-06': ['아이의 신장이 보름이면 몇 센티미터나 늘어납니다.'],
    'CTX-07': ['눈이 건조해서 생긴 질환을 영양제 하나로 치료하세요.'],
    'CTX-09': ['혈관 벽에 들러붙은 기름 찌꺼기를 걷어 내고 막힌 곳을 뚫습니다.'],
    'CTX-10': ['먹기만 하면 자는 동안 뱃살이 저절로 녹고 몸무게가 줄어요.',
               '다음 달에는 체중계로 확실한 체중 감량을 확인할 수 있어요.'],
    'CTX-11': ['맞는 비만 주사와 작용이 똑같아서 배고픔을 억제합니다.'],
    'CTX-12': ['먹고 나면 독감이나 감기에 걸릴 일이 없어요.'],
    'CTX-13': ['영양제를 먹고 귓속에서 들리던 소음이 멈췄습니다.'],
    'CTX-17': ['먹기만 하면 아토피로 가렵던 피부가 깨끗해집니다.'],
    'CTX-18': ['섭취하면 간에 쌓인 독을 씻어 내고 노폐물을 모두 없앱니다.'],
    'CTX-21': ['흰 가운을 입은 제가 이 상품을 의사 자격으로 추천합니다.'],
    'CTX-25': ['누가 얼마나 많이 먹어도 부작용이나 문제가 생기지 않습니다.'],
}
_NEUTRAL = [
    '고소한 맛과 산뜻한 향이 납니다.', '포장지에 적힌 섭취량을 확인하세요.',
    '점심 식사와 함께 물로 섭취하세요.', '개봉 후 냉장 보관하세요.',
    '이 제품은 질병을 예방하거나 치료하는 의약품이 아닙니다.',
    '임산부나 어린이는 이 제품을 섭취하지 마세요.',
    '제품에 표시된 사용 방법과 섭취 시 주의사항을 확인하세요.',
    '행사 사은품은 수량이 한정됩니다.', '상품명과 원재료명은 제품에 표시되어 있습니다.',
]
_TOPIC = {
    'CTX-01': r'혈당|인슐린|당뇨|혈당약',
    'CTX-03': r'수면|잠을|잠이|잠들|불면',
    'CTX-04': r'기억|집중|수능|시험공부',
    'CTX-05': r'adhd|주의력|산만',
    'CTX-06': r'키가|키가커|키가자|키성장|신장|어린이키|애도[^.!?]{0,13}컸',
    'CTX-07': r'안구|건조한눈|눈물|뻑뻑|빡빡하던눈',
    'CTX-09': r'혈관|혈전|플라크|피가흐',
    'CTX-10': r'뱃살|체중|몸무게|내장지방|살이|몇킬로|체지방',
    'CTX-11': r'비만주사|식욕|위고비|마운자로|주사.{0,18}배고',
    'CTX-12': r'독감|감기|면역|바이러스',
    'CTX-13': r'귀에서|귓속|귀울림|이명|청신경',
    'CTX-14': r'관절|연골|무릎',
    'CTX-17': r'아토피|두통|염증|피부질환',
    'CTX-18': r'독소|해독|노폐물',
    'CTX-20': r'노화|세포나이|신체나이|회춘',
    'CTX-25': r'부작용|탈날일|누구나|임산부|어린이',
}


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'[‐‑‒–—−]', '-', text)
    return re.sub(r'glp-?1(?![0-9])', 'glp1', text)


def compact(text: str) -> str:
    return re.sub(r'[\s#\u200b-\u200d\u2060\ufeff]+', '', normalize(text))


def char_ngrams(text: str) -> Counter:
    text = ''.join(ch for ch in compact(text) if ch.isalnum())
    return Counter(text[i:i+n] for n in (2, 3, 4) for i in range(len(text)-n+1))


class CharTfidf:
    def __init__(self, documents):
        frequency = Counter()
        for text in documents:
            frequency.update(char_ngrams(text).keys())
        self.idf = {g: math.log((1+len(documents))/(1+n))+1 for g, n in frequency.items()}
        self.unknown_idf = math.log(1+len(documents))+1
        self.vectors = [self.vector(text) for text in documents]

    def vector(self, text):
        weights = {g: (1+math.log(count))*self.idf.get(g, self.unknown_idf)
                   for g, count in char_ngrams(text).items()}
        length = math.sqrt(sum(x*x for x in weights.values())) or 1
        return {g: x/length for g, x in weights.items()}

    def scores(self, texts):
        for text in texts:
            query = self.vector(text)
            yield [sum(value*query.get(g, 0) for g, value in v.items()) for v in self.vectors]


class LocalE5:
    """Use only local files; neither the tokenizer nor ORT can fetch a model here."""

    def __init__(self, directory: Path):
        model, tokens = directory / 'model.onnx', directory / 'tokenizer.json'
        if not model.is_file() or not tokens.is_file():
            raise FileNotFoundError('Local E5 assets missing')
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(tokens))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id('<pad>'), pad_token='<pad>')
        self.np = np

    def encode(self, texts):
        np = self.np
        result = []
        for i in range(0, len(texts), 16):
            batch = self.tokenizer.encode_batch(['query: '+t for t in texts[i:i+16]])
            mask = np.array([x.attention_mask for x in batch], dtype=np.int64)
            values = {
                'input_ids': np.array([x.ids for x in batch], dtype=np.int64),
                'attention_mask': mask,
                'token_type_ids': np.array([x.type_ids for x in batch], dtype=np.int64),
            }
            hidden = self.session.run(None, {k:v for k,v in values.items() if k in self.inputs})[0]
            if hidden.ndim == 3:
                masked = hidden * mask[:, :, None]
                hidden = masked.sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
            result.append(hidden / np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), 1e-12))
        return np.concatenate(result)


class EvidenceRetriever:
    """Rank risky prototypes against a comparison example for the same rule.

    Thresholds are provisional retrieval cutoffs, not measured probabilities.
    This component cannot promote a result to '위반 의심'.
    """

    def __init__(self, rules, model_dir):
        self.rules = rules
        self.corpus = []
        self.index = []
        for rule in rules:
            start=len(self.corpus)
            self.corpus.extend([rule['risk'], *_RISK_VARIANTS.get(rule['id'],[])])
            end=len(self.corpus)
            self.corpus.append(rule['contrast'])
            self.index.append((start,end))
        self.neutral_start=len(self.corpus)
        self.corpus.extend(_NEUTRAL)
        self.sparse = CharTfidf(self.corpus)
        self.dense = None
        if self.rules and model_dir is not None:
            try:
                encoder = LocalE5(model_dir)
                vectors=encoder.encode(self.corpus)
                self.mean=vectors.mean(axis=0,keepdims=True)
                self.dense_vectors=self.center(vectors,encoder.np)
                self.dense = encoder
            except Exception:
                # Missing/corrupt model or unsupported runtime: preserve the basic path.
                self.dense = None

    def center(self,vectors,np):
        centered=vectors-self.mean
        return centered/np.maximum(np.linalg.norm(centered,axis=1,keepdims=True),1e-12)

    def retrieve(self, texts):
        if not self.rules:
            return ([[] for _ in texts], False)
        matrix = None
        if texts and self.dense is not None:
            try:
                matrix = self.center(self.dense.encode(texts),self.dense.np) @ self.dense_vectors.T
            except Exception:
                matrix = None
        results = []
        for i, sparse in enumerate(self.sparse.scores(texts)):
            ranked = []
            neutral_score=max(sparse[self.neutral_start:])
            dense_neutral=float(matrix[i,self.neutral_start:].max()) if matrix is not None else 0
            topics=compact(texts[i])
            dense_scores=[float(matrix[i,a:b].max()) for a,b in self.index] if matrix is not None else []
            for j,((start,end),rule) in enumerate(zip(self.index,self.rules)):
                score=max(sparse[start:end])
                comparison=max(sparse[end],neutral_score)
                lex=score>=.34 and score-comparison>=.10
                semantic=dense_scores[j] if matrix is not None else 0
                contrast=max(float(matrix[i,end]),dense_neutral) if matrix is not None else 0
                topic=bool(rule['id'] in _TOPIC and re.search(_TOPIC[rule['id']],topics))
                other=max((v for k,v in enumerate(dense_scores) if k!=j),default=0)
                dense=(semantic>=.38 and semantic-contrast>=.12 and score>=.095) or (
                    topic and semantic>=.25 and semantic-contrast>=.06
                ) or (semantic>=.46 and semantic-contrast>=.18 and semantic-other>=.09)
                if lex or dense:
                    ranking=.64*semantic+.36*score+.09*topic if matrix is not None else score
                    ranked.append((ranking, rule))
            ranked.sort(key=lambda item: item[0], reverse=True)
            results.append([rule for _,rule in ranked[:2]])
        return results, matrix is not None

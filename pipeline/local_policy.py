"""Narrow rules for explicit claims; broad similarity is handled separately."""
import re
from .semantic import normalize

DISEASE = r'(?:당뇨(?:병)?|고혈압|암(?!호|기|석|벽|흑)|관절염|골다공증|불면증|우울증|아토피|치매|감기|독감|이명|난청|adhd|안구건조증|변비|생리통|탈모|염증|비염|난임)'
EFFECT = r'(?:완치|치료|예방|방지|고치|낫[고는습]|나았|없애|없어[져집졌]|사라[져집졌]|잡아|제거|개선|회복|걸리지\s*않|안\s*걸|걸릴\s*(?:일|걱정)(?:이|도)?\s*없)'
MEDICINE = r'(?:처방(?:약|받은\s*약)?|병원에서[^.!?]{0,28}약|당뇨약|혈당약|수면제|위고비|마운자로|혈압약|관절약|치료제|(?<![가-힣])약)'
PRODUCT = r'(?:이\s*(?:식품|제품|거|걸|것|차)|먹고|마시고|먹으면|섭취|드시면|드신|드세요|한\s*포)'

# Suppress only specific caution/negation/report wording in its own clause.
# A blanket search for '안' would suppress the affirmative claim '안 걸려요'.
NON_CLAIM = re.compile(
    r'(?:섭취|복용|병용)[^.!?]{0,25}(?:전문가와\s*상담|의사와\s*상담|주의하|주의가|피하)'
    r'|(?:완치|치료|예방|개선|대체|대신)[^.!?]{0,28}(?:하지\s*않|하는\s*(?:것은|건)\s*아니|효과가?\s*없|할\s*수\s*없|되지\s*않|사실이\s*아니|아닙니다)'
    r'|(?:당뇨|불면증|고혈압|암)\s*치료(?:를|도)?\s*(?:받|중이|중인|중\s*이)'
    r'|(?:임산부|수유부|어린이)[^.!?]{0,30}(?:피해\s*주|피하|섭취를\s*삼가)'
    r'|(?:의약품|치료제)\s*(?:이|가)?\s*아닙'
    r'|(?:광고|표현)[^.!?]{0,12}(?:적발|제재)(?:됐|되었|했다|했습니다|됐습니다)'
    r'|(?:효능|제품)과?\s*직접(?:적인)?\s*관련이?\s*없'
)
MED_REPLACE = re.compile(
    MEDICINE + r'[^.!?]{0,34}(?:대신|끊(?:어|으)|(?:챙길|먹을|복용할)\s*필요(?:가|도)?\s*(?:없|없더)|안\s*(?:먹|챙겨)어도|안\s*(?:먹|챙겨)|안\s*먹어도)'
    r'|(?:먹는|천연)\s*(?:위고비|마운자로)'
)
DISEASE_CLAIM = re.compile(DISEASE + r'[^.!?]{0,25}' + EFFECT)
STRONG_CLAIM = re.compile(r'완치|치료(?:합니다|해|됩)|(?:예방|제거|개선)(?:합니다|해\s*줍)|사라(?:집|졌)|안\s*걸')
WEIGHT = re.compile(r'(?:\d+(?:\.\d+)?\s*(?:kg|키로|킬로그램))[^.!?]{0,14}(?:감량|빠(?:집|져|졌)|뺄\s*수)|(?:감량|빠진)\s*\d+(?:\.\d+)?\s*(?:kg|키로)')
PERIOD_OR_PROMISE = re.compile(r'\d+\s*(?:일|주|달|시간)|딱|누구나|무조건|보장|확정')
EXPERT = re.compile(r'(?:의사|약사|한의사|교수)[^.!?]{0,22}(?:이\s*제품|제가|직접)[^.!?]{0,20}(?:보장|추천|보증)|(?:의사|약사)\s*추천\s*(?:제품|건기식)')
SAFETY = re.compile(r'(?:누구나|무조건|임산부도|어린이도)[^.!?]{0,28}(?:부작용\s*없|안전하게)')
VASCULAR = re.compile(r'(?:막힌\s*혈관[^.!?]{0,10}뚫|혈전[^.!?]{0,12}(?:녹여|제거))')


def explicit_rule(text):
    text = normalize(text)
    if NON_CLAIM.search(text):
        return None
    if MED_REPLACE.search(text):
        return 'CTX-11' if re.search(r'위고비|마운자로',text) else 'CTX-01'
    if (DISEASE_CLAIM.search(text) and (STRONG_CLAIM.search(text) or re.search(PRODUCT,text))) or (
            '완치' in text and re.search(PRODUCT,text)):
        for regex,rule in [(r'골다공증','CTX-02'),(r'불면증','CTX-03'),(r'당뇨','CTX-01'),
                           (r'안구건조증','CTX-07'),(r'변비','CTX-08'),(r'이명|난청','CTX-13'),
                           (r'아토피|두통|염증','CTX-17'),(r'감기|독감','CTX-12'),(r'관절염','CTX-14')]:
            if re.search(regex,text):
                return rule
        return 'CTX-17'
    if VASCULAR.search(text):
        return 'CTX-09'
    if EXPERT.search(text):
        return 'CTX-21'
    if SAFETY.search(text):
        return 'CTX-25'
    if WEIGHT.search(text) and PERIOD_OR_PROMISE.search(text):
        return 'CTX-10'
    if re.search(r'키가?[^.!?]{0,10}\d+(?:\.\d+)?\s*(?:cm|센티)',text) and PERIOD_OR_PROMISE.search(text):
        return 'CTX-06'
    return None

import unittest
from pathlib import Path

from pipeline.extractors import ExtractedText, TimedText
from pipeline.judge_engine import JudgeEngine


class LocalScreeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = JudgeEngine.from_local_data(model_dir=Path('storage/nonexistent-model-for-tests'))

    def screen(self, text, **options):
        return self.engine.judge(ExtractedText('test', stt_segments=[TimedText(text, 2, 5, 'stt')]), **options)

    def test_implicit_medication_replacement_does_not_need_a_wordlist_hit(self):
        result = self.screen('이거 먹고 나면 병원에서 받아 온 약을 챙길 필요도 없더라고요.')
        self.assertEqual(result.verdict.value, '위반 의심')
        self.assertFalse(result.banned_hits or result.ambiguous_hits)
        item = next(e for e in result.findings if e.rule_id == 'CTX-33')
        self.assertEqual((item.start_sec, item.end_sec, item.source), (2, 5, 'stt'))

    def test_caution_and_negated_efficacy_are_not_a_positive_claim(self):
        for text in ['당뇨약 복용 중이면 전문가와 상담하세요.', '이 제품은 당뇨를 완치하지 않습니다.',
                     '이 식품은 질병 치료를 위한 의약품이 아닙니다.',
                     '당뇨약을 대신한다는 주장은 사실이 아닙니다.', '현재 당뇨 치료를 받고 있습니다.']:
            with self.subTest(text=text):
                self.assertEqual(self.screen(text).verdict.value, '확인 필요')

    def test_disclaimer_does_not_hide_affirmative_claim_in_next_clause(self):
        result = self.screen('의약품이 아닙니다. 하지만 이 제품이 당뇨를 완치합니다.')
        self.assertEqual(result.verdict.value, '위반 의심')

    def test_same_channel_fragment_and_non_lexical_scan(self):
        extracted = ExtractedText('test', stt_segments=[
            TimedText('이걸 먹고 나서 병원에서 받은 약을', 1, 2, 'stt'),
            TimedText('챙길 필요가 없었어요.', 2, 3, 'stt')])
        result = self.engine.judge(extracted)
        self.assertTrue(any((f.start_sec, f.end_sec) == (1, 3) for f in result.findings))

    def test_approved_function_is_product_specific(self):
        text = '식후 혈당 상승 억제에 도움을 줄 수 있음.'
        unknown = self.screen(text, product_id='missing-product')
        known = self.screen(text, product_id='2023021000236372')
        self.assertEqual(unknown.product_status, '미확인')
        self.assertEqual(known.product_status, '등록 제품 대조')
        self.assertTrue(any(f.kind == 'recognized' for f in known.findings))
        self.assertFalse(any(f.kind == 'recognized' for f in unknown.findings))
        self.assertEqual(known.verdict.value, '확인 필요')

    def test_model_unavailable_does_not_prevent_local_review(self):
        result = self.screen('딱 2주면 누구나 7kg가 빠집니다.')
        self.assertEqual(result.verdict.value, '위반 의심')
        self.assertTrue(result.model_mode)
        self.assertGreater(result.scanned_windows, 0)

    def test_treatment_mention_with_nonadjacent_claim_leaves_room_for_review(self):
        result = self.screen('특허 사업을 시작했습니다. 당뇨의 원인을 설명해 봅니다.')
        self.assertEqual(result.verdict.value, '확인 필요')

    def test_screen_and_voice_can_form_a_review_candidate_together(self):
        extracted=ExtractedText('test',
            ocr_segments=[TimedText('당뇨',1,3,'ocr')],
            stt_segments=[TimedText('이거 먹으면 없어집니다.',2,4,'stt')])
        result=self.engine.judge(extracted)
        self.assertEqual(result.verdict.value,'확인 필요')
        self.assertTrue(any((f.kind,f.source,f.start_sec,f.end_sec)==('combined','ocr+stt',1,4)
                            for f in result.findings))


if __name__ == '__main__':
    unittest.main()

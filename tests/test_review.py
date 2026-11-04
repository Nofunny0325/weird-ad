import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from data.wordlists_schema import WordEntry, WordLists
from pipeline.extractors import ExtractedText, TimedText, TextExtractionService, OCRExtractor
from pipeline.judge_engine import AIContextAnalyzer, JudgeEngine


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = patch('pipeline.judge_engine.MorphTokenizer.available', return_value=False)
        self.tokenizer.start()
        self.addCleanup(self.tokenizer.stop)
        self.words = WordLists([WordEntry('완치', '질병 치료', '검토 기준 R01')], [WordEntry('즉시 효과', '효과 과장')])

    def extracted(self, text):
        return ExtractedText('test', [TimedText(text, 2.5, 4.0, 'ocr')])

    def test_empty_text_requires_review(self):
        result = JudgeEngine(self.words).judge(self.extracted('   '))
        self.assertEqual(result.verdict.value, '확인 필요')
        self.assertTrue(result.reasons)

    def test_no_keyword_does_not_clear_unverified_product(self):
        result = JudgeEngine(self.words).judge(self.extracted('맛있는 차입니다'))
        self.assertEqual(result.verdict.value, '확인 필요')
        self.assertIn('성분', ' '.join(result.limitations))

    def test_keyword_without_ai_is_not_a_conviction(self):
        analyzer = SimpleNamespace(analyze=lambda hits: (_ for _ in ()).throw(ValueError('no key')))
        result = JudgeEngine(self.words, analyzer).judge(self.extracted('당뇨를 완치하는 것은 아닙니다'))
        self.assertEqual(result.verdict.value, '확인 필요')
        hit = result.to_dict()['banned_hits'][0]
        self.assertEqual((hit['start_sec'], hit['end_sec'], hit['source']), (2.5, 4.0, 'ocr'))
        self.assertEqual(hit['legal_basis'], '검토 기준 R01')

    def test_both_lists_are_analyzed_and_positive_is_only_suspected(self):
        received = []
        def analyze(hits):
            received.extend(hits)
            return True, '질병 효과와 즉시 효과를 제품 섭취에 연결함'
        result = JudgeEngine(self.words, SimpleNamespace(analyze=analyze)).judge(self.extracted('완치와 즉시 효과'))
        self.assertEqual(result.verdict.value, '위반 의심')
        self.assertEqual(len(received), 2)
        self.assertTrue(result.limitations)

    def test_negative_ai_does_not_clear_missing_product_checks(self):
        result = JudgeEngine(self.words, SimpleNamespace(analyze=lambda hits: (False, '치료 효과를 부정함'))).judge(self.extracted('완치가 아닙니다'))
        self.assertEqual(result.verdict.value, '확인 필요')

    def test_same_text_keeps_separate_channel_locations(self):
        extracted = self.extracted('완치')
        extracted.stt_segments.append(TimedText('완치', 8, 9, 'stt'))
        result = JudgeEngine(self.words, SimpleNamespace(analyze=lambda hits: (False, '추가 확인'))).judge(extracted)
        self.assertEqual([(h.source, h.start_sec) for h in result.banned_hits], [('ocr', 2.5), ('stt', 8)])

    def test_phrase_spanning_adjacent_speech_segments_retains_union_interval(self):
        words = WordLists([WordEntry('부작용 없음', '안전성')], [])
        extracted = ExtractedText('test', stt_segments=[TimedText('부작용', 1, 2, 'stt'), TimedText('없음', 2, 3, 'stt')])
        result = JudgeEngine(words, SimpleNamespace(analyze=lambda hits: (False, '추가 확인'))).judge(extracted)
        self.assertEqual(len(result.banned_hits), 1)
        self.assertEqual((result.banned_hits[0].start_sec, result.banned_hits[0].end_sec), (1, 3))

    def test_ocr_cannot_displace_next_spoken_negation(self):
        captured = []
        def analyze(hits):
            captured.extend(hits)
            return False, '치료 주장 부정'
        extracted = ExtractedText('test', [TimedText('행사', 0, 1, 'ocr'), TimedText('신제품', .5, 1.5, 'ocr'), TimedText('할인', 1.5, 2.5, 'ocr')], [TimedText('이 제품은 완치', 0, 1, 'stt'), TimedText('효과가 없습니다', 1, 2, 'stt')])
        JudgeEngine(self.words, SimpleNamespace(analyze=analyze)).judge(extracted)
        self.assertIn('효과가 없습니다', captured[0].context)

    def ai_response(self, content):
        analyzer = AIContextAnalyzer(api_key='test-only')
        analyzer._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(content=[SimpleNamespace(type='text', text=content)])))
        from pipeline.judge_engine import MatchedWord
        return analyzer.analyze([MatchedWord(self.words.banned[0], '완치')])

    def test_malformed_ai_response_is_not_guessed(self):
        for content in ['misleading true', '[]', '{"misleading":"false","reasoning":"x"}', '{"misleading":false}', '{"misleading":true,"reasoning":""}']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.ai_response(content)

    def test_valid_ai_boolean_is_preserved(self):
        self.assertEqual(self.ai_response(json.dumps({'misleading': False, 'reasoning': '부정 문맥'})), (False, '부정 문맥'))

    def test_stt_failure_preserves_ocr_and_warning(self):
        ocr = SimpleNamespace(extract=lambda path: [TimedText('차', 0, 1, 'ocr')])
        stt = SimpleNamespace(extract=lambda path: (_ for _ in ()).throw(RuntimeError('model unavailable')))
        result = TextExtractionService(ocr, stt).extract_all('test', None)
        self.assertEqual(result.full_text(), '차')
        self.assertTrue(result.warnings)
        self.assertEqual(result.channel_status['stt'], 'failed')

    def test_skipped_stt_is_explicit(self):
        ocr = SimpleNamespace(extract=lambda path: [])
        result = TextExtractionService(ocr).extract_all('test', None, run_stt=False)
        self.assertEqual(result.channel_status['stt'], 'skipped')
        self.assertTrue(result.warnings)

    def test_repeated_caption_extends_interval_but_blank_separates_occurrences(self):
        from pathlib import Path
        ocr = OCRExtractor()
        frames = [Path(f'frame_{i}.png') for i in range(4)]
        with patch.object(ocr, '_extract_frames', return_value=frames), patch('pipeline.extractors.Image.open'), patch('pipeline.extractors.pytesseract.image_to_string', side_effect=['완치', '완치', '', '완치']):
            segments = ocr.extract(Path('fixture.mp4'))
        self.assertEqual([(s.text, s.start_sec, s.end_sec) for s in segments], [('완치', 0, 2), ('완치', 3, 4)])


if __name__ == '__main__':
    unittest.main()

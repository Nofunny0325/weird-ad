import socket
import unittest
from unittest.mock import patch

from pipeline.extractors import ExtractedText, TimedText
from pipeline.judge_engine import ROOT, JudgeEngine


class InstalledModelTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'storage/models/e5/model.onnx').is_file() and
                         (ROOT/'storage/models/e5/tokenizer.json').is_file(),
                         'optional: run scripts/setup_models.py first')
    def test_cpu_model_flags_unlisted_phrasing_while_offline(self):
        with patch.object(socket.socket,'connect',side_effect=AssertionError('unexpected networking')):
            engine=JudgeEngine.from_local_data(ROOT/'storage/models/e5')
            self.assertIsNotNone(engine.retriever.dense)
            e=ExtractedText('test',stt_segments=[
                TimedText('이거 한 알이면 혈관에 쌓인 찌꺼기가 쓸려 나갑니다.',0,3,'stt'),
                TimedText('고소하고 산뜻한 맛입니다.',8,11,'stt')])
            result=engine.judge(e)
        self.assertIn('로컬 E5',result.model_mode)
        self.assertFalse(result.banned_hits or result.ambiguous_hits)
        self.assertEqual(result.verdict.value,'확인 필요')
        self.assertIn(('CTX-09',0,3),[(f.rule_id,f.start_sec,f.end_sec) for f in result.findings])
        self.assertTrue(all(f.start_sec!=8 for f in result.findings))


if __name__=='__main__':
    unittest.main()

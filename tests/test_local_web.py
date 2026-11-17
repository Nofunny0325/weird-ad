import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as web
from pipeline.extractors import ExtractedText, TimedText
from pipeline.judge_engine import JudgeEngine


class LocalWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = JudgeEngine.from_local_data(model_dir=None)

    def test_text_without_keyword_is_screened(self):
        with patch.object(web,'default_engine',return_value=self.engine):
            response = web.app.test_client().post('/review-text', data={
                'transcript': '이거 먹고 나면 병원에서 받아 온 약을 챙길 필요도 없더라고요.'})
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code,200)
        self.assertIn('위반 의심',body)
        self.assertIn('CTX-01',body)
        self.assertIn('입력 문장',body)

    def test_uploaded_video_is_removed_on_success_and_on_validation_failure(self):
        for valid in (True,False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                extracted=ExtractedText('id',ocr_segments=[TimedText('당뇨를 완치합니다.',3,4,'ocr')])
                with patch.object(web,'UPLOAD_TMP_DIR',root), patch.object(web,'default_engine',return_value=self.engine), \
                     patch.object(web,'probe_video',return_value=10 if valid else None), \
                     patch.object(web.TextExtractionService,'extract_all',return_value=extracted) as extract:
                    response=web.app.test_client().post('/analyze',data={'video':(io.BytesIO(b'fixture'),'../escape.mp4')})
                self.assertEqual(response.status_code,200 if valid else 400)
                self.assertEqual(list(root.iterdir()),[])
                if valid:
                    self.assertIn('3.0',response.get_data(as_text=True))
                    self.assertEqual(extract.call_args.args[1].parent.parent,root)
                else:
                    self.assertFalse(extract.called)

    def test_non_video_extension_is_rejected(self):
        response=web.app.test_client().post('/analyze',data={'video':(io.BytesIO(b'fixture'),'ad.exe')})
        self.assertEqual(response.status_code,400)


if __name__=='__main__':
    unittest.main()

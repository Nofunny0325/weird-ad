import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import app as web
from pipeline.extractors import ExtractedText, TimedText
from pipeline.judge_engine import JudgeEngine


class LocalWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = JudgeEngine.from_local_data(model_dir=None)

    def test_health_check_does_not_load_models(self):
        with patch.object(web, 'default_engine', side_effect=AssertionError('health loaded a model')):
            response = web.app.test_client().get('/api/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {'status': 'ok'})

    def test_text_without_keyword_is_screened(self):
        with patch.object(web,'default_engine',return_value=self.engine):
            response = web.app.test_client().post('/review-text', data={
                'transcript': '이거 먹고 나면 병원에서 받아 온 약을 챙길 필요도 없더라고요.'})
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code,200)
        self.assertIn('위반 의심',body)
        self.assertIn('CTX-33',body)
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

    def test_groq_is_used_only_when_selected_and_missing_key_keeps_local_review(self):
        client = Mock()
        client.complete.return_value = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"findings": []}'}}]}
        client.model = 'test-model'
        for selected, configured in [(False, True), (True, True), (True, False)]:
            with self.subTest(selected=selected, configured=configured), \
                 patch.object(web, 'configured_client', return_value=client if configured else None) as lookup, \
                 patch.object(web, 'default_engine', return_value=self.engine):
                response = web.app.test_client().post('/review-text', data={
                    'transcript': '당뇨를 완치합니다.', 'use_groq': 'on' if selected else ''})
            self.assertEqual(response.status_code, 200)
            body = response.get_data(as_text=True)
            self.assertIn('위반 의심', body)
            self.assertEqual(lookup.called, selected)
            if selected and configured:
                self.assertIn('Groq 문맥 검토 완료', body)
            elif selected:
                self.assertIn('Groq 연결 설정이 없어', body)


    def test_memory_constrained_upload_releases_models_between_stages(self):
        order = Mock()
        order.load_e5.return_value = self.engine
        order.extract.return_value = ExtractedText('id', ocr_segments=[TimedText('당뇨를 완치합니다.', 0, 1, 'ocr')])
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(web.app.config, LOW_MEMORY=True), \
             patch.object(web, 'UPLOAD_TMP_DIR', Path(temp)), \
             patch.object(web, 'probe_video', return_value=1), \
             patch.object(web, 'default_engine', order.load_e5), \
             patch.object(web, 'trim_model_pages', order.trim_model_pages), \
             patch.object(web.TextExtractionService, 'extract_all', order.extract):
            response = web.app.test_client().post('/analyze', data={
                'video': (io.BytesIO(b'fixture'), 'ad.mp4'), 'run_stt': 'on'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([entry[0] for entry in order.mock_calls], [
            'load_e5.cache_clear', 'trim_model_pages', 'extract', 'load_e5'])
        self.assertTrue(order.extract.call_args.kwargs['isolate_stt'])


if __name__=='__main__':
    unittest.main()

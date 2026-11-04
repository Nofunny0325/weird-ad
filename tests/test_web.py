import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as web
from pipeline.extractors import ExtractedText, TimedText


class WebReviewTests(unittest.TestCase):
    def test_upload_filename_cannot_choose_storage_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            upload_dir = Path(tmp) / 'uploads'
            upload_dir.mkdir()
            received = []
            def receive(path, name):
                received.append(Path(path))
                from types import SimpleNamespace
                return SimpleNamespace(video_id='test', saved_path=Path(path))
            with patch.object(web, 'UPLOAD_TMP_DIR', upload_dir), patch.object(web, 'check_dependencies', return_value={}), patch.object(web.InputHandler, 'handle_file_upload', side_effect=receive), patch.object(web.InputHandler, 'validate', return_value=False):
                response = web.app.test_client().post('/analyze', data={'video': (io.BytesIO(b'fixture'), '../escape.mp4')}, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 302)
            self.assertEqual(received[0].parent.resolve(), upload_dir.resolve())
            self.assertFalse((Path(tmp) / 'escape.mp4').exists())
            self.assertEqual(list(upload_dir.iterdir()), [])

    def test_result_explains_uncertainty_and_locations(self):
        from pipeline.judge_engine import JudgeEngine
        from data.wordlists_schema import WordLists
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / 'video.mp4'
            video_path.write_bytes(b'fixture')
            from types import SimpleNamespace
            video = SimpleNamespace(video_id='test', saved_path=video_path)
            extracted = ExtractedText('test', [TimedText('맛있는 차', 3, 4, 'ocr')])
            with patch.object(web, 'UPLOAD_TMP_DIR', Path(tmp)), patch.object(web, 'check_dependencies', return_value={}), patch.object(web.InputHandler, 'handle_file_upload', return_value=video), patch.object(web.InputHandler, 'validate', return_value=True), patch.object(web.TextExtractionService, 'extract_all', return_value=extracted):
                response = web.app.test_client().post('/analyze', data={'video': (io.BytesIO(b'fixture'), 'ad.mp4')}, content_type='multipart/form-data')
            text = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn('확인 필요', text)
            self.assertIn('성분', text)
            self.assertNotIn('badge ok', text)


if __name__ == '__main__':
    unittest.main()

import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.extractors import OCRExtractor, STTExtractor, TextExtractionService, TimedText


class ExtractionTests(unittest.TestCase):
    def test_stt_failure_preserves_ocr_and_warning(self):
        ocr=SimpleNamespace(extract=lambda path:[TimedText('차',0,1,'ocr')])
        def unavailable(path):
            raise RuntimeError('model unavailable')
        result=TextExtractionService(ocr,SimpleNamespace(extract=unavailable)).extract_all('test',None)
        self.assertEqual(result.full_text(),'차')
        self.assertEqual(result.channel_status,{'ocr':'completed','stt':'failed'})
        self.assertTrue(result.warnings)

    def test_stt_not_requested(self):
        result=TextExtractionService(SimpleNamespace(extract=lambda path:[])).extract_all('test',None,run_stt=False)
        self.assertEqual(result.channel_status,{'ocr':'empty','stt':'skipped'})
        self.assertTrue(result.warnings)

    def test_repeated_caption_extends_interval_until_blank_frame(self):
        ocr=OCRExtractor()
        with patch.object(ocr,'_extract_frames',return_value=[Path(f'x{i}.png') for i in range(4)]), \
             patch('pipeline.extractors.Image.open'), \
             patch('pipeline.extractors.pytesseract.image_to_string',side_effect=['완치','완치','','완치']):
            seg=ocr.extract(Path('fixture.mp4'))
        self.assertEqual([(s.text,s.start_sec,s.end_sec) for s in seg],[('완치',0,2),('완치',3,4)])

    def test_speech_model_receives_decoded_mono_pcm_without_pyav(self):
        stt=STTExtractor()
        def fake_audio(_,out):
            with wave.open(str(out),'wb') as wav:
                wav.setparams((1,2,16000,0,'NONE','not compressed'))
                wav.writeframes(bytes([0,0,0,128,0,64]))
        def transcribe(samples,**options):
            self.assertEqual(getattr(samples,'tolist',lambda:None)(),[0.,-1.,.5])
            self.assertEqual(options['language'],'ko')
            return [SimpleNamespace(text='문장',start=1.,end=2.)],None
        stt._model=SimpleNamespace(transcribe=transcribe)
        with patch.object(stt,'_extract_audio',fake_audio):
            self.assertEqual(stt.extract(Path('fixture.mp4')),[TimedText('문장',1.,2.,'stt')])


if __name__=='__main__':
    unittest.main()

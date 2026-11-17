"""
텍스트 추출 모듈
- OCR: 영상 프레임에서 화면에 보이는 텍스트(자막, 상품 문구 등) 추출
- STT: 오디오 트랙에서 나레이션 텍스트 추출

구현:
  - 프레임/오디오 분리: ffmpeg (subprocess 호출)
  - OCR: pytesseract (Tesseract OCR, 한국어 kor + 영어 eng)
  - STT: faster-whisper (Whisper 모델의 경량 구현체)

사전 준비물 (이 파일이 실제로 동작하려면):
  - 시스템: ffmpeg, tesseract-ocr (+ tesseract-ocr-kor 언어팩)
  - 파이썬: pip install pytesseract pillow faster-whisper
"""

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import tempfile
import shutil
import os
import re
import wave
from functools import lru_cache

from PIL import Image
import pytesseract

MAX_VIDEO_SECONDS = 180
DEFAULT_STT_DIR = Path(__file__).resolve().parents[1] / 'storage' / 'models' / 'whisper-small'
_INPUT_OPTIONS = ['-protocol_whitelist','file,pipe','-format_whitelist','mov,matroska,webm']


def _ffmpeg_command():
    return os.environ.get('FOOD_AD_FFMPEG', 'ffmpeg')


def probe_video(path: Path) -> float | None:
    """Read media stream metadata with ffmpeg without decoding the complete input."""
    try:
        result = subprocess.run([_ffmpeg_command(), '-hide_banner', '-nostdin', *_INPUT_OPTIONS, '-i', str(path)],
                                capture_output=True, timeout=12)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError('영상 정보 확인 시간 초과') from exc
    report=result.stderr.decode(errors='replace')
    if not re.search(r'Stream #[^\n]+Video:',report):
        return None
    match=re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)',report)
    if not match:
        return None
    h,m,s=map(float,match.groups())
    duration=h*3600+m*60+s
    return duration if duration>0 else None


@dataclass
class TimedText:
    """시간 정보가 있는 텍스트 조각 (자막 한 줄, 발화 한 문장 등)"""
    text: str
    start_sec: float
    end_sec: float
    source: str  # "ocr" 또는 "stt"


@dataclass
class ExtractedText:
    video_id: str
    ocr_segments: list[TimedText] = field(default_factory=list)
    stt_segments: list[TimedText] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    channel_status: dict[str, str] = field(default_factory=dict)

    def all_segments(self) -> list[TimedText]:
        """시간순 정렬된 전체 텍스트 조각"""
        return sorted(self.ocr_segments + self.stt_segments, key=lambda t: t.start_sec)

    def full_text(self) -> str:
        """판단엔진에 넘길 전체 텍스트 (OCR + STT 합본)"""
        return " ".join(seg.text for seg in self.all_segments())


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        [_ffmpeg_command(), "-nostdin", "-y", *_INPUT_OPTIONS, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 실행 실패: {result.stderr.decode(errors='ignore')[-2000:]}")


class OCRExtractor:
    """영상 프레임에서 화면 텍스트 추출 (Tesseract OCR)"""

    def __init__(self, frame_interval_sec: float = 1.0, lang: str = "kor+eng"):
        self.frame_interval_sec = frame_interval_sec
        self.lang = lang
        # psm 6 = "균일한 텍스트 블록"으로 가정하고 읽음.
        # 기본 psm(3, 자동 레이아웃 분석)은 여백이 많고 글자가 적은
        # 광고 자막류 프레임에서 "Empty page"로 오판하는 경우가 많아 고정값으로 지정.
        self.tesseract_config = "--psm 6"

    def _extract_frames(self, video_path: Path, out_dir: Path) -> list[Path]:
        fps = 1.0 / self.frame_interval_sec
        pattern = str(out_dir / "frame_%05d.png")
        _run_ffmpeg(["-i", str(video_path), "-t", str(MAX_VIDEO_SECONDS), "-vf", f"fps={fps}", pattern])
        return sorted(out_dir.glob("frame_*.png"))

    def extract(self, video_path: Path) -> list[TimedText]:
        segments: list[TimedText] = []
        prev_text = None  # 직전 프레임과 같은 텍스트면 중복으로 보고 건너뜀

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            frames = self._extract_frames(video_path, tmp_dir)

            for idx, frame_path in enumerate(frames):
                start_sec = idx * self.frame_interval_sec
                end_sec = start_sec + self.frame_interval_sec

                with Image.open(frame_path) as image:
                    text = pytesseract.image_to_string(
                        image, lang=self.lang, config=self.tesseract_config, timeout=12
                    ).strip()
                # 여러 줄 OCR 결과를 한 줄로 정리
                text = " ".join(line.strip() for line in text.splitlines() if line.strip())

                if not text:
                    prev_text = None
                    continue
                if text == prev_text:
                    segments[-1].end_sec = end_sec
                    continue

                prev_text = text
                segments.append(
                    TimedText(text=text, start_sec=start_sec, end_sec=end_sec, source="ocr")
                )

        return segments


@lru_cache(maxsize=2)
def _whisper_model(model_path):
    from faster_whisper import WhisperModel
    threads = 1 if os.getenv('FOOD_AD_LOW_MEMORY') == '1' else 0
    return WhisperModel(model_path, device='cpu', compute_type='int8', local_files_only=True, cpu_threads=threads)


def release_whisper_model():
    _whisper_model.cache_clear()


class STTExtractor:
    """영상 오디오 트랙에서 발화 텍스트 추출 (faster-whisper)"""

    def __init__(self, model_size: str = "small", language: str = "ko"):
        self.model_size = model_size
        self.language = language
        self._model = None  # 지연 로딩 (무거운 모델이라 실제 호출 시점에 로드)

    def _load_model(self):
        if self._model is None:
            model_path = os.environ.get('FOOD_AD_STT_DIR')
            if not model_path and self.model_size == 'small' and (DEFAULT_STT_DIR / 'model.bin').is_file():
                model_path = str(DEFAULT_STT_DIR)
            self._model = _whisper_model(model_path or self.model_size)
        return self._model

    def _extract_audio(self, video_path: Path, out_path: Path) -> None:
        _run_ffmpeg(
            ["-i", str(video_path), "-t", str(MAX_VIDEO_SECONDS), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(out_path)]
        )

    def extract(self, video_path: Path) -> list[TimedText]:
        model = self._load_model()

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "audio.wav"
            self._extract_audio(video_path, audio_path)

            import numpy as np
            with wave.open(str(audio_path),'rb') as source:
                if (source.getnchannels(),source.getsampwidth(),source.getframerate()) != (1,2,16000):
                    raise ValueError('Expected mono 16 kHz PCM')
                samples=np.frombuffer(source.readframes(source.getnframes()),dtype='<i2').astype(np.float32)/32768
            segments_iter, _info = model.transcribe(
                samples, language=self.language, vad_filter=True
            )

            segments = [
                TimedText(
                    text=seg.text.strip(),
                    start_sec=seg.start,
                    end_sec=seg.end,
                    source="stt",
                )
                for seg in segments_iter
                if seg.text.strip()
            ]

        return segments


class TextExtractionService:
    """OCR + STT를 묶어서 실행하는 상위 서비스"""

    def __init__(self, ocr: OCRExtractor | None = None, stt: STTExtractor | None = None):
        self.ocr = ocr or OCRExtractor()
        self.stt = stt or STTExtractor()

    def extract_all(
        self, video_id: str, video_path: Path, run_stt: bool = True
    ) -> ExtractedText:
        warnings = []
        status = {}
        ocr_segments = []
        try:
            ocr_segments = self.ocr.extract(video_path)
            status['ocr'] = 'completed' if ocr_segments else 'empty'
            if not ocr_segments:
                warnings.append('화면에서 텍스트를 추출하지 못했습니다. 화면 확인이 필요합니다.')
        except Exception:
            status['ocr'] = 'failed'
            warnings.append('화면 글자 추출에 실패했습니다. 화면 내용은 검토하지 못했습니다.')
        stt_segments: list[TimedText] = []
        if run_stt:
            try:
                stt_segments = self.stt.extract(video_path)
                status['stt'] = 'completed' if stt_segments else 'empty'
                if not stt_segments:
                    warnings.append('음성을 추출하지 못했습니다. 무음 영상인지 인식 누락인지 확인이 필요합니다.')
            except Exception:
                status['stt'] = 'failed'
                warnings.append('음성 분석에 실패했습니다. 음성 내용은 검토하지 못했습니다.')
        else:
            status['stt'] = 'skipped'
            warnings.append('음성 분석을 실행하지 않았습니다. 음성 내용은 미검토입니다.')

        return ExtractedText(
            video_id=video_id,
            ocr_segments=ocr_segments,
            stt_segments=stt_segments,
            warnings=warnings,
            channel_status=status,
        )


def check_dependencies() -> dict[str, bool]:
    """
    실행 환경에 필요한 외부 의존성이 설치되어 있는지 점검.
    main.py 실행 전에 호출해서 뭐가 빠졌는지 바로 알 수 있게 함.
    """
    status = {}
    status["ffmpeg"] = shutil.which(_ffmpeg_command()) is not None
    status["tesseract"] = shutil.which("tesseract") is not None
    try:
        import pytesseract as _  # noqa
        status["pytesseract"] = True
    except ImportError:
        status["pytesseract"] = False
    try:
        import faster_whisper as _  # noqa
        status["faster_whisper"] = True
    except ImportError:
        status["faster_whisper"] = False
    return status

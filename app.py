"""Flask UI for local food-ad screening."""
from ctypes import CDLL
from functools import wraps
import gc
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock
import uuid

from flask import Flask, render_template, request

from pipeline.extractors import (
    MAX_VIDEO_SECONDS, ExtractedText, TimedText, TextExtractionService,
    check_dependencies, probe_video,
)
from pipeline.judge_engine import ROOT, default_engine
from pipeline.groq_client import configured_client

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=64 * 1024 * 1024, LOW_MEMORY=os.getenv('FOOD_AD_LOW_MEMORY') == '1')
analysis_slot = Lock()
UPLOAD_TMP_DIR = ROOT / 'storage' / 'tmp_uploads'
ALLOWED_EXT = {'.mp4', '.mov', '.mkv', '.webm'}
MAX_TEXT_CHARS = 12_000


def page(*, result=None, full_text='', filename='', error='', status=200):
    return render_template('index.html', deps=check_dependencies(), result=result, error=error,
        filename=filename, full_text=full_text, max_duration=MAX_VIDEO_SECONDS, max_text=MAX_TEXT_CHARS,
        cloud_ready=bool(os.getenv('GROQ_API_KEY', '').strip())), status


def single_analysis_on_small_hosts(view):
    @wraps(view)
    def run():
        if not app.config['LOW_MEMORY']:
            return view()
        if not analysis_slot.acquire(blocking=False):
            return page(error='다른 검토가 진행 중입니다. 잠시 후 다시 시도하세요.', status=503)
        try:
            return view()
        finally:
            analysis_slot.release()
    return run


def trim_model_pages():
    gc.collect()
    if sys.platform == 'linux':
        trim = getattr(CDLL(None), 'malloc_trim', None)
        if trim is not None:
            trim(0)


def review_extracted(extracted, product_id, client):
    result = default_engine().judge(extracted, product_id, groq_client=client)
    if request.form.get('use_groq') == 'on' and client is None:
        result.context_status = 'unavailable'
        result.limitations.append('Groq 연결 설정이 없어 로컬 기준으로 검토했습니다.')
    return result


@app.errorhandler(413)
def too_large(_):
    return page(error='업로드 요청은 64 MiB보다 작아야 합니다.', status=413)


@app.get('/')
def index():
    return page()


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.post('/review-text')
@single_analysis_on_small_hosts
def review_text():
    transcript = request.form.get('transcript', '').strip()
    if not transcript or len(transcript)>MAX_TEXT_CHARS:
        return page(error=f'문장을 1~{MAX_TEXT_CHARS:,}자로 입력하세요.', status=400)
    # Offsets are not timestamps for pasted transcripts; the UI labels this channel separately.
    extracted=ExtractedText('text',stt_segments=[TimedText(transcript,0,0,'text')],
        channel_status={'ocr':'skipped','stt':'skipped'})
    client = configured_client() if request.form.get('use_groq') == 'on' else None
    result = review_extracted(extracted, request.form.get('product_id', ''), client)
    return page(result=result,full_text=transcript,filename='입력 문장')


@app.post('/analyze')
@single_analysis_on_small_hosts
def analyze():
    file=request.files.get('video')
    ext=Path(file.filename or '').suffix.lower() if file else ''
    if ext not in ALLOWED_EXT:
        return page(error='MP4, MOV, MKV, WEBM 영상 파일을 선택하세요.',status=400)
    UPLOAD_TMP_DIR.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='review-',dir=UPLOAD_TMP_DIR) as directory:
        path=Path(directory)/('video'+ext)
        try:
            file.save(path)
            duration=probe_video(path)
            if duration is None:
                return page(error='영상 형식 또는 재생 시간을 확인할 수 없습니다.',status=400)
            if duration>MAX_VIDEO_SECONDS:
                return page(error=f'{MAX_VIDEO_SECONDS}초 이하 영상만 분석할 수 있습니다.',status=400)
            run_stt = request.form.get('run_stt') == 'on'
            client = configured_client() if request.form.get('use_groq') == 'on' else None
            limit_memory = app.config['LOW_MEMORY']
            if limit_memory:
                default_engine.cache_clear()
                trim_model_pages()
            extracted = TextExtractionService().extract_all(
                uuid.uuid4().hex, path, run_stt=run_stt, isolate_stt=limit_memory, groq_client=client,
            )
            result = review_extracted(extracted, request.form.get('product_id', ''), client)
            return page(result=result,full_text=extracted.full_text(),filename=Path(file.filename or '동영상').name)
        except (FileNotFoundError, TimeoutError):
            return page(error='영상 처리 도구를 실행할 수 없습니다. ffmpeg 설치 상태를 확인하세요.',status=503)
        except OSError:
            app.logger.exception('Local video processing failed')
            return page(error='서버에서 영상 파일을 처리할 수 없습니다.',status=500)


if __name__=='__main__':
    port=int(os.getenv('PORT','5000'))
    if os.getenv('PORT'):
        from waitress import serve
        serve(app,host='0.0.0.0',port=port,threads=2)
    else:
        app.run(host='127.0.0.1',port=port,debug=False)

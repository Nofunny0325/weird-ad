"""Bounded Groq inference calls. Credentials and provider error bodies stay server-side."""

from functools import lru_cache
from http.client import HTTPException
import json
import os
from pathlib import Path
from threading import Lock
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

API_ROOT = 'https://api.groq.com/openai/v1/'
MAX_RESPONSE_BYTES = 1024 * 1024


class GroqError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)

    @property
    def user_message(self):
        return {
            'rate_limited': 'Groq 사용 한도에 도달해 로컬 검사로 이어갔습니다.',
            'unauthorized': 'Groq 인증을 확인할 수 없어 로컬 검사로 이어갔습니다.',
            'timeout': 'Groq 응답 시간이 초과되어 로컬 검사로 이어갔습니다.',
            'invalid_response': 'Groq 응답의 원문 근거 또는 형식을 확인하지 못했습니다.',
            'generation_incomplete': 'Groq 문맥 답변의 생성을 완료하지 못해 로컬 검사로 이어갔습니다.',
        }.get(self.code, 'Groq에 연결하지 못해 로컬 검사로 이어갔습니다.')


class GroqClient:
    def __init__(self, api_key, model=None, stt_model=None):
        self._api_key = api_key
        self.model = model or 'openai/gpt-oss-120b'
        self.stt_model = stt_model or 'whisper-large-v3'
        self._cooldowns = {}
        self._lock = Lock()

    def _request(self, endpoint, data, content_type, timeout):
        with self._lock:
            blocked_until, reason = self._cooldowns.get(endpoint, (0, 'rate_limited'))
            if monotonic() < blocked_until:
                raise GroqError(reason)
        request = Request(API_ROOT + endpoint, data=data, headers={
            'Authorization': 'Bearer ' + self._api_key,
            'Content-Type': content_type,
            'User-Agent': 'weird-ad-review/1.0',
        })
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise GroqError('invalid_response')
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise GroqError('invalid_response')
            return value
        except HTTPError as error:
            code = 'request_failed'
            seconds = 15
            if error.code == 400:
                try:
                    body = json.loads(error.read(MAX_RESPONSE_BYTES))
                except (ValueError, HTTPException, OSError):
                    body = {}
                detail = body.get('error', {}) if isinstance(body, dict) else {}
                if isinstance(detail, dict) and detail.get('code') == 'json_validate_failed':
                    code = 'generation_incomplete'
            if error.code == 429:
                code = 'rate_limited'
                try:
                    seconds = max(1, min(86400, float(error.headers.get('Retry-After', '60'))))
                except (TypeError, ValueError):
                    seconds = 60
            elif error.code in (401, 403):
                code, seconds = 'unauthorized', 300
            with self._lock:
                self._cooldowns[endpoint] = (monotonic() + seconds, code)
            error.close()
            raise GroqError(code) from None
        except (TimeoutError, URLError) as error:
            code = 'timeout' if isinstance(error, TimeoutError) else 'connection_failed'
            raise GroqError(code) from None
        except HTTPException:
            raise GroqError('connection_failed') from None
        except (ValueError, UnicodeError, OSError):
            raise GroqError('invalid_response') from None

    def complete(self, messages, schema):
        payload = {
            'model': self.model, 'messages': messages, 'temperature': 0,
            'max_completion_tokens': 5120,
            'reasoning_effort': 'high',
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'food_ad_review', 'strict': True, 'schema': schema,
            }},
        }
        return self._request('chat/completions', json.dumps(payload).encode(), 'application/json', 35)

    def transcribe(self, audio_path: Path):
        if audio_path.stat().st_size > 24 * 1024 * 1024:
            raise GroqError('audio_too_large')
        boundary = 'food-ad-' + uuid.uuid4().hex
        fields = {
            'model': self.stt_model, 'language': 'ko', 'temperature': '0',
            'response_format': 'verbose_json', 'timestamp_granularities[]': 'segment',
        }
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
                 for key, value in fields.items()]
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                      'Content-Type: audio/wav\r\n\r\n').encode())
        parts.extend([audio_path.read_bytes(), f'\r\n--{boundary}--\r\n'.encode()])
        return self._request('audio/transcriptions', b''.join(parts),
                             'multipart/form-data; boundary=' + boundary, 40)


@lru_cache(maxsize=2)
def _configured_client(api_key, model, stt_model):
    return GroqClient(api_key, model, stt_model)


def configured_client():
    key = os.getenv('GROQ_API_KEY', '').strip()
    if not key:
        return None
    return _configured_client(key, os.getenv('GROQ_REVIEW_MODEL', 'openai/gpt-oss-120b'),
                              os.getenv('GROQ_STT_MODEL', 'whisper-large-v3'))

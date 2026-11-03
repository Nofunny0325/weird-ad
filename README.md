# 부당광고 판별 파이프라인

## 설치

```bash
# 시스템 패키지 (Mac: brew install ffmpeg tesseract tesseract-lang / Ubuntu: apt install ffmpeg tesseract-ocr tesseract-ocr-kor)
pip install -r requirements.txt
```

ANTHROPIC_API_KEY 환경변수 설정 (오인유발어 감지 시에만 호출됨):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 확인

```bash
python3 main.py --check
```

ffmpeg / tesseract(+한국어팩) / pytesseract / faster-whisper 설치 여부와
ANTHROPIC_API_KEY 설정 여부를 보여준다.

## 실행

```bash
# OCR + STT + 판정 전체
python3 main.py --video 내영상.mp4

# STT 없이 OCR만 (faster-whisper 미설치 상태에서도 테스트 가능)
python3 main.py --video 내영상.mp4 --no-stt
```

## 실제로 동작 확인된 부분 (2026-11-03 기준)

- OCR: ffmpeg 프레임 추출 → pytesseract(kor+eng, `--psm 6`) → 한국어 자막 텍스트 정확히 인식 확인
- 금지어 매칭 → 즉시 "부당광고" 판정 (AI 호출 없이 종료) 확인
- 오인유발어 매칭 → AI 분석 단계 진입까지 확인 (AI 호출 자체는 API 키/네트워크 필요해서 실제 테스트 환경에서는 미실행)

## 아직 채워야 할 것

- `data/banned_words.sample.json`, `data/ambiguous_words.sample.json`
  → 팀에서 만들고 있는 실제 단어 목록으로 교체
- `judge_engine.py`의 `AIContextAnalyzer` 프롬프트
  → 표시광고법 실제 판단 기준 반영해서 다듬기 (지금은 뼈대 프롬프트)
- STT 테스트: `faster-whisper` 설치 후 실제 나레이션 있는 영상으로 확인 필요
  (현재 개발 환경은 네트워크 제한으로 모델 다운로드 불가 → 팀 PC에서 테스트 권장)
- 문장 분리기: 현재 `.!?` 기준 단순 분리 — 한국어 문장 특성상
  종결어미 기반 분리기로 교체하면 더 정확해짐
- 단어 매칭: `kiwipiepy` 형태소 분석 기반으로 전환 완료
  (설치 안 되어 있으면 자동으로 substring 매칭으로 폴백 — 개발 환경 네트워크 제한으로
  이 저장소 안에서는 실제 kiwipiepy 호출까지는 테스트하지 못함, 설치 후 확인 필요)

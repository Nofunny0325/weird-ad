# 식품 광고 검토 파이프라인

영상 업로드 → 화면 OCR·선택적 음성 전사 → 주의 표현 탐색 → 선택적 AI 문맥 분석 → 근거 구간·판단 이유·검토 한계 표시.

## 현재 결과의 의미 (2026-11-04)

- **위반 의심:** AI 문맥 분석에서 오인 가능성이 제기됨. 사람의 검토가 필요한 선별 결과이며 위법 확정이 아니다.
- **확인 필요:** 빈 추출, 사전 미탐지, 문맥 분석 실패/미설정, 제품 정보 미검토 등. 정상 광고라고 간주하지 않는다.
- `검토 범위 내 주요 의심 없음` 상태는 향후 검토 범위가 충족될 때를 위한 결과 형식이며 현재 엔진에서는 자동으로 출력하지 않는다.

현재 사전은 4개 예시 표현이다. 제품 분류·성분/함량·인정 기능성·인체시험·연결 판매페이지·화면의 비언어적 연출은 미검토다. 금지어 파일의 `banned` 필드명은 호환을 위해 유지하지만 단어 자체로 위법을 확정하지 않는다. 학습된 식품광고 전용 모델이나 법적 적법성 인증 서비스가 아니다.

## 실행

Python 3.10 이상을 사용한다. Windows PowerShell 예:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 main.py --check
.\.venv\Scripts\python.exe -X utf8 app.py
```

브라우저에서 http://127.0.0.1:5000 을 연다. 화면 OCR에는 PATH에 있는 ffmpeg와 Tesseract 및 한국어 `kor` 언어 데이터가 필요하다. `tesseract --list-langs`로 한국어 설치를 확인한다. 음성 전사에는 faster-whisper와 모델 다운로드가 필요하며 첫 실행 시 시간이 걸릴 수 있다.

AI 문맥 분석에는 `ANTHROPIC_API_KEY`와 anthropic 패키지가 필요하다. 키를 코드나 저장소에 넣지 말고 실행 환경 변수로 설정한다. AI를 사용하면 감지 문구와 주변 추출 텍스트가 외부 API로 전송된다. 키가 없거나 호출·응답 검증에 실패하면 검출 근거를 유지하고 확인 필요로 표시한다. AI는 우선 검토 표현과 오인유발 표현 모두에 대해 호출될 수 있다.

전체 영상 기능 없이 판정·웹 회귀 테스트만 실행하려면 flask, pillow, pytesseract만 설치하면 된다. 테스트에서는 실제 외부 AI 호출과 영상 인식 엔진을 대체한다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

CLI:

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py --video 광고.mp4
.\.venv\Scripts\python.exe -X utf8 main.py --video 광고.mp4 --no-stt
```

결과 JSON에는 표현, 출처 채널, 시작/종료 초, 주변 문맥, 판단 이유, 추출 상태와 미검토 항목이 포함된다. OCR 시간은 1초 간격 표본의 추정 구간이며 실제 글자가 나타난 정확한 순간이 아니다. 인접한 동일 채널 두 구간의 간격이 2초 이내이면 구간 경계에 걸친 표현도 탐색한다. 세 구간 이상에 걸친 표현과 긴 거리의 문맥은 별도 검토가 필요하다.

## 이번 검증과 남은 작업

회귀 테스트는 빈 텍스트·부정문 후보·AI 오류/문자열 boolean·동일 문구의 채널별 시각·인접 발화의 표현과 부정 문맥·반복 OCR 자막·부분 추출·웹 결과·임시 업로드 경로를 확인한다. 실제 한국어 광고 영상에서의 OCR/STT 정확도와 실제 Anthropic API 응답 품질은 이번 변경에서 검증하지 않았다.

다음 단계는 사례에 근거한 표현/기준 데이터, 제품 식별과 공식 기록 대조, 인체시험 적용성 검토, 실제 정상·위반·경계 광고 평가다. URL 영상 다운로드는 아직 구현되지 않았다. `web_demo.html`은 이전 독립 데모이며 실행 서버의 화면은 `templates/index.html`이다. Docker/클라우드 배포는 이번 변경에서 검증하지 않았다.

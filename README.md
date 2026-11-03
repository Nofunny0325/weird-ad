# 부당광고 판별 파이프라인

## 설치 (Windows)

1. **ffmpeg** 설치
   - https://www.gyan.dev/ffmpeg/builds/ 에서 "release essentials" zip 다운로드
   - 압축 풀고 안의 `bin` 폴더 경로를 환경변수 PATH에 추가
   - 확인: 새 터미널(cmd/PowerShell) 열고 `ffmpeg -version`

2. **Tesseract OCR** 설치
   - https://github.com/UB-Mannheim/tesseract/wiki 에서 Windows 설치파일 다운로드·실행
   - 설치 중 "Additional language data" 항목에서 **Korean** 체크 (한국어 인식 필수)
   - 설치 후 `C:\Program Files\Tesseract-OCR` 을 PATH에 추가
   - 확인: `tesseract --version`, `tesseract --list-langs` (목록에 `kor` 있어야 함)

3. **파이썬 패키지**
   ```
   pip install -r requirements.txt
   ```

4. **ANTHROPIC_API_KEY** 설정 (오인유발어 감지 시에만 호출됨)
   - PowerShell: `setx ANTHROPIC_API_KEY "sk-ant-..."` (설정 후 터미널 재시작)
   - 또는 cmd: `set ANTHROPIC_API_KEY=sk-ant-...` (이 터미널 세션에서만 유효)

## 설치 (Mac / Linux)

```bash
# Mac
brew install ffmpeg tesseract tesseract-lang

# Ubuntu
sudo apt install ffmpeg tesseract-ocr tesseract-ocr-kor

pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## 확인

```bash
python3 main.py --check
```

ffmpeg / tesseract(+한국어팩) / pytesseract / faster-whisper 설치 여부와
ANTHROPIC_API_KEY 설정 여부를 보여준다. 넷 다 "OK"가 떠야 전체 기능이 동작함.

## 웹사이트로 실행하기

```bash
python3 app.py
```

터미널에 `Running on http://127.0.0.1:5000` 이 뜨면, 브라우저에서
**http://127.0.0.1:5000** 접속. 영상 파일 업로드 → "심사하기" 누르면
서버가 OCR(+STT)로 텍스트를 뽑아서 바로 판정 결과를 보여준다.

- 로컬에서만 동작 (내 컴퓨터에서만 접속 가능, 외부 공개 아님 — 해커톤 데모용으로는 충분)
- STT 체크박스는 `faster-whisper`가 설치되어 있어야 의미 있음 (없으면 자동으로 OCR만 실행)
- 실제로 영상 업로드해서 끝까지 동작하는 것 확인함 (아래 "동작 확인" 참고)

## 클라우드 배포 (내 컴퓨터가 꺼져 있어도 접속 가능하게)

`127.0.0.1`은 본인 컴퓨터 안에서만 열리는 주소라서, 컴퓨터가 꺼지면 당연히 접속이 끊긴다.
다른 사람이 24시간 아무 때나 접속하게 하려면 **클라우드 서버**에 올려야 한다.
이 저장소에는 `Dockerfile`이 포함되어 있어서, 서버 쪽에 ffmpeg/tesseract(한국어팩)를
자동으로 설치하도록 되어 있다 — 즉 접속하는 사람도, 배포하는 사람도 따로 설치할 게 없다.

**Render (무료 티어) 로 배포하는 법**

1. https://render.com 가입 (GitHub 계정으로 로그인 가능)
2. "New +" → "Web Service" 선택
3. 이 GitHub 저장소(`kim090624/weird-ad`) 연결
4. 설정:
   - **Environment**: `Docker` 선택 (저장소에 있는 `Dockerfile`을 자동으로 인식함)
   - **Instance Type**: Free
5. "Environment Variables"에 추가:
   - `ANTHROPIC_API_KEY` = `sk-ant-...` (본인 키)
6. "Create Web Service" 클릭 → 빌드 로그가 뜨면서 몇 분 기다리면
   `https://weird-ad-xxxx.onrender.com` 같은 주소가 생성됨
7. 이 주소를 아무한테나 공유하면, 본인 컴퓨터가 꺼져 있어도 접속 가능

**주의할 점**
- 무료 티어는 한동안 아무도 안 들어오면 서버가 잠들었다가, 누가 들어오면 다시 깨어남
  (깨어나는 데 몇십 초 정도 걸릴 수 있음 — 해커톤 심사 직전에 한 번 미리 접속해서 깨워두면 좋음)
- faster-whisper(STT)는 처음 실행 시 모델 파일을 받느라 느릴 수 있음 — 무료 티어 메모리가
  부족하면 STT 체크박스를 끄고 OCR만 쓰는 것도 방법
- Dockerfile은 이 환경(개발 샌드박스)에 Docker가 없어서 직접 빌드 테스트는 못 했음.
  `ffmpeg`/`tesseract-ocr`/`tesseract-ocr-kor`는 Debian 계열 표준 패키지명이라
  일반적으로 문제없이 설치되지만, Render에 처음 배포할 때 빌드 로그를 한 번 확인할 것

## CLI로 실행하기 (터미널에서 바로)

```bash
# OCR + STT + 판정 전체
python3 main.py --video 내영상.mp4

# STT 없이 OCR만 (faster-whisper 미설치 상태에서도 테스트 가능)
python3 main.py --video 내영상.mp4 --no-stt
```

## 실제로 동작 확인된 부분 (2026-11-03 기준)

- OCR: ffmpeg 프레임 추출 → pytesseract(kor+eng, `--psm 6`) → 한국어 자막 텍스트 정확히 인식 확인
- 금지어 매칭 → 즉시 "부당광고" 판정 (AI 호출 없이 종료) 확인
- 오인유발어 매칭 → AI 분석 단계 진입까지 확인
- 웹서버(`app.py`): 실제로 브라우저 대신 curl로 영상 파일을 업로드해서
  OCR → 금지어 매칭 → "부당광고" 판정 → 결과 페이지 렌더링까지 전부 확인함
- (참고) AI 분석 호출 자체는 claude.ai 아티팩트 데모에서 실제로 성공 확인함
  (Claude API로 "업계 최고의 효과는 객관적 비교 근거 없이..." 같은 실제 분석 근거를 받음).
  다만 이 저장소의 `judge_engine.py`가 쓰는 `anthropic` SDK 경로는 API 키가 있는
  팀 PC에서 별도로 확인 필요 (이 개발 환경은 네트워크 제한으로 anthropic 패키지 설치 불가)

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

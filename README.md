# 건강기능식품 광고 검토

광고 영상을 올리거나 광고 문장을 입력하면, 사전에 걸리는 말이 없어도 화면 글자와 발화의 각 구간을 검토합니다. 분석은 서버 CPU에서 실행하며 영상이나 전사본을 외부 모델 API로 전송하지 않습니다.

| 결과 | 의미 |
|---|---|
| 위반 의심 | 질병 치료·약 대체·수치 보장 등의 명확한 주장 패턴이 잡힘. 근거 장면 확인 필요. |
| 확인 필요 | 유사 후보, 미확인 제품, 근거 부족 또는 미탐지를 포함. 적법 판정 아님. |

**로컬 판정:** 한국어 문자를 2~4자 단위로 비교하는 TF‑IDF와, 다국어 E5-small 문장 임베딩을 함께 사용합니다. 위험 예문과 인정 기능성·주의 안내 등 비교 문구를 함께 검색합니다. 유사 후보는 검토에 표시할 뿐 위반 의심으로 자동 승격하지 않습니다. 별도의 규칙이 긍정 주장, 약 대체, 특정 기간·감량량 보장 등을 확인하며 각각 근거 구간을 표시합니다. E5 파일이 없거나 로드되지 않으면 문자 유사도와 표현 규칙으로 계속 실행합니다.

**데이터:** `data/food_ad_review.json`에 표현 62개, 문맥 검토 규칙 32개, 기능성 대조 기준 16개가 들어 있습니다. 검토 예시는 학습·성능 검증용 실광고가 아닙니다. `data/product_catalog.json`에는 1개 제품의 공개 예시가 들어 있습니다. 운영 데이터는 같은 JSON 구조로 만들고 `FOOD_AD_PRODUCT_CATALOG`에 로컬 파일 경로를 지정할 수 있습니다. 입력된 제품 식별번호가 정확히 일치할 때만 기록과 문구를 대조합니다. 영상 속 제품을 자동으로 식별하거나 외부 사이트를 실시간 조회하지 않습니다.

## Windows 로컬 실행

Python 3.10 이상, 영상 사용 시 ffmpeg와 Tesseract(`kor`, `eng`)가 필요합니다. 문장 검토에는 영상 도구가 필요하지 않습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 scripts\setup_models.py
.\.venv\Scripts\python.exe -X utf8 main.py --check
.\.venv\Scripts\python.exe -X utf8 app.py
```

브라우저에서 http://127.0.0.1:5000 을 엽니다. ffmpeg가 PATH에 없으면 `FOOD_AD_FFMPEG`에 실행 파일 경로를 지정할 수 있습니다. Tesseract는 PATH에 추가해야 합니다.

모델 설치는 공개 가중치를 한 번 받는 과정입니다. `setup_models.py`는 지정 버전의 E5 모델·토크나이저(합계 약 135MB)를 내려받고 SHA‑256을 확인합니다. 음성 인식도 사용하려면 `scripts/setup_models.py --with-stt`를 실행하세요. Whisper도 로컬에서 실행하며, 설치 경로는 `storage/models/whisper-small`입니다. 다른 로컬 모델은 `FOOD_AD_STT_DIR`로 지정할 수 있습니다. 둘 다 API 키가 필요하지 않습니다. 오프라인 장비로 옮기려면 로컬 모델 디렉터리와 Python 패키지를 미리 준비합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py --text "이걸 먹고부터 병원에서 받아 온 약을 챙길 필요도 없었어요."
.\.venv\Scripts\python.exe -X utf8 main.py --video 광고.mp4 --no-stt
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

## Docker

`docker build -t weird-ad .`로 공개 E5 모델이 포함된 이미지를 만들고, `docker run --rm -p 5000:5000 weird-ad`로 실행합니다. 빌드에 `--build-arg WITH_STT=1`을 추가하면 메모리 사용량이 적은 다국어 Whisper `base`가 포함됩니다. 여유 메모리가 있는 환경에서 `small`을 쓰려면 `--build-arg WHISPER_MODEL=small`을 추가합니다. 외부 네트워크는 모델 설치 또는 빌드에만 필요합니다. 영상은 요청 중 임시 보관하며, 지원 길이는 180초 이하이고 업로드 요청은 64 MiB 미만입니다.

Docker는 메모리가 적은 서버에서도 실행할 수 있도록 한 번에 한 요청을 검토합니다. 충분한 메모리가 있는 서버에서는 `docker run`에 `-e FOOD_AD_LOW_MEMORY=0`을 추가할 수 있습니다.

## 검토 범위

화면 글자는 1초 간격 OCR이라 짧게 표시되는 문구를 놓칠 수 있습니다. 전후 사진의 비언어적 암시, 제품 성분 실측, 시험 결과의 완제품 적용 여부, 판매페이지, 자율심의는 자동 확인하지 않습니다. 모델은 한국어 광고 위반 분류기에 맞춰 학습하거나 보정하지 않았습니다. 운영 성능을 평가하려면 실제 광고를 캠페인 단위로 분리한 별도 평가셋에서 미탐·오탐·사람 검토로 넘긴 비율을 측정해야 합니다.

모델 출처: [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) · [faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small)

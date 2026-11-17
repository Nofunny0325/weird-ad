# 클라우드 배포용 이미지.
# 서버 쪽에 ffmpeg + tesseract(한국어팩)를 직접 설치하므로,
# 접속자나 배포 플랫폼이 따로 아무것도 설치할 필요 없음.
FROM python:3.11-slim

# ffmpeg(영상 처리) + tesseract-ocr(OCR) + 한국어 언어팩
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-kor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Public weights are fetched once during the build; inference needs no network.
# Build with --build-arg WITH_STT=1 to also include the optional speech model.
ARG WITH_STT=0
ARG WHISPER_MODEL=base
RUN if [ "$WITH_STT" = "1" ]; then python scripts/setup_models.py --with-stt --stt-model "$WHISPER_MODEL"; else python scripts/setup_models.py; fi
ENV PORT=5000 HF_HUB_OFFLINE=1 FOOD_AD_STT_DIR=/app/storage/models/whisper-${WHISPER_MODEL} \
    FOOD_AD_LOW_MEMORY=1 OMP_THREAD_LIMIT=1
EXPOSE 5000
CMD ["python3", "app.py"]

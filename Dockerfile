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

# Render 등은 컨테이너에 PORT 환경변수를 넘겨줌. app.py가 이를 읽어서 바인딩함.
ENV FLASK_DEBUG=0
CMD ["python3", "app.py"]

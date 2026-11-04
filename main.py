"""
전체 파이프라인 진입점

영상 업로드 → 텍스트 추출(OCR+STT) → 판단엔진 → 결과 반환

실행 전 체크: python3 main.py --check
"""

import argparse
import json
import os
import sys

from pipeline.input_handler import InputHandler
from pipeline.extractors import TextExtractionService, OCRExtractor, STTExtractor, check_dependencies
from pipeline.judge_engine import JudgeEngine, AIContextAnalyzer
from data.wordlists_schema import WordLists


def run_pipeline(
    uploaded_file_path: str,
    original_filename: str,
    run_stt: bool = True,
    banned_path: str = "data/banned_words.sample.json",
    ambiguous_path: str = "data/ambiguous_words.sample.json",
):
    # 1. 입력 처리
    input_handler = InputHandler()
    video = input_handler.handle_file_upload(uploaded_file_path, original_filename)

    if not input_handler.validate(video):
        raise ValueError("유효하지 않은 영상 파일입니다.")

    # 2. 텍스트 추출 (OCR + STT)
    extraction_service = TextExtractionService(
        ocr=OCRExtractor(frame_interval_sec=1.0),
        stt=STTExtractor(model_size="small", language="ko"),
    )
    extracted_text = extraction_service.extract_all(video.video_id, video.saved_path, run_stt=run_stt)

    # 3. 단어 사전 로드
    wordlists = WordLists.load(banned_path=banned_path, ambiguous_path=ambiguous_path)

    # 4. 판단엔진 실행 (ANTHROPIC_API_KEY 환경변수 필요 — 주의 표현이 감지된 경우 호출 가능)
    judge_engine = JudgeEngine(wordlists=wordlists, ai_analyzer=AIContextAnalyzer())
    result = judge_engine.judge(extracted_text)

    return result, extracted_text


def print_dependency_check():
    status = check_dependencies()
    print("=== 의존성 점검 ===")
    for name, ok in status.items():
        mark = "OK" if ok else "누락"
        print(f"  [{mark}] {name}")
    if not status["faster_whisper"]:
        print("\n  -> pip install faster-whisper  (STT 사용 시 필요)")
    if not status["pytesseract"]:
        print("\n  -> pip install pytesseract pillow")
    print(f"\n  ANTHROPIC_API_KEY 설정됨: {'예' if os.environ.get('ANTHROPIC_API_KEY') else '아니오 (주의 표현 문맥 분석 시 필요)'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="부당광고 판별 파이프라인")
    parser.add_argument("--check", action="store_true", help="의존성 설치 상태만 확인하고 종료")
    parser.add_argument("--video", type=str, help="분석할 영상 파일 경로")
    parser.add_argument("--no-stt", action="store_true", help="STT(음성 인식) 건너뛰고 OCR만 실행")
    args = parser.parse_args()

    if args.check:
        print_dependency_check()
        sys.exit(0)

    if not args.video:
        parser.print_help()
        sys.exit(1)

    result, extracted = run_pipeline(
        uploaded_file_path=args.video,
        original_filename=os.path.basename(args.video),
        run_stt=not args.no_stt,
    )

    print("=== 추출된 텍스트 ===")
    print(extracted.full_text())
    print("\n=== 판정 결과 ===")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

"""Command line interface for local ad screening."""
import argparse
import json
from pathlib import Path

from pipeline.extractors import (
    MAX_VIDEO_SECONDS, ExtractedText, TimedText, TextExtractionService, check_dependencies, probe_video,
)
from pipeline.judge_engine import default_engine
from pipeline.groq_client import configured_client


def main():
    parser=argparse.ArgumentParser(description='로컬 기준과 선택적 Groq 보강으로 식품 광고 선별 검토')
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--check',action='store_true',help='의존성 확인')
    source.add_argument('--video',type=Path,help='영상 파일')
    source.add_argument('--text',help='검토할 광고 문장')
    parser.add_argument('--no-stt',action='store_true')
    parser.add_argument('--groq',action='store_true',help='Groq로 음성·문장을 전송하여 보강 검토')
    parser.add_argument('--product-id',default='',help='로컬 목록에서 대조할 제품 식별번호')
    args=parser.parse_args()
    client = configured_client() if args.groq else None
    if args.groq and client is None:
        parser.error('Groq 보강 검토에는 서버 환경변수 GROQ_API_KEY가 필요합니다.')
    if args.check:
        engine = default_engine()
        print(json.dumps({**check_dependencies(),'e5_model':engine.retriever.dense is not None},
                         ensure_ascii=False,indent=2))
        return
    if args.video:
        try:
            duration=probe_video(args.video)
        except (FileNotFoundError,TimeoutError):
            parser.error('ffmpeg를 실행할 수 없습니다.')
        if not duration or duration>MAX_VIDEO_SECONDS:
            parser.error(f'{MAX_VIDEO_SECONDS}초 이하의 유효한 영상 파일이 필요합니다.')
        extracted=TextExtractionService().extract_all('video',args.video,run_stt=not args.no_stt, groq_client=client)
    else:
        extracted=ExtractedText('text',stt_segments=[TimedText(args.text,0,0,'text')])
    data=default_engine().judge(extracted,args.product_id,groq_client=client).to_dict()
    data['extracted_text']=extracted.full_text()
    print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

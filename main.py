"""Command line interface for local ad screening."""
import argparse
import json
from pathlib import Path

from pipeline.extractors import (
    MAX_VIDEO_SECONDS, ExtractedText, TimedText, TextExtractionService, check_dependencies, probe_video,
)
from pipeline.judge_engine import default_engine


def main():
    parser=argparse.ArgumentParser(description='외부 추론 API 없이 식품 광고 선별 검토')
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--check',action='store_true',help='의존성 확인')
    source.add_argument('--video',type=Path,help='영상 파일')
    source.add_argument('--text',help='검토할 광고 문장')
    parser.add_argument('--no-stt',action='store_true')
    parser.add_argument('--product-id',default='',help='로컬 목록에서 대조할 제품 식별번호')
    args=parser.parse_args()
    engine=default_engine()
    if args.check:
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
        extracted=TextExtractionService().extract_all('video',args.video,run_stt=not args.no_stt)
    else:
        extracted=ExtractedText('text',stt_segments=[TimedText(args.text,0,0,'text')])
    data=engine.judge(extracted,args.product_id).to_dict()
    data['extracted_text']=extracted.full_text()
    print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

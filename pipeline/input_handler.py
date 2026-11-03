"""
입력 처리 모듈
- 소비자가 업로드한 녹화 영상(mp4 등)을 받아서
  이후 단계(OCR, STT)가 쓸 수 있는 형태로 저장/정리한다.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import uuid


@dataclass
class UploadedVideo:
    video_id: str          # 내부 처리용 고유 ID
    original_filename: str
    saved_path: Path       # 서버에 저장된 원본 영상 경로
    source_url: str | None = None  # URL 입력일 경우 원본 링크 (없으면 None)


class InputHandler:
    """
    업로드 파일 또는 URL을 받아 처리 가능한 로컬 파일로 만든다.
    """

    def __init__(self, storage_dir: str = "storage/uploads"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def handle_file_upload(self, uploaded_file_path: str, original_filename: str) -> UploadedVideo:
        """
        소비자가 업로드한 파일을 영구 저장 위치로 복사하고 메타데이터를 생성한다.

        uploaded_file_path: 업로드 직후 임시 저장된 파일 경로 (프레임워크가 넘겨줌)
        original_filename: 사용자가 올린 원본 파일명
        """
        video_id = str(uuid.uuid4())
        ext = Path(original_filename).suffix or ".mp4"
        dest_path = self.storage_dir / f"{video_id}{ext}"

        shutil.copy(uploaded_file_path, dest_path)

        return UploadedVideo(
            video_id=video_id,
            original_filename=original_filename,
            saved_path=dest_path,
        )

    def handle_url_input(self, url: str) -> UploadedVideo:
        """
        URL이 들어온 경우 다운로드를 시도한다.
        (실제 다운로드 로직은 플랫폼별로 다르므로 추후 구현)
        """
        raise NotImplementedError(
            "URL 다운로드는 추후 구현 — 현재는 파일 업로드만 지원"
        )

    def validate(self, video: UploadedVideo) -> bool:
        """
        기본 검증: 파일 존재 여부, 확장자, 용량 등.
        """
        if not video.saved_path.exists():
            return False
        allowed_ext = {".mp4", ".mov", ".mkv", ".webm"}
        if video.saved_path.suffix.lower() not in allowed_ext:
            return False
        return True

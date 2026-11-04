"""주의 표현 사전. banned는 우선 검토 목록이며 단어 자체로 위법을 확정하지 않는다."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class WordEntry:
    word: str
    category: str          # 세부 카테고리 (예: "효능과장", "의학적 효능 오인" 등 — 팀 기준에 맞게)
    legal_basis: str = ""   # 근거 법조문 (예: "표시광고법 제3조") — 있으면 기록


@dataclass
class WordLists:
    banned: list[WordEntry]       # 우선 검토 표현 (기존 파일 호환을 위해 필드명 유지)
    ambiguous: list[WordEntry]    # 오인유발어 — AI 문장 분석 트리거

    @classmethod
    def load(cls, banned_path: str, ambiguous_path: str) -> "WordLists":
        """
        JSON 포맷 예시 (팀 단어 목록을 이 형태로 export):
        [
          {"word": "완치", "category": "효능과장", "legal_basis": "표시광고법 제3조"},
          ...
        ]
        """
        banned = cls._load_entries(banned_path)
        ambiguous = cls._load_entries(ambiguous_path)
        return cls(banned=banned, ambiguous=ambiguous)

    @staticmethod
    def _load_entries(path: str) -> list[WordEntry]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [
            WordEntry(
                word=item["word"],
                category=item.get("category", ""),
                legal_basis=item.get("legal_basis", ""),
            )
            for item in data
        ]

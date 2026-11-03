"""
단어 사전 스키마

팀에서 만드는 단어 모음 파일(csv/json 등)을 이 구조로 맞춰서 로드한다.
실제 단어 목록 자체는 다른 쪽에서 작업 중이므로, 여기서는 "포맷"만 정의.

금지어(banned): 법률상 광고에 아예 들어가면 안 되는 표현
  예: "FDA 승인", "100% 완치" 같은 절대적·허위 표현

오인유발어(ambiguous): 단어 자체는 문제 없지만 문맥에 따라
  소비자를 오인시킬 수 있는 표현 → AI 문장 분석으로 2차 판정 필요
  예: "최고", "유일한", "즉시 효과"
"""

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
    banned: list[WordEntry]       # 금지어 — 감지 즉시 부당광고 판정
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

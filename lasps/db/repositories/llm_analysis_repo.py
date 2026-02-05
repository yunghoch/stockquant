import datetime
from typing import List, Optional

from sqlalchemy import and_
from sqlalchemy.orm import Session

from lasps.db.models.llm_analysis import LlmAnalysis
from lasps.db.repositories.base_repository import BaseRepository


class LlmAnalysisRepository(BaseRepository[LlmAnalysis]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, LlmAnalysis)

    def get_by_stock_and_date(
        self, stock_code: str, date: datetime.date
    ) -> Optional[LlmAnalysis]:
        return (
            self.session.query(LlmAnalysis)
            .filter(
                and_(
                    LlmAnalysis.stock_code == stock_code,
                    LlmAnalysis.date == date,
                )
            )
            .first()
        )

    def get_by_date(self, date: datetime.date) -> List[LlmAnalysis]:
        return (
            self.session.query(LlmAnalysis)
            .filter(LlmAnalysis.date == date)
            .all()
        )

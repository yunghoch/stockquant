import datetime
from typing import List, Optional

from sqlalchemy import and_
from sqlalchemy.orm import Session

from lasps.db.models.prediction import Prediction
from lasps.db.repositories.base_repository import BaseRepository


class PredictionRepository(BaseRepository[Prediction]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, Prediction)

    def get_by_date(
        self,
        date: datetime.date,
        model_version: Optional[str] = None,
    ) -> List[Prediction]:
        q = self.session.query(Prediction).filter(Prediction.date == date)
        if model_version:
            q = q.filter(Prediction.model_version == model_version)
        return q.all()

    def get_by_stock_and_date(
        self,
        stock_code: str,
        date: datetime.date,
        model_version: str,
    ) -> Optional[Prediction]:
        return (
            self.session.query(Prediction)
            .filter(
                and_(
                    Prediction.stock_code == stock_code,
                    Prediction.date == date,
                    Prediction.model_version == model_version,
                )
            )
            .first()
        )

    def get_buy_signals(
        self,
        date: datetime.date,
        model_version: Optional[str] = None,
    ) -> List[Prediction]:
        q = self.session.query(Prediction).filter(
            and_(Prediction.date == date, Prediction.prediction == 2)
        )
        if model_version:
            q = q.filter(Prediction.model_version == model_version)
        return q.order_by(Prediction.confidence.desc()).all()

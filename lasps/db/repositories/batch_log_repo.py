import datetime
from typing import Optional

from sqlalchemy.orm import Session

from lasps.db.models.batch_log import BatchLog
from lasps.db.repositories.base_repository import BaseRepository


class BatchLogRepository(BaseRepository[BatchLog]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, BatchLog)

    def get_by_date(self, date: datetime.date) -> Optional[BatchLog]:
        return (
            self.session.query(BatchLog)
            .filter(BatchLog.date == date)
            .first()
        )

    def start(self, date: datetime.date, model_version: Optional[str] = None) -> BatchLog:
        log = BatchLog(
            date=date,
            status="running",
            started_at=datetime.datetime.utcnow(),
            model_version=model_version,
        )
        self.session.add(log)
        self.session.flush()
        return log

    def complete(self, log: BatchLog, stocks_predicted: int) -> BatchLog:
        log.status = "completed"
        log.completed_at = datetime.datetime.utcnow()
        log.stocks_predicted = stocks_predicted
        self.session.flush()
        return log

    def fail(self, log: BatchLog, error_message: str) -> BatchLog:
        log.status = "failed"
        log.completed_at = datetime.datetime.utcnow()
        log.error_message = error_message
        self.session.flush()
        return log

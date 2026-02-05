"""Initial schema — 13 tables

Revision ID: 001
Revises: None
Create Date: 2026-02-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. sectors
    op.create_table(
        "sectors",
        sa.Column("id", sa.SmallInteger, primary_key=True, autoincrement=False),
        sa.Column("code", sa.String(3), unique=True, nullable=False),
        sa.Column("name", sa.String(30), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # 2. stocks
    op.create_table(
        "stocks",
        sa.Column("code", sa.String(6), primary_key=True),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("sector_id", sa.SmallInteger, sa.ForeignKey("sectors.id"), nullable=True),
        sa.Column("sector_code", sa.String(3), nullable=True),
        sa.Column("market_cap", sa.BigInteger, nullable=True),
        sa.Column("per", sa.Numeric(8, 2), nullable=True),
        sa.Column("pbr", sa.Numeric(8, 3), nullable=True),
        sa.Column("roe", sa.Numeric(8, 2), nullable=True),
        sa.Column("debt_ratio", sa.Numeric(8, 2), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # 3. batch_logs (predictions가 FK 참조하므로 먼저 생성)
    op.create_table(
        "batch_logs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("date", sa.Date, unique=True, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("started_at", sa.DateTime, nullable=False),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("stocks_predicted", sa.Integer, nullable=True),
        sa.Column("model_version", sa.String(30), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
    )

    # 4. daily_prices
    op.create_table(
        "daily_prices",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_code", sa.String(6), sa.ForeignKey("stocks.code"), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("open", sa.Integer, nullable=False),
        sa.Column("high", sa.Integer, nullable=False),
        sa.Column("low", sa.Integer, nullable=False),
        sa.Column("close", sa.Integer, nullable=False),
        sa.Column("volume", sa.BigInteger, nullable=False),
        sa.UniqueConstraint("stock_code", "date", name="uq_daily_prices_stock_date"),
    )

    # 5. investor_trading
    op.create_table(
        "investor_trading",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_code", sa.String(6), sa.ForeignKey("stocks.code"), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("foreign_net", sa.BigInteger, nullable=False),
        sa.Column("inst_net", sa.BigInteger, nullable=False),
        sa.Column("individual_net", sa.BigInteger, nullable=True),
        sa.UniqueConstraint("stock_code", "date", name="uq_investor_trading_stock_date"),
    )

    # 6. short_selling
    op.create_table(
        "short_selling",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_code", sa.String(6), sa.ForeignKey("stocks.code"), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("short_volume", sa.BigInteger, nullable=False),
        sa.Column("short_ratio", sa.Numeric(6, 3), nullable=True),
        sa.UniqueConstraint("stock_code", "date", name="uq_short_selling_stock_date"),
    )

    # 7. technical_indicators
    op.create_table(
        "technical_indicators",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_code", sa.String(6), sa.ForeignKey("stocks.code"), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("ma5", sa.Float, nullable=True),
        sa.Column("ma20", sa.Float, nullable=True),
        sa.Column("ma60", sa.Float, nullable=True),
        sa.Column("ma120", sa.Float, nullable=True),
        sa.Column("rsi", sa.Float, nullable=True),
        sa.Column("macd", sa.Float, nullable=True),
        sa.Column("macd_signal", sa.Float, nullable=True),
        sa.Column("macd_hist", sa.Float, nullable=True),
        sa.Column("bb_upper", sa.Float, nullable=True),
        sa.Column("bb_middle", sa.Float, nullable=True),
        sa.Column("bb_lower", sa.Float, nullable=True),
        sa.Column("bb_width", sa.Float, nullable=True),
        sa.Column("atr", sa.Float, nullable=True),
        sa.Column("obv", sa.Float, nullable=True),
        sa.Column("volume_ma20", sa.Float, nullable=True),
        sa.UniqueConstraint("stock_code", "date", name="uq_technical_indicators_stock_date"),
    )

    # 8. market_sentiment
    op.create_table(
        "market_sentiment",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_code", sa.String(6), sa.ForeignKey("stocks.code"), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("volume_ratio", sa.REAL, nullable=True),
        sa.Column("volatility_ratio", sa.REAL, nullable=True),
        sa.Column("gap_direction", sa.REAL, nullable=True),
        sa.Column("rsi_norm", sa.REAL, nullable=True),
        sa.Column("foreign_inst_flow", sa.REAL, nullable=True),
        sa.UniqueConstraint("stock_code", "date", name="uq_market_sentiment_stock_date"),
    )

    # 9. qvm_scores
    op.create_table(
        "qvm_scores",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_code", sa.String(6), sa.ForeignKey("stocks.code"), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("q_score", sa.REAL, nullable=True),
        sa.Column("v_score", sa.REAL, nullable=True),
        sa.Column("m_score", sa.REAL, nullable=True),
        sa.Column("qvm_score", sa.REAL, nullable=True),
        sa.Column("rank", sa.SmallInteger, nullable=True),
        sa.Column("selected", sa.Boolean, nullable=True),
        sa.UniqueConstraint("stock_code", "date", name="uq_qvm_scores_stock_date"),
    )

    # 10. predictions
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_code", sa.String(6), sa.ForeignKey("stocks.code"), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("prediction", sa.SmallInteger, nullable=False),
        sa.Column("label", sa.String(4), nullable=False),
        sa.Column("confidence", sa.REAL, nullable=True),
        sa.Column("prob_sell", sa.REAL, nullable=True),
        sa.Column("prob_hold", sa.REAL, nullable=True),
        sa.Column("prob_buy", sa.REAL, nullable=True),
        sa.Column("model_version", sa.String(30), nullable=False),
        sa.Column("sector_id", sa.SmallInteger, sa.ForeignKey("sectors.id"), nullable=True),
        sa.Column("batch_id", sa.BigInteger, sa.ForeignKey("batch_logs.id"), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint(
            "stock_code", "date", "model_version",
            name="uq_predictions_stock_date_version",
        ),
    )

    # 11. llm_analyses
    op.create_table(
        "llm_analyses",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_code", sa.String(6), sa.ForeignKey("stocks.code"), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("analysis_text", sa.Text, nullable=False),
        sa.Column("model_name", sa.String(50), nullable=True),
        sa.Column("prediction_id", sa.BigInteger, sa.ForeignKey("predictions.id"), nullable=True),
        sa.Column("batch_id", sa.BigInteger, sa.ForeignKey("batch_logs.id"), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("stock_code", "date", name="uq_llm_analyses_stock_date"),
    )

    # 12. model_checkpoints
    op.create_table(
        "model_checkpoints",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("version", sa.String(30), unique=True, nullable=False),
        sa.Column("phase", sa.SmallInteger, nullable=False),
        sa.Column("file_path", sa.String(255), nullable=False),
        sa.Column("val_loss", sa.REAL, nullable=True),
        sa.Column("accuracy", sa.REAL, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # 13. training_labels
    op.create_table(
        "training_labels",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_code", sa.String(6), sa.ForeignKey("stocks.code"), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("label", sa.SmallInteger, nullable=False),
        sa.Column("split", sa.String(5), nullable=False),
        sa.Column("sector_id", sa.SmallInteger, sa.ForeignKey("sectors.id"), nullable=True),
        sa.UniqueConstraint("stock_code", "date", name="uq_training_labels_stock_date"),
    )


def downgrade() -> None:
    op.drop_table("training_labels")
    op.drop_table("model_checkpoints")
    op.drop_table("llm_analyses")
    op.drop_table("predictions")
    op.drop_table("qvm_scores")
    op.drop_table("market_sentiment")
    op.drop_table("technical_indicators")
    op.drop_table("short_selling")
    op.drop_table("investor_trading")
    op.drop_table("daily_prices")
    op.drop_table("batch_logs")
    op.drop_table("stocks")
    op.drop_table("sectors")

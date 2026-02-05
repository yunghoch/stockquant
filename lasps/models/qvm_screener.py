"""Quality-Value-Momentum (QVM) Screener.

Screens stocks by composite QVM score combining quality metrics (ROE,
debt ratio), value metrics (PER, PBR), and momentum metrics (volume,
market cap) to select top candidates for prediction.
"""

import pandas as pd


class QVMScreener:
    """Quality-Value-Momentum screener for stock universe selection.

    Computes a composite QVM score from quality, value, and momentum
    sub-scores and returns the top-N ranked stocks.

    Scoring:
        - Q (Quality): high ROE + low debt ratio
        - V (Value): low PER + low PBR
        - M (Momentum): high volume + high market cap
    """

    def screen(
        self,
        stocks_df: pd.DataFrame,
        top_n: int = 50,
    ) -> pd.DataFrame:
        """Screen stocks and return top-N by QVM score.

        Args:
            stocks_df: DataFrame with columns: code, market_cap, per,
                pbr, roe, debt_ratio, volume_avg_20.
            top_n: Number of top stocks to return.

        Returns:
            DataFrame of top-N stocks with added score columns
            (q_score, v_score, m_score, qvm_score), reset index.
        """
        df = stocks_df.copy()

        # Quality: high ROE, low debt
        df["q_score"] = (
            df["roe"].rank(pct=True)
            + (1 - df["debt_ratio"].rank(pct=True))
        )

        # Value: low PER, low PBR
        df["v_score"] = (
            (1 - df["per"].rank(pct=True))
            + (1 - df["pbr"].rank(pct=True))
        )

        # Momentum: high volume, high market cap
        df["m_score"] = (
            df["volume_avg_20"].rank(pct=True)
            + df["market_cap"].rank(pct=True)
        )

        df["qvm_score"] = df["q_score"] + df["v_score"] + df["m_score"]

        return df.nlargest(top_n, "qvm_score").reset_index(drop=True)

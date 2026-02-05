# lasps/services/llm_analyst.py

"""LLM Analyst Service using Claude API.

Analyzes top-10 predicted stocks with Claude for detailed investment
insights. Monthly cost: ~$30 (1 call/day, 10 stocks).
"""

import anthropic
from typing import Dict, List
from loguru import logger


class LLMAnalyst:
    """Top 10 종목 Claude 분석기.

    모델 예측 결과 상위 10종목에 대해 Claude API로 상세 분석을 수행한다.
    월 비용: ~$30 (일 1회, 10종목)

    Args:
        api_key: Anthropic API key.
        model: Claude model name to use.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze_stock(self, stock_info: Dict) -> str:
        """Analyze a single stock using Claude.

        Args:
            stock_info: Dict with keys: name, sector_name, prediction_label,
                confidence, per, pbr, roe.

        Returns:
            Analysis text string from Claude.
        """
        prompt = (
            f"다음 종목을 분석하고 투자 의견을 제시해주세요.\n\n"
            f"종목명: {stock_info.get('name', '')}\n"
            f"업종: {stock_info.get('sector_name', '')}\n"
            f"모델 예측: {stock_info.get('prediction_label', '')}\n"
            f"신뢰도: {stock_info.get('confidence', 0):.1%}\n"
            f"주요 지표:\n"
            f"- PER: {stock_info.get('per', 'N/A')}\n"
            f"- PBR: {stock_info.get('pbr', 'N/A')}\n"
            f"- ROE: {stock_info.get('roe', 'N/A')}\n\n"
            f"300자 이내로 핵심 포인트만 요약해주세요."
        )

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return f"분석 실패: {e}"

    def analyze_top_stocks(self, stocks: List[Dict]) -> List[Dict]:
        """Analyze top 10 stocks from prediction results.

        Args:
            stocks: List of stock dicts, sorted by confidence descending.
                Each dict should contain name, sector_name, prediction_label,
                confidence, per, pbr, roe.

        Returns:
            List of stock dicts with added 'llm_analysis' key.
        """
        results = []
        for stock in stocks[:10]:
            analysis = self.analyze_stock(stock)
            results.append({**stock, "llm_analysis": analysis})
            logger.info(f"Analyzed: {stock.get('name', 'Unknown')}")
        return results

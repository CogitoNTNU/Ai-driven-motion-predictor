"""FinBERT-based sentiment analysis for market news."""

import asyncio
import logging
from functools import partial

logger = logging.getLogger(__name__)

_MODEL_NAME = "ProsusAI/finbert"


class FinBERTAnalyzer:
    """Lazy-loading FinBERT sentiment analyzer.

    The model is loaded from HuggingFace on first use. All inference runs in a
    thread-pool executor so the async event loop is never blocked.

    Example::

        analyzer = FinBERTAnalyzer()
        score = await analyzer.score(["Markets rally on strong jobs data."])
        # score is a float in [-1.0, 1.0]
    """

    def __init__(self) -> None:
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline

            logger.info("Loading FinBERT model '%s' — this may take a moment on first run.", _MODEL_NAME)
            self._pipeline = pipeline(
                "text-classification",
                model=_MODEL_NAME,
                top_k=None,
                truncation=True,
                max_length=512,
            )
        return self._pipeline

    def _score_sync(self, texts: list[str]) -> float:
        """Run FinBERT on *texts* and return the average sentiment score.

        Each article is scored as ``positive_prob - negative_prob``, and the
        results are averaged across all articles.

        Args:
            texts: List of article texts to score.

        Returns:
            Average sentiment in ``[-1.0, 1.0]``, or ``0.0`` if *texts* is empty.
        """
        if not texts:
            return 0.0
        pipe = self._load()
        scores: list[float] = []
        for result in pipe(texts, batch_size=8):
            label_scores = {item["label"]: item["score"] for item in result}
            scores.append(label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0))
        return sum(scores) / len(scores)

    async def score(self, texts: list[str]) -> float:
        """Async wrapper — scores *texts* in a thread-pool executor.

        Args:
            texts: List of news article texts (title + summary).

        Returns:
            Average FinBERT sentiment score in ``[-1.0, 1.0]``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._score_sync, texts))

    def _score_articles_sync(self, texts: list[str]) -> list[dict]:
        """Run FinBERT on each text, returning individual probability dicts.

        Args:
            texts: Article texts to score.

        Returns:
            List of dicts with keys ``positive``, ``negative``, ``neutral``,
            and ``net_score`` for each input text.
        """
        if not texts:
            return []
        pipe = self._load()
        results = []
        for result in pipe(texts, batch_size=8):
            label_scores = {item["label"]: item["score"] for item in result}
            pos = label_scores.get("positive", 0.0)
            neg = label_scores.get("negative", 0.0)
            neu = label_scores.get("neutral", 0.0)
            results.append({"positive": pos, "negative": neg, "neutral": neu, "net_score": pos - neg})
        return results

    async def score_articles(self, texts: list[str]) -> list[dict]:
        """Async wrapper — scores each text individually.

        Args:
            texts: Article texts to score.

        Returns:
            List of dicts with ``positive``, ``negative``, ``neutral``,
            ``net_score`` per article.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._score_articles_sync, texts))

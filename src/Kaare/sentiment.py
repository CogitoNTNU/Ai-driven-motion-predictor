"""FinBERT-based sentiment analysis for market news."""

import asyncio
import logging
import re
from functools import partial

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """Clean text by removing/replacing problematic Unicode characters.

    Args:
        text: Raw text that may contain special characters.

    Returns:
        Cleaned text with ASCII-compatible characters.
    """
    if not text:
        return ""

    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201c": '"',  # Left double quotation mark
        "\u201d": '"',  # Right double quotation mark
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2026": "...",  # Horizontal ellipsis
        "\u00a0": " ",  # Non-breaking space
        "\u00ad": "",  # Soft hyphen
        "\u200b": "",  # Zero-width space
        "\ufeff": "",  # Byte order mark
        "\xa8": "",  # Diaeresis/umlaut
        "\xa9": "(c)",  # Copyright symbol
        "\xae": "(R)",  # Registered trademark
        "\u2122": "(TM)",  # Trademark symbol
    }

    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)

    # Remove any remaining non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


_MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
_pipeline_instance = None


def _get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        import time
        from transformers import pipeline
        from Kaare import config

        logger.info(
            "Loading sentiment model '%s' on device='%s' — this may take a moment on first run.",
            _MODEL_NAME,
            config.DEVICE,
        )
        t0 = time.perf_counter()
        _pipeline_instance = pipeline(
            "text-classification",
            model=_MODEL_NAME,
            top_k=None,
            truncation=True,
            max_length=512,
            device=config.DEVICE,
        )
        logger.info("Sentiment model loaded in %.2fs", time.perf_counter() - t0)
    return _pipeline_instance


class FinBERTAnalyzer:
    """Lazy-loading FinBERT sentiment analyzer.

    The model is loaded from HuggingFace on first use. All inference runs in a
    thread-pool executor so the async event loop is never blocked.

    Example::

        analyzer = FinBERTAnalyzer()
        score = await analyzer.score(["Markets rally on strong jobs data."])
        # score is a float in [-1.0, 1.0]
    """

    def _load(self):
        return _get_pipeline()

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
        # Clean texts to remove problematic Unicode characters
        cleaned_texts = [_clean_text(t) for t in texts]
        cleaned_texts = [t for t in cleaned_texts if t]  # Remove empty strings
        if not cleaned_texts:
            return 0.0
        pipe = self._load()
        scores: list[float] = []
        for result in pipe(cleaned_texts, batch_size=8):
            label_scores = {item["label"]: item["score"] for item in result}
            scores.append(
                label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0)
            )
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
        # Clean texts to remove problematic Unicode characters
        cleaned_texts = [_clean_text(t) for t in texts]
        cleaned_texts = [t for t in cleaned_texts if t]  # Remove empty strings
        if not cleaned_texts:
            return []
        pipe = self._load()
        results = []
        for result in pipe(cleaned_texts, batch_size=8):
            label_scores = {item["label"]: item["score"] for item in result}
            pos = label_scores.get("positive", 0.0)
            neg = label_scores.get("negative", 0.0)
            neu = label_scores.get("neutral", 0.0)
            results.append(
                {
                    "positive": pos,
                    "negative": neg,
                    "neutral": neu,
                    "net_score": pos - neg,
                }
            )
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
        return await loop.run_in_executor(
            None, partial(self._score_articles_sync, texts)
        )

    def _score_one_sync(self, text: str) -> dict:
        import time

        pipe = self._load()
        t0 = time.perf_counter()
        result = pipe([text])[0]
        logger.debug("Inference for 1 article: %.3fs", time.perf_counter() - t0)
        label_scores = {item["label"]: item["score"] for item in result}
        pos = label_scores.get("positive", 0.0)
        neg = label_scores.get("negative", 0.0)
        neu = label_scores.get("neutral", 0.0)
        return {
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "net_score": pos - neg,
        }

    async def score_articles_stream(self, texts: list[str]):
        """Async generator that yields a score dict for each text as it completes.

        Runs one article at a time in the thread-pool executor so the caller can
        stream results back to the client without waiting for the full batch.

        Args:
            texts: Article texts to score.

        Yields:
            Dict with ``positive``, ``negative``, ``neutral``, ``net_score``.
        """
        loop = asyncio.get_running_loop()
        for text in texts:
            yield await loop.run_in_executor(None, partial(self._score_one_sync, text))

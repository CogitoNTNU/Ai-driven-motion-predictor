# FinBERT Daily Market Sentiment Pipeline — Claude Code Guide

## Goal

Build a Python pipeline that:

1. Downloads financial news from HuggingFace (`Brianferrell787/financial-news-multisource`)
2. Runs FinBERT sentiment analysis on each article
3. Aggregates into a **daily market sentiment score** (both market-wide and per-ticker)
4. Stores the results in a **PostgreSQL/ParadeDB** database for use as features in an LSTM stock prediction model

---

## Project Structure

```
finbert-sentiment/
├── .env                    # HF_TOKEN, DATABASE_URL
├── requirements.txt
├── config.py               # Central config (DB, subsets, batch sizes)
├── download_news.py        # Step 1: Stream + filter + store raw news
├── run_sentiment.py        # Step 2: Batch FinBERT inference on stored news
├── aggregate_daily.py      # Step 3: Compute daily sentiment features
├── schema.sql              # Database schema
└── README.md
```

---

## Step 0: Environment Setup

### 0.1 — Create project directory

```bash
mkdir finbert-sentiment && cd finbert-sentiment
python -m venv venv
source venv/bin/activate
```

### 0.2 — Install dependencies

Create `requirements.txt`:

```
datasets>=2.19
transformers>=4.40
torch>=2.0
pandas>=2.0
psycopg2-binary>=2.9
python-dotenv>=1.0
tqdm>=4.66
```

Then install:

```bash
pip install -r requirements.txt
```

### 0.3 — Create `.env` file

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
DATABASE_URL=postgresql://user:password@localhost:5432/stockpred
BATCH_SIZE=64
DEVICE=cuda
```

- Get the HF token from https://huggingface.co/settings/tokens
- You must first request access to the dataset at https://huggingface.co/datasets/Brianferrell787/financial-news-multisource
- Set DEVICE=cpu if no GPU available (will be much slower)

### 0.4 — Authenticate with HuggingFace

```bash
huggingface-cli login
# Paste your token when prompted
```

---

## Step 1: Database Schema

Create `schema.sql`:

```sql
-- Raw news articles (downloaded from HuggingFace)
CREATE TABLE IF NOT EXISTS raw_news (
    id BIGSERIAL PRIMARY KEY,
    date_utc TIMESTAMPTZ NOT NULL,
    trading_date DATE NOT NULL,           -- NYSE-aligned trading date
    text TEXT NOT NULL,
    text_hash CHAR(64) NOT NULL,          -- SHA256 for dedup
    dataset_subset TEXT NOT NULL,          -- e.g. 'fnspid_news'
    source TEXT,                           -- e.g. 'Bloomberg', 'CNBC'
    tickers TEXT[],                        -- array of ticker symbols if available
    extra_fields JSONB,                    -- full original extra_fields
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_news_dedup 
    ON raw_news (text_hash, trading_date);
CREATE INDEX IF NOT EXISTS idx_raw_news_trading_date 
    ON raw_news (trading_date);
CREATE INDEX IF NOT EXISTS idx_raw_news_tickers 
    ON raw_news USING GIN (tickers);

-- FinBERT sentiment scores per article
CREATE TABLE IF NOT EXISTS article_sentiment (
    id BIGSERIAL PRIMARY KEY,
    raw_news_id BIGINT NOT NULL REFERENCES raw_news(id),
    positive REAL NOT NULL,
    negative REAL NOT NULL,
    neutral REAL NOT NULL,
    net_score REAL NOT NULL,              -- positive - negative, range [-1, 1]
    model_version TEXT DEFAULT 'ProsusAI/finbert',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_article_sentiment_news_id 
    ON article_sentiment (raw_news_id);

-- Daily aggregated sentiment (market-wide)
CREATE TABLE IF NOT EXISTS daily_market_sentiment (
    trading_date DATE PRIMARY KEY,
    mean_score REAL NOT NULL,
    median_score REAL NOT NULL,
    std_score REAL,
    article_count INT NOT NULL,
    pct_positive REAL,                    -- fraction of articles with positive > 0.5
    pct_negative REAL,                    -- fraction of articles with negative > 0.5
    mean_score_3d REAL,                   -- 3-day rolling average
    mean_score_7d REAL,                   -- 7-day rolling average
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Daily aggregated sentiment (per ticker)
CREATE TABLE IF NOT EXISTS daily_ticker_sentiment (
    trading_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    mean_score REAL NOT NULL,
    article_count INT NOT NULL,
    std_score REAL,
    pct_positive REAL,
    pct_negative REAL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (trading_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_daily_ticker_sentiment_ticker 
    ON daily_ticker_sentiment (ticker, trading_date);
```

Run it against your database:

```bash
psql $DATABASE_URL -f schema.sql
```

---

## Step 2: Config

Create `config.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
DEVICE = os.getenv("DEVICE", "cpu")

# Finance-focused subsets (these have the best signal-to-noise for market sentiment)
# Ordered roughly by quality/relevance
FINANCE_SUBSETS = [
    "fnspid_news",                 # 1999-2023, ticker-tagged, large
    "benzinga_6000stocks",         # 2000s-2010s, ticker-tagged
    "bloomberg_reuters",           # 2006-2013, full articles, high quality
    "cnbc_headlines",              # 2006-2020, market-focused
    "sp500_daily_headlines",       # 2008-2024, market recaps
    "finsen_us_2007_2023",         # 2007-2023, financial news with categories
    "yahoo_finance_felixdrinkall", # 2017-2023, ticker-tagged
]

# Optional: broader subsets if you want general market sentiment too
BROAD_SUBSETS = [
    "nyt_headlines_1990_2020",
    "all_the_news_2",
    "headlines_10sites_2007_2022",
]

# FinBERT config
FINBERT_MODEL = "ProsusAI/finbert"
MAX_TOKEN_LENGTH = 512
```

---

## Step 3: Download & Store Raw News

Create `download_news.py`:

```python
"""
Download financial news from HuggingFace and store in PostgreSQL.
Streams data to avoid memory issues with 57M+ rows.
Uses batch inserts for performance.
"""

import json
import hashlib
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from datasets import load_dataset
from tqdm import tqdm

from config import DATABASE_URL, HF_TOKEN, FINANCE_SUBSETS


def parse_trading_date(row: dict) -> str:
    """
    Extract the NYSE-aligned trading date.
    Uses extra_fields.date_trading if available (already accounts for 
    weekends/holidays), otherwise falls back to the top-level date.
    """
    extras = json.loads(row["extra_fields"])
    trading_dt = extras.get("date_trading")
    if trading_dt:
        # date_trading is like "2020-01-02T14:30:00Z" — take just the date part
        return trading_dt[:10]
    # Fallback: use top-level date
    return row["date"][:10]


def parse_tickers(row: dict) -> list[str]:
    """Extract ticker symbols from extra_fields if present."""
    extras = json.loads(row["extra_fields"])
    tickers = extras.get("stocks", [])
    if isinstance(tickers, str):
        tickers = [tickers]
    return tickers


def text_hash(text: str) -> str:
    """SHA256 hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_subset_name(row: dict) -> str:
    """Extract subset identifier from extra_fields."""
    extras = json.loads(row["extra_fields"])
    return extras.get("dataset", "unknown")


def get_source(row: dict) -> Optional[str]:
    """Extract source/publisher from extra_fields."""
    extras = json.loads(row["extra_fields"])
    return extras.get("source") or extras.get("publisher")


def download_and_store(subsets: list[str], batch_size: int = 1000):
    """
    Stream subsets from HuggingFace and batch-insert into PostgreSQL.
    Uses ON CONFLICT to skip duplicates.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    data_files = [f"data/{s}/*.parquet" for s in subsets]

    print(f"Loading {len(subsets)} subsets: {subsets}")
    ds = load_dataset(
        "Brianferrell787/financial-news-multisource",
        data_files=data_files,
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    batch = []
    total_inserted = 0
    total_skipped = 0

    for row in tqdm(ds, desc="Downloading news"):
        text = row["text"]

        # Skip empty or very short texts (noise)
        if not text or len(text.strip()) < 20:
            total_skipped += 1
            continue

        trading_date = parse_trading_date(row)
        tickers = parse_tickers(row)
        subset = get_subset_name(row)
        source = get_source(row)
        thash = text_hash(text)

        batch.append((
            row["date"],           # date_utc
            trading_date,          # trading_date
            text,                  # text
            thash,                 # text_hash
            subset,                # dataset_subset
            source,                # source
            tickers,               # tickers
            row["extra_fields"],   # extra_fields (raw JSON string)
        ))

        if len(batch) >= batch_size:
            inserted = flush_batch(cur, batch)
            total_inserted += inserted
            total_skipped += len(batch) - inserted
            batch = []
            conn.commit()

            if total_inserted % 50000 == 0:
                print(f"  Inserted: {total_inserted:,} | Skipped (dedup): {total_skipped:,}")

    # Final batch
    if batch:
        inserted = flush_batch(cur, batch)
        total_inserted += inserted
        conn.commit()

    cur.close()
    conn.close()
    print(f"\nDone. Total inserted: {total_inserted:,} | Total skipped: {total_skipped:,}")


def flush_batch(cur, batch: list) -> int:
    """Insert a batch of rows, skipping duplicates. Returns count inserted."""
    sql = """
        INSERT INTO raw_news 
            (date_utc, trading_date, text, text_hash, dataset_subset, source, tickers, extra_fields)
        VALUES %s
        ON CONFLICT (text_hash, trading_date) DO NOTHING
    """
    # psycopg2 needs the array cast for tickers
    template = "(%s, %s, %s, %s, %s, %s, %s::text[], %s::jsonb)"

    before = cur.rowcount
    execute_values(cur, sql, batch, template=template)
    return cur.rowcount if cur.rowcount > 0 else 0


if __name__ == "__main__":
    download_and_store(FINANCE_SUBSETS)
```

### Running Step 3

```bash
python download_news.py
```

**Expected time:** This will take several hours depending on your connection. The finance-focused subsets are roughly 5-15M rows combined. You can start with a single subset for testing:

```python
# Quick test with just S&P 500 headlines (~50k rows)
download_and_store(["sp500_daily_headlines"])
```

---

## Step 4: FinBERT Sentiment Scoring

Create `run_sentiment.py`:

```python
"""
Run FinBERT inference on all unscored articles in the database.
Processes in batches for GPU efficiency.
Truncates long articles to headline + first paragraph for better results.
"""

import torch
import psycopg2
from psycopg2.extras import execute_values
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from config import DATABASE_URL, FINBERT_MODEL, MAX_TOKEN_LENGTH, BATCH_SIZE, DEVICE


def load_model():
    """Load FinBERT model and tokenizer."""
    print(f"Loading {FINBERT_MODEL} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")
    return tokenizer, model


def smart_truncate(text: str, max_chars: int = 1500) -> str:
    """
    Truncate intelligently: keep headline + first paragraph.
    FinBERT's max is 512 tokens (~380 words). We pre-truncate by chars
    to avoid tokenizer overhead on very long articles.
    
    The dataset uses 'title\\n\\nbody' format, so we split on double newline.
    """
    if len(text) <= max_chars:
        return text

    parts = text.split("\n\n", 2)
    if len(parts) >= 2:
        # headline + first paragraph
        truncated = parts[0] + "\n\n" + parts[1]
        return truncated[:max_chars]
    return text[:max_chars]


def score_batch(texts: list[str], tokenizer, model) -> list[dict]:
    """
    Run FinBERT on a batch of texts.
    Returns list of {positive, negative, neutral, net_score}.
    FinBERT output order: [positive, negative, neutral]
    """
    truncated = [smart_truncate(t) for t in texts]

    inputs = tokenizer(
        truncated,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_TOKEN_LENGTH,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    results = []
    for p in probs:
        results.append({
            "positive": float(p[0]),
            "negative": float(p[1]),
            "neutral": float(p[2]),
            "net_score": float(p[0] - p[1]),
        })
    return results


def get_unscored_articles(cur, batch_size: int) -> list[tuple]:
    """
    Fetch articles that haven't been scored yet.
    Returns list of (id, text).
    """
    cur.execute("""
        SELECT rn.id, rn.text
        FROM raw_news rn
        LEFT JOIN article_sentiment s ON s.raw_news_id = rn.id
        WHERE s.id IS NULL
        ORDER BY rn.id
        LIMIT %s
    """, (batch_size,))
    return cur.fetchall()


def run_sentiment_scoring():
    """Main loop: fetch unscored articles, score them, store results."""
    tokenizer, model = load_model()

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Count total unscored
    cur.execute("""
        SELECT COUNT(*) FROM raw_news rn
        LEFT JOIN article_sentiment s ON s.raw_news_id = rn.id
        WHERE s.id IS NULL
    """)
    total_unscored = cur.fetchone()[0]
    print(f"Total unscored articles: {total_unscored:,}")

    if total_unscored == 0:
        print("Nothing to score.")
        return

    processed = 0
    pbar = tqdm(total=total_unscored, desc="Scoring articles")

    while True:
        # Fetch a chunk of unscored articles
        # Use a larger fetch to reduce DB round-trips, process in smaller GPU batches
        rows = get_unscored_articles(cur, batch_size=BATCH_SIZE * 10)
        if not rows:
            break

        # Process in GPU-sized batches
        for i in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[i : i + BATCH_SIZE]
            ids = [r[0] for r in batch_rows]
            texts = [r[1] for r in batch_rows]

            try:
                scores = score_batch(texts, tokenizer, model)
            except Exception as e:
                print(f"\nError scoring batch starting at id={ids[0]}: {e}")
                # Score individually to find the problematic article
                for row_id, text in zip(ids, texts):
                    try:
                        single = score_batch([text], tokenizer, model)
                        insert_scores(cur, [(row_id, single[0])])
                    except Exception as inner_e:
                        print(f"  Skipping article {row_id}: {inner_e}")
                conn.commit()
                pbar.update(len(batch_rows))
                processed += len(batch_rows)
                continue

            # Batch insert scores
            insert_data = list(zip(ids, scores))
            insert_scores(cur, insert_data)
            conn.commit()

            processed += len(batch_rows)
            pbar.update(len(batch_rows))

    pbar.close()
    cur.close()
    conn.close()
    print(f"\nDone. Scored {processed:,} articles.")


def insert_scores(cur, data: list[tuple]):
    """
    Batch insert sentiment scores.
    data is list of (raw_news_id, score_dict).
    """
    values = [
        (row_id, s["positive"], s["negative"], s["neutral"], s["net_score"])
        for row_id, s in data
    ]
    execute_values(
        cur,
        """
        INSERT INTO article_sentiment (raw_news_id, positive, negative, neutral, net_score)
        VALUES %s
        ON CONFLICT (raw_news_id) DO NOTHING
        """,
        values,
    )


if __name__ == "__main__":
    run_sentiment_scoring()
```

### Running Step 4

```bash
# On GPU (Runpod/Colab/NTNU HPC):
DEVICE=cuda python run_sentiment.py

# On CPU (much slower, ~10-20 articles/sec):
DEVICE=cpu python run_sentiment.py
```

**Performance estimates:**
- GPU (T4/A10): ~100-200 articles/sec → 5M articles in ~7-14 hours
- CPU: ~10-20 articles/sec → 5M articles in ~3-6 days

**Tip:** If running on a remote server, use `nohup` or `tmux`:

```bash
tmux new -s sentiment
DEVICE=cuda python run_sentiment.py
# Ctrl+B, D to detach — it keeps running
```

---

## Step 5: Aggregate Daily Sentiment

Create `aggregate_daily.py`:

```python
"""
Aggregate article-level sentiment into daily market sentiment features.
Produces both market-wide and per-ticker daily scores.
Computes rolling averages for use as LSTM features.
"""

import psycopg2
from config import DATABASE_URL


def aggregate_market_wide(cur):
    """
    Compute daily market-wide sentiment from all scored articles.
    Includes rolling 3-day and 7-day averages.
    """
    print("Aggregating market-wide daily sentiment...")

    cur.execute("""
        INSERT INTO daily_market_sentiment 
            (trading_date, mean_score, median_score, std_score, 
             article_count, pct_positive, pct_negative)
        SELECT 
            rn.trading_date,
            AVG(s.net_score) AS mean_score,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.net_score) AS median_score,
            STDDEV(s.net_score) AS std_score,
            COUNT(*) AS article_count,
            AVG(CASE WHEN s.positive > 0.5 THEN 1.0 ELSE 0.0 END) AS pct_positive,
            AVG(CASE WHEN s.negative > 0.5 THEN 1.0 ELSE 0.0 END) AS pct_negative
        FROM raw_news rn
        JOIN article_sentiment s ON s.raw_news_id = rn.id
        GROUP BY rn.trading_date
        ON CONFLICT (trading_date) DO UPDATE SET
            mean_score = EXCLUDED.mean_score,
            median_score = EXCLUDED.median_score,
            std_score = EXCLUDED.std_score,
            article_count = EXCLUDED.article_count,
            pct_positive = EXCLUDED.pct_positive,
            pct_negative = EXCLUDED.pct_negative,
            updated_at = NOW()
    """)
    print(f"  Upserted {cur.rowcount} daily market rows.")

    # Compute rolling averages
    print("Computing rolling averages...")
    cur.execute("""
        UPDATE daily_market_sentiment d SET
            mean_score_3d = sub.avg_3d,
            mean_score_7d = sub.avg_7d
        FROM (
            SELECT 
                trading_date,
                AVG(mean_score) OVER (
                    ORDER BY trading_date 
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS avg_3d,
                AVG(mean_score) OVER (
                    ORDER BY trading_date 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS avg_7d
            FROM daily_market_sentiment
        ) sub
        WHERE d.trading_date = sub.trading_date
    """)
    print("  Rolling averages updated.")


def aggregate_per_ticker(cur):
    """
    Compute daily per-ticker sentiment.
    Only for articles that have ticker tags.
    """
    print("Aggregating per-ticker daily sentiment...")

    cur.execute("""
        INSERT INTO daily_ticker_sentiment
            (trading_date, ticker, mean_score, article_count, 
             std_score, pct_positive, pct_negative)
        SELECT 
            rn.trading_date,
            UNNEST(rn.tickers) AS ticker,
            AVG(s.net_score) AS mean_score,
            COUNT(*) AS article_count,
            STDDEV(s.net_score) AS std_score,
            AVG(CASE WHEN s.positive > 0.5 THEN 1.0 ELSE 0.0 END) AS pct_positive,
            AVG(CASE WHEN s.negative > 0.5 THEN 1.0 ELSE 0.0 END) AS pct_negative
        FROM raw_news rn
        JOIN article_sentiment s ON s.raw_news_id = rn.id
        WHERE rn.tickers IS NOT NULL AND array_length(rn.tickers, 1) > 0
        GROUP BY rn.trading_date, UNNEST(rn.tickers)
        ON CONFLICT (trading_date, ticker) DO UPDATE SET
            mean_score = EXCLUDED.mean_score,
            article_count = EXCLUDED.article_count,
            std_score = EXCLUDED.std_score,
            pct_positive = EXCLUDED.pct_positive,
            pct_negative = EXCLUDED.pct_negative,
            updated_at = NOW()
    """)
    print(f"  Upserted {cur.rowcount} daily ticker rows.")


def print_summary(cur):
    """Print a quick summary of what we have."""
    cur.execute("SELECT MIN(trading_date), MAX(trading_date), COUNT(*) FROM daily_market_sentiment")
    row = cur.fetchone()
    print(f"\nMarket sentiment: {row[0]} to {row[1]} ({row[2]} trading days)")

    cur.execute("SELECT COUNT(DISTINCT ticker) FROM daily_ticker_sentiment")
    n_tickers = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM daily_ticker_sentiment")
    n_rows = cur.fetchone()[0]
    print(f"Ticker sentiment: {n_tickers} unique tickers, {n_rows} ticker-day rows")

    # Show sample
    print("\nSample daily market sentiment (last 5 days):")
    cur.execute("""
        SELECT trading_date, mean_score, article_count, pct_positive, pct_negative, mean_score_7d
        FROM daily_market_sentiment
        ORDER BY trading_date DESC
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"  {row[0]} | score={row[1]:.4f} | n={row[2]} | "
              f"+%={row[3]:.2f} | -%={row[4]:.2f} | 7d_avg={row[5]:.4f if row[5] else 'N/A'}")


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    aggregate_market_wide(cur)
    aggregate_per_ticker(cur)
    conn.commit()

    print_summary(cur)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
```

### Running Step 5

```bash
python aggregate_daily.py
```

---

## Step 6: Query Sentiment for LSTM Features

Once everything is populated, you can pull features for your model like this:

```python
"""Example: pull daily features for AAPL to feed into LSTM."""
import pandas as pd
import psycopg2
from config import DATABASE_URL

conn = psycopg2.connect(DATABASE_URL)

# Market-wide sentiment + AAPL-specific sentiment joined together
query = """
SELECT 
    m.trading_date,
    m.mean_score AS market_sentiment,
    m.std_score AS market_sentiment_std,
    m.article_count AS market_article_count,
    m.mean_score_3d AS market_sentiment_3d,
    m.mean_score_7d AS market_sentiment_7d,
    m.pct_positive AS market_pct_positive,
    m.pct_negative AS market_pct_negative,
    t.mean_score AS ticker_sentiment,
    t.article_count AS ticker_article_count,
    t.std_score AS ticker_sentiment_std
FROM daily_market_sentiment m
LEFT JOIN daily_ticker_sentiment t 
    ON t.trading_date = m.trading_date AND t.ticker = %s
WHERE m.trading_date BETWEEN %s AND %s
ORDER BY m.trading_date
"""

df = pd.read_sql(query, conn, params=["AAPL", "2010-01-01", "2024-01-01"])
conn.close()

# Fill NaN ticker sentiment with market sentiment (days with no AAPL articles)
df["ticker_sentiment"] = df["ticker_sentiment"].fillna(df["market_sentiment"])
df["ticker_article_count"] = df["ticker_article_count"].fillna(0)

print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"Date range: {df['trading_date'].min()} to {df['trading_date'].max()}")
```

---

## Execution Order Summary

```bash
# 1. Set up database
psql $DATABASE_URL -f schema.sql

# 2. Download news (hours, run in tmux)
python download_news.py

# 3. Score with FinBERT (hours-days depending on GPU, run in tmux)
python run_sentiment.py

# 4. Aggregate daily
python aggregate_daily.py

# 5. Use the data in your LSTM pipeline
python your_lstm_model.py  # pull from daily_market_sentiment / daily_ticker_sentiment
```

---

## Notes & Tips

- **Start small:** Test the full pipeline with just `sp500_daily_headlines` first (~50k rows). Once it works end-to-end, add the larger subsets.
- **GPU options:** Google Colab (free T4), Runpod ($0.20-0.40/hr for A10), or NTNU HPC if you have access. Running FinBERT on CPU is feasible but slow.
- **Incremental runs:** All scripts use `ON CONFLICT` / dedup logic, so you can re-run them safely without duplicating data.
- **Look-ahead bias:** The `date_trading` field from the dataset already maps to the *next* NYSE trading session. This means the sentiment for a given `trading_date` represents news known *before* that session opens — exactly what you want for prediction.
- **Feature ideas for LSTM beyond mean_score:**
  - `article_count` — abnormally high volume often precedes big moves
  - `std_score` — disagreement/dispersion signals uncertainty
  - `pct_positive - pct_negative` — alternative to mean
  - Rolling averages (3d, 7d) — momentum of sentiment
  - Sentiment delta (today vs yesterday) — change detection
- **Cross-subset dedup:** The dataset may have overlapping articles across subsets (e.g., same Reuters article in `bloomberg_reuters` and `fnspid_news`). The `text_hash + trading_date` unique index handles this.
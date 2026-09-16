"""Financial-news sentiment utilities using a finance-domain transformer."""

import pandas as pd
from transformers import pipeline


def build_sentiment_model(model_name="ProsusAI/finbert"):
    """Create the FinBERT text-classification pipeline."""
    return pipeline("text-classification", model=model_name, tokenizer=model_name)


def score_headlines(headlines: pd.Series, sentiment_pipeline) -> pd.DataFrame:
    """Score a collection of financial headlines."""
    results = sentiment_pipeline(headlines.fillna("").tolist(), truncation=True)
    return pd.DataFrame(results)

import pandas as pd
import os
from datetime import datetime

from .danmaku_io import load_danmaku_prefer_parquet
from .sentiment_bilibili import score_sentiment, score_anxiety

BASE_DIR = "D:/Claude_Code/bilibili-ai-tag-analysis"
OUTPUT_PATH = os.path.join(BASE_DIR, "data/processed/step5/shared/danmaku_daily_metrics.parquet")

def aggregate_daily():
    print("[daily_aggregator] Loading danmaku data...")
    df = load_danmaku_prefer_parquet()
    if df is None or df.empty:
        raise ValueError("No danmaku data loaded.")
    print(f"[daily_aggregator] Loaded {len(df)} rows.")

    # Ensure ctime is numeric seconds
    df["ctime"] = pd.to_numeric(df["ctime"], errors="coerce")
    df = df.dropna(subset=["ctime"])
    df["date"] = pd.to_datetime(df["ctime"], unit="s").dt.date

    # Content column
    content_col = "content"
    if content_col not in df.columns:
        raise ValueError(f"Column {content_col} not found in danmaku data.")

    print("[daily_aggregator] Scoring sentiment and anxiety...")
    df["sentiment"] = df[content_col].apply(score_sentiment)
    df["anxiety"] = df[content_col].apply(score_anxiety)

    print("[daily_aggregator] Aggregating by date...")
    daily = df.groupby("date").agg(
        total_danmaku=("bvid", "size"),
        unique_videos=("bvid", "nunique"),
        mean_sentiment=("sentiment", "mean"),
        mean_anxiety=("anxiety", "mean"),
        std_sentiment=("sentiment", "std"),
        std_anxiety=("anxiety", "std"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    daily.to_parquet(OUTPUT_PATH, index=False)
    print(f"[daily_aggregator] Saved to {OUTPUT_PATH} ({len(daily)} days)")
    return daily

if __name__ == "__main__":
    aggregate_daily()

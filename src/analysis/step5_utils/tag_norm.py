import pandas as pd
import numpy as np
import os

BASE_DIR = "D:/Claude_Code/bilibili-ai-tag-analysis"
VIDEO_CSV = os.path.join(BASE_DIR, "data/cleaned-data/bilibili_video_cleaned.csv")

def parse_tags(tag_str):
    if pd.isna(tag_str) or not isinstance(tag_str, str):
        return []
    for sep in ["|", ","]:
        if sep in tag_str:
            return [t.strip() for t in tag_str.split(sep) if t.strip()]
    return [tag_str.strip()] if tag_str.strip() else []

def load_video_df():
    df = pd.read_csv(VIDEO_CSV)
    df["tag_list"] = df["tags"].apply(parse_tags)
    df["pubdate_dt"] = pd.to_datetime(df["pubdate"], errors="coerce")
    df["year_quarter"] = df["pubdate_dt"].dt.to_period("Q").astype(str)
    return df

def add_relative_performance(df, fallback_median=1.0):
    group_medians = df.groupby(["search_tag", "year_quarter"])["stat_view"].transform("median")
    df["relative_performance"] = df["stat_view"] / group_medians.replace(0, np.nan)
    df["relative_performance"] = df["relative_performance"].fillna(df["stat_view"] / fallback_median)
    return df

def add_quota_weights(df):
    counts = df.groupby(["search_tag", "year_quarter"]).size().reset_index(name="count")
    df = df.merge(counts, on=["search_tag", "year_quarter"], how="left")
    df["quota_weight"] = 1.0 / df["count"]
    mean_w = df["quota_weight"].mean()
    df["quota_weight"] = df["quota_weight"] / mean_w
    return df

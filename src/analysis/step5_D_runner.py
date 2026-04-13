import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import ttest_ind

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.step5_utils.tag_norm import load_video_df
from analysis.step5_utils.sentiment_bilibili import score_sentiment
from analysis.step5_utils.plot_style import setup_fonts, savefig_safe

BASE_DIR = "D:/Claude_Code/bilibili-ai-tag-analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/step5/step5_D")
REPORT_PATH = os.path.join(BASE_DIR, "results/step5/step5_D_timeline_emotion_report.md")
FIG_DIR = os.path.join(BASE_DIR, "results/step5/figures")

def simple_dtw_distance(s1, s2):
    # Local DTW with small window to avoid O(n^2) blowup
    n, m = len(s1), len(s2)
    window = max(abs(n - m), 3)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw[n, m]

def run():
    setup_fonts()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    video_df = load_video_df()
    video_df = video_df[(video_df["stat_danmaku"] >= 500) & (video_df["duration"] > 0)].copy()
    video_df = video_df.dropna(subset=["bvid", "duration", "search_tag"])
    # Subsample to max 500 videos to control runtime
    MAX_V = 500
    if len(video_df) > MAX_V:
        video_df = video_df.sample(n=MAX_V, random_state=42).reset_index(drop=True)
    print(f"[step5_D] {len(video_df)} videos selected (after subsample)")

    dmk_path = os.path.join(BASE_DIR, "data/processed/step5/shared/danmaku_core.parquet")
    dmk = pd.read_parquet(dmk_path, columns=["bvid", "content", "progress"])
    dmk = dmk.merge(video_df[["bvid", "duration", "search_tag"]], on="bvid", how="inner")
    dmk["progress_norm"] = dmk["progress"] / dmk["duration"] * 100.0
    dmk["progress_norm"] = dmk["progress_norm"].clip(0, 100)
    dmk["sentiment"] = dmk["content"].astype(str).apply(score_sentiment)

    # 50 bins: 0-2%, 2-4%, ..., 98-100%
    bins = np.arange(0, 101, 2)
    bin_labels = np.arange(1, 51)
    dmk["bin"] = pd.cut(dmk["progress_norm"], bins=bins, labels=bin_labels, include_lowest=True)

    # Aggregate per video per bin
    grouped = dmk.groupby(["bvid", "bin"]).agg(
        mean_sentiment=("sentiment", "mean"),
        density=("content", "size"),
    ).reset_index()

    curve_records = []
    peak_records = []
    valid_bvids = []
    all_curves = {}

    for bvid in video_df["bvid"]:
        sub = grouped[grouped["bvid"] == bvid].sort_values("bin")
        if len(sub) < 20:
            continue
        valid_bvids.append(bvid)
        # Reindex to 50 bins
        full = pd.DataFrame({"bin": bin_labels})
        full = full.merge(sub, on="bin", how="left")
        full["mean_sentiment"] = full["mean_sentiment"].fillna(0)
        full["density"] = full["density"].fillna(0)

        s_arr = full["mean_sentiment"].values
        d_arr = full["density"].values
        all_curves[bvid] = s_arr

        # Find peaks
        s_peaks, _ = find_peaks(s_arr, height=0)
        d_peaks, _ = find_peaks(d_arr, height=np.median(d_arr))

        peak_records.append({
            "bvid": bvid,
            "sentiment_peak_bins": ",".join(str(p) for p in s_peaks),
            "density_peak_bins": ",".join(str(p) for p in d_peaks),
            "n_sentiment_peaks": len(s_peaks),
            "n_density_peaks": len(d_peaks),
        })

        # Save curve data
        for i, (sent, dens) in enumerate(zip(s_arr, d_arr)):
            curve_records.append({"bvid": bvid, "bin": i + 1, "mean_sentiment": sent, "density": dens})

    print(f"[step5_D] {len(valid_bvids)} videos with sufficient bins")

    if len(valid_bvids) == 0:
        raise ValueError("No valid videos for emotion timeline analysis.")

    # DTW comparison within same search_tag vs random
    tag_groups = {}
    for bvid in valid_bvids:
        tag = video_df[video_df["bvid"] == bvid]["search_tag"].values[0]
        tag_groups.setdefault(tag, []).append(bvid)

    intra_dists = []
    random_dists = []
    np.random.seed(42)
    for tag, bvids in tag_groups.items():
        if len(bvids) < 3:
            continue
        # intra-tag pairs
        for i in range(min(len(bvids) - 1, 20)):
            b1, b2 = bvids[i], bvids[i + 1]
            intra_dists.append(simple_dtw_distance(all_curves[b1], all_curves[b2]))
        # random pairs
        for _ in range(min(len(bvids), 20)):
            b1 = np.random.choice(bvids)
            b2 = np.random.choice([b for b in valid_bvids if b not in bvids])
            random_dists.append(simple_dtw_distance(all_curves[b1], all_curves[b2]))

    if intra_dists and random_dists:
        t_stat, p_dtw = ttest_ind(intra_dists, random_dists, equal_var=False)
        median_intra = np.median(intra_dists)
        median_random = np.median(random_dists)
    else:
        t_stat = p_dtw = median_intra = median_random = np.nan

    # Save outputs
    pd.DataFrame(curve_records).to_parquet(os.path.join(OUTPUT_DIR, "emotion_curves.parquet"), index=False)
    pd.DataFrame(peak_records).to_parquet(os.path.join(OUTPUT_DIR, "peak_records.parquet"), index=False)
    pd.DataFrame({
        "metric": ["median_intra_dtw", "median_random_dtw", "t_stat", "p_value"],
        "value": [median_intra, median_random, t_stat, p_dtw]
    }).to_parquet(os.path.join(OUTPUT_DIR, "dtw_summary.parquet"), index=False)

    # Visualize overall mean curve
    curve_df = pd.DataFrame(curve_records)
    mean_curve = curve_df.groupby("bin").agg(mean_sentiment=("mean_sentiment", "mean"), density=("density", "mean")).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "steelblue"
    ax1.plot(mean_curve["bin"], mean_curve["mean_sentiment"], color=color1, label="平均情感")
    ax1.set_xlabel("视频进度 bin (2% 为一个 bin)")
    ax1.set_ylabel("平均情感", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.plot(mean_curve["bin"], mean_curve["density"], color=color2, label="弹幕密度")
    ax2.set_ylabel("平均弹幕密度", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Peaks on mean curve
    s_peaks, _ = find_peaks(mean_curve["mean_sentiment"].fillna(0), height=0)
    d_peaks, _ = find_peaks(mean_curve["density"].fillna(0))
    for sp in s_peaks[:3]:
        ax1.axvline(mean_curve["bin"].iloc[sp], color=color1, linestyle="--", alpha=0.5)
    for dp in d_peaks[:3]:
        ax2.axvline(mean_curve["bin"].iloc[dp], color=color2, linestyle="--", alpha=0.5)

    fig.tight_layout()
    savefig_safe(fig, os.path.join(FIG_DIR, "step5_D_mean_emotion_curve.png"))

    # Branch determination
    if not np.isnan(p_dtw) and p_dtw < 0.05 and median_intra < median_random:
        branch = "A"
        conclusion = f"同标签视频的情绪曲线 DTW 距离显著小于随机配对（median {median_intra:.3f} vs {median_random:.3f}, p={p_dtw:.4f}），存在“黄金节奏”。"
    else:
        branch = "B"
        conclusion = f"DTW 距离无显著差异（median {median_intra:.3f} vs {median_random:.3f}, p={p_dtw:.4f}），不存在通用情绪节奏模板。"

    lines = [
        "# 子任务 D：弹幕情绪时间线与视频结构的映射",
        "",
        "## 数据说明",
        f"- 使用 `danmaku_core.parquet` 与视频数据。",
        f"- 筛选 `stat_danmaku >= 500` 且 `duration > 0` 的视频，共 {len(video_df)} 条。",
        f"- 最终满足有效 bin 覆盖（≥20 bins）的视频：{len(valid_bvids)} 条。",
        "",
        "## 数据处理",
        "- 将 `progress` 按视频时长归一化为 0–100%。",
        "- 每 2% 划分为一个 bin，共 50 bins。",
        "- 使用自定义情感词表对每个弹幕进行情感打分，计算每个 bin 的平均情感与弹幕密度。",
        "- 使用 `scipy.signal.find_peaks` 检测情感和密度的局部峰值。",
        "- 对同 `search_tag` 的视频对计算 DTW 距离，并与跨标签随机配对比较（Welch t-test）。",
        "",
        "## 数据分析",
        f"- 中位数 DTW 距离（同标签内）：{median_intra:.3f}" if not np.isnan(median_intra) else "- 中位数 DTW 距离（同标签内）：N/A",
        f"- 中位数 DTW 距离（随机跨标签）：{median_random:.3f}" if not np.isnan(median_random) else "- 中位数 DTW 距离（随机跨标签）：N/A",
        f"- Welch t-test：t={t_stat:.2f}, p={p_dtw:.4f}" if not np.isnan(p_dtw) else "- Welch t-test：N/A",
        "",
        "## 验证",
        "- 50 bins/视频的曲线数据、峰值记录、DTW 汇总已保存。",
        "- 平均情感-密度双轴图已保存。",
        "",
        "## 结论",
        f"- {conclusion}",
        "",
        "## 业务洞察",
    ]

    if branch == "A":
        lines.append("- 提炼出对应品类视频的“标准剧本心电图”，指导中腰部 UP 主按节奏点铺设内容包袱和互动诱饵。")
    else:
        lines.append("- 警示 MCN 机构不要盲目套用工业化模版。在 AI 等高信息密度垂类，平铺直叙的硬核讲解同样能获得极佳的长尾效应。")

    lines.extend([
        "",
        "## 局限与未来方向",
        "- 仅凭进度条无法精准定位触发情绪的画面/声音。未来需引入多模态对齐（音频语速、画面切换频率）与弹幕情绪峰值进行交叉相关分析。",
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[step5_D] Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run()

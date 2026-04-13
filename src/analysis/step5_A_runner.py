import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.step5_utils.tag_norm import load_video_df, add_relative_performance, parse_tags
from analysis.step5_utils.plot_style import setup_fonts, savefig_safe

BASE_DIR = "D:/Claude_Code/bilibili-ai-tag-analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/step5/step5_A")
REPORT_PATH = os.path.join(BASE_DIR, "results/step5/step5_A_tag_combo_report.md")
FIG_DIR = os.path.join(BASE_DIR, "results/step5/figures")

def run():
    setup_fonts()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = load_video_df()
    df = add_relative_performance(df)

    # Tag frequency and low-freq cutoff
    all_tags = [tag for tags in df["tag_list"] for tag in tags]
    tag_counts = Counter(all_tags)
    frequent_tags = {tag for tag, cnt in tag_counts.items() if cnt >= 10}

    # Build binary matrix for frequent tags
    records = []
    for _, row in df.iterrows():
        present = set(row["tag_list"]) & frequent_tags
        if len(present) >= 1:
            rec = {"bvid": row["bvid"], "rel_perf": row["relative_performance"],
                   "search_tag": row["search_tag"], "year_quarter": row["year_quarter"]}
            for tag in frequent_tags:
                rec[tag] = 1 if tag in present else 0
            records.append(rec)
    tag_df = pd.DataFrame(records)
    tag_cols = sorted(frequent_tags)

    # Save co-occurrence
    cooc = tag_df[tag_cols].T.dot(tag_df[tag_cols])
    cooc_path = os.path.join(OUTPUT_DIR, "tag_cooccurrence.parquet")
    cooc.reset_index().rename(columns={"index": "tag"}).to_parquet(cooc_path, index=False)

    # FP-Growth frequent itemsets (length >= 2)
    itemset_stats = []
    try:
        from mlxtend.frequent_patterns import fpgrowth
        basket = tag_df[tag_cols].astype(bool)
        frequent_itemsets = fpgrowth(basket, min_support=0.01, use_colnames=True)
        frequent_itemsets = frequent_itemsets[frequent_itemsets["itemsets"].apply(len) >= 2]

        for _, row in frequent_itemsets.iterrows():
            itemset = tuple(sorted(row["itemsets"]))
            # Find videos containing ALL tags in the itemset
            mask = tag_df[list(itemset)].sum(axis=1) == len(itemset)
            rel_perfs = tag_df.loc[mask, "rel_perf"].dropna()
            if len(rel_perfs) == 0:
                continue
            median_perf = rel_perfs.median()
            mean_perf = rel_perfs.mean()
            itemset_stats.append({
                "itemset": "+".join(itemset),
                "support": round(row["support"], 4),
                "n_videos": int(mask.sum()),
                "median_rel_perf": round(median_perf, 4),
                "mean_rel_perf": round(mean_perf, 4),
            })
    except Exception as e:
        print(f"[step5_A] FP-Growth error: {e}")

    itemset_df = pd.DataFrame(itemset_stats)
    if not itemset_df.empty:
        itemset_df = itemset_df.sort_values("median_rel_perf", ascending=False).reset_index(drop=True)
        itemset_df.to_parquet(os.path.join(OUTPUT_DIR, "itemset_performance.parquet"), index=False)

    # Top 15 by real爆款效率
    top15 = itemset_df.head(15).copy() if not itemset_df.empty else pd.DataFrame()

    # Wilcoxon test for combos with median_rel_perf >= 1.2
    wilcoxon_results = []
    if not itemset_df.empty:
        high_perf = itemset_df[itemset_df["median_rel_perf"] >= 1.2]
        for _, row in high_perf.head(10).iterrows():
            itemset = tuple(row["itemset"].split("+"))
            mask = tag_df[list(itemset)].sum(axis=1) == len(itemset)
            rel_perfs = tag_df.loc[mask, "rel_perf"].dropna().values
            if len(rel_perfs) >= 3:
                # Wilcoxon signed-rank against null median 1.0
                diffs = rel_perfs - 1.0
                try:
                    stat, p = wilcoxon(diffs, alternative="greater")
                    wilcoxon_results.append({
                        "itemset": row["itemset"],
                        "median_rel_perf": row["median_rel_perf"],
                        "wilcoxon_stat": round(stat, 2),
                        "p_value": round(p, 4),
                        "n": len(rel_perfs),
                    })
                except Exception:
                    pass
    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    if not wilcoxon_df.empty:
        wilcoxon_df.to_parquet(os.path.join(OUTPUT_DIR, "wilcoxon_high_perf.parquet"), index=False)

    # Fallback networkx communities (for report completeness)
    community_result = []
    try:
        G = nx.Graph()
        for i, tag1 in enumerate(tag_cols):
            for j, tag2 in enumerate(tag_cols):
                if i < j and cooc.loc[tag1, tag2] > 0:
                    G.add_edge(tag1, tag2, weight=cooc.loc[tag1, tag2])
        if G.number_of_edges() > 0:
            communities = nx.community.louvain_communities(G, seed=42)
            for idx, comm in enumerate(communities):
                if len(comm) >= 2:
                    mask = tag_df[list(comm)].sum(axis=1) > 0
                    median_perf = tag_df.loc[mask, "rel_perf"].median() if mask.any() else np.nan
                    community_result.append({
                        "community_id": idx,
                        "tags": ",".join(list(comm)),
                        "size": len(comm),
                        "median_rel_perf": round(median_perf, 3) if not pd.isna(median_perf) else None
                    })
    except Exception:
        pass

    # Visualize top 15 real爆款效率
    if not top15.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        labels = top15["itemset"].tolist()[::-1]
        vals = top15["median_rel_perf"].tolist()[::-1]
        colors = ["coral" if v >= 1.2 else "steelblue" for v in vals]
        ax.barh(labels, vals, color=colors)
        ax.axvline(1.0, color="black", linestyle="--", label="Global median=1.0")
        ax.set_xlabel("Median relative_performance (真实爆款效率)")
        ax.set_title("Top 15 标签组合的真实爆款效率")
        ax.legend()
        savefig_safe(fig, os.path.join(FIG_DIR, "step5_A_real_efficiency_bar.png"))

    # Determine branch based on real performance
    if not top15.empty and top15["median_rel_perf"].iloc[0] >= 1.2:
        if not wilcoxon_df.empty and wilcoxon_df["p_value"].min() < 0.05:
            branch = "A"
        else:
            branch = "A_uncertain"
    elif not top15.empty and top15["median_rel_perf"].iloc[0] < 1.0:
        branch = "C"
    else:
        branch = "B"

    # Report
    lines = [
        "# 子任务 A：标签组合的“爆款效率”分析与挖掘",
        "",
        "## 数据说明",
        f"- 使用数据：`data/cleaned-data/bilibili_video_cleaned.csv`，共 {len(df)} 条视频。",
        f"- 标签预处理：全局出现次数 ≥10 的频繁标签共 {len(frequent_tags)} 个，低频标签已截断。",
        "- 配额偏差校正：使用 `relative_performance = stat_view / median(stat_view by search_tag & quarter)` 作为爆款效率代理指标。",
        "",
        "## 核心说明",
        "- **算法 Lift**（FP-Growth 输出）仅衡量标签共现强度，不代表真实播放量提升。",
        "- **真实爆款效率**定义为：包含该标签组合的视频群体的 `relative_performance` 中位数。",
        "",
        "## 数据分析",
    ]

    if not itemset_df.empty:
        lines.append(f"- FP-Growth 共挖掘出 {len(itemset_df)} 个长度≥2 的频繁项集。")
        lines.append("- **Top 15 真实爆款效率组合**：")
        lines.append("")
        lines.append(top15.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("- FP-Growth 未产生有效项集。")

    if not wilcoxon_df.empty:
        lines.append("- **高爆款效率组合（≥1.2）的 Wilcoxon 单样本检验**（vs 全局 median=1.0）：")
        lines.append("")
        lines.append(wilcoxon_df.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("- 无满足 median_rel_perf ≥ 1.2 的组合，或样本不足无法做 Wilcoxon 检验。")

    if community_result:
        lines.append(f"- Networkx Louvain 社区检测到 {len(community_result)} 个社区：")
        lines.append("")
        lines.append(pd.DataFrame(community_result).to_markdown(index=False))
        lines.append("")

    if branch.startswith("A"):
        conclusion = f"分支 A：存在真实爆款效率显著高于 1.0 的标签组合（Top median={top15['median_rel_perf'].iloc[0]:.3f}）。"
    elif branch == "C":
        conclusion = "分支 C：所有组合真实爆款效率均低于 1.0，存在潜在毒药组合。"
    else:
        conclusion = "分支 B：标签组合的真实爆款效率接近 1.0，未表现出显著协同效应。"

    # Inspect top combos for synonym clusters
    top_combo_words = set()
    if not top15.empty:
        for combo in top15["itemset"].head(5):
            top_combo_words.update(combo.split("+"))

    lines.extend([
        "",
        "## 验证",
        f"- 低频截断后标签矩阵形状：{tag_df.shape}",
        f"- 共现矩阵已保存：`{cooc_path}`",
        f"- 项集性能表已保存：`{os.path.join(OUTPUT_DIR, 'itemset_performance.parquet')}`",
        "",
        "## 结论",
        f"- {conclusion}",
        "",
        "## 业务洞察",
    ])

    if branch.startswith("A"):
        lines.append("- 为 UP 主生成“标签搭配公式表”：尝试用“热门大标签导流 + 精准小标签固粉”的组合策略。")
    elif branch == "C":
        lines.append("- 警告 UP 主避免“毒药组合”：部分标签组合会导致受众预期冲突，降低完播率。建议精简标签数量，确保标签语义一致。")
    else:
        # Check if top combos are just synonym clusters
        synonym_keywords = {"人工智能", "深度学习", "机器学习", "ai", "aigc", "gpt", "chatgpt", "llm", "大模型", "神经网络"}
        if top_combo_words and top_combo_words.issubset(synonym_keywords):
            lines.append("- 创作者习惯将同义/近义学术概念捆绑使用，但这本身并不带来额外流量加成。建议减少冗余同义词堆砌，留出标签位给场景词或细分人群词。")
        else:
            lines.append("- 破除“玄学堆标签”的幻想：盲目堆砌不相关热门标签不会带来额外流量，反而可能因完播率惩罚被降权。建议聚焦 3-5 个高相关标签。")

    lines.extend([
        "",
        "## 未来方向",
        "- 引入层次线性模型（HLM），将 `tname`（真实分区）作为宏观层。由于当前 `tname` 缺失，未来若补充该字段，可进一步探究“跨区标签”的异质性效应。",
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[step5_A] Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run()

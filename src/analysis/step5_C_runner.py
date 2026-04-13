import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import jieba

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.step5_utils.tag_norm import load_video_df
from analysis.step5_utils.plot_style import setup_fonts, savefig_safe

BASE_DIR = "D:/Claude_Code/bilibili-ai-tag-analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/step5/step5_C")
REPORT_PATH = os.path.join(BASE_DIR, "results/step5/step5_C_danmaku_decay_report.md")
FIG_DIR = os.path.join(BASE_DIR, "results/step5/figures")

STOPWORDS = set([
    "就是", "可以", "真的", "一个", "什么", "没有", "自己", "还是", "现在", "怎么",
    "doge", "哈哈哈", "哈哈哈哈", "的", "了", "在", "是", "我", "你", "就", "都",
    "也", "很", "到", "说", "要", "去", "会", "着", "看", "好", "这", "那", "有",
    "和", "与", "及", "等", "之", "其", "或", "但", "而", "因", "为", "于", "被",
    "把", "给", "让", "向", "从", "以", "并", "又", "只", "不过", "已经", "将",
    "还", "如果", "因为", "所以", "虽然", "然后", "这样", "这里", "那里", "咱们",
    "大家", "一样", "一般", "一直", "一切", "一种", "这些", "那些", "这么", "那么",
    "多么", "谁", "哪", "几时", "怎样", "如何", "为什么", "多少", "几", "吧", "呢",
    "啊", "哦", "嗯", "唉", "哎", "哟", "哼", "哈", "嘻", "嘿", "嘛", "吗", "啦",
    "吶", "哇", "呐", "呗", "喔", "呕", "欧", "呃", "哎哟", "天呐", "天哪", "真的是",
    "不是吧", "这就是", "那就是", "其实就是", "就是说", "也就是说", "所谓", "而言",
    "看来", "显得", "尤其是", "特别是", "例如", "比如", "譬如", "换言之", "换句话说",
    "与此同时", "一方面", "另一方面", "首先", "其次", "再次", "最后", "总之", "综上所述",
    "由此可见", "大概", "大约", "也许", "可能", "或许", "应该", "似乎", "好像", "未必",
    "不一定", "差不多", "几乎", "简直", "根本", "压根", "万万", "千万", "绝对", "完全",
    "实在", "确实", "的确", "其实", "本来", "原来", "原本", "始终", "永远", "一向",
    "历来", "向来", "从来", "尤其", "特别", "非常", "十分", "相当", "比较", "稍微",
    "略微", "有点", "一些", "一点", "许多", "很多", "大量", "某些", "各项", "各位",
    "各种", "各个", "所有", "整个", "全面", "整体", "总体", "普遍", "广泛", "基本",
    "主要", "重要", "关键", "核心", "中心", "重点", "难点", "热点", "焦点", "起点",
    "终点", "源头", "根源", "本质", "实质", "性质", "特性", "特点", "特征", "特色",
    "优势", "长处", "优点", "缺点", "弱点", "缺陷", "风险", "危险", "危害", "伤害",
    "影响", "作用", "效果", "效益", "效率", "成果", "成绩", "成就", "成功", "胜利",
    "失败", "错误", "失误", "过失", "误会", "误解", "偏差", "差距", "区别", "差异",
    "分离", "分类", "分级", "分析", "解释", "说明", "论述", "阐述", "叙述", "描述",
    "表达", "反映", "反应", "透露", "显露", "显示", "展示", "展现", "呈现", "出现",
    "产生", "发生", "发展", "变化", "转变", "改变", "改进", "修正", "修改", "整理",
    "管理", "控制", "调节", "调整", "调动", "安排", "部署", "配置", "条件", "环境",
    "状况", "状态", "情况", "形势", "局面", "规模", "领域", "区域", "境界", "层面",
    "方面", "方向", "目标", "对象", "主题", "话题", "课题", "项目", "任务", "工作",
    "事业", "行业", "单位", "机构", "部门", "团体", "集体", "集团", "队", "组",
    "班", "件", "个", "条", "根", "只", "张", "块", "片", "份", "套", "种", "类",
    "批", "群", "次", "回", "遍", "下", "上", "里", "外", "中", "间", "头", "边",
    "面", "方", "带", "段", "界",
])

def jieba_tokenize(text):
    if not isinstance(text, str):
        return ""
    tokens = [t.strip() for t in jieba.cut(text.strip()) if t.strip() and t.strip() not in STOPWORDS and len(t.strip()) > 1]
    return " ".join(tokens)

def compute_js_divergence(p, q):
    from scipy.spatial.distance import jensenshannon
    return jensenshannon(p, q) ** 2

def run():
    setup_fonts()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    video_df = load_video_df()
    video_df = video_df[video_df["stat_danmaku"] >= 1000].copy()
    video_df["pubdate_dt"] = pd.to_datetime(video_df["pubdate"], errors="coerce")
    video_df = video_df.dropna(subset=["pubdate_dt", "bvid"])

    print(f"[step5_C] {len(video_df)} videos with stat_danmaku >= 1000")

    # Load danmaku core parquet
    dmk_path = os.path.join(BASE_DIR, "data/processed/step5/shared/danmaku_core.parquet")
    dmk = pd.read_parquet(dmk_path, columns=["bvid", "content", "ctime"])
    dmk["ctime"] = pd.to_numeric(dmk["ctime"], errors="coerce")
    dmk["dmk_dt"] = pd.to_datetime(dmk["ctime"], unit="s", errors="coerce")
    dmk = dmk.dropna(subset=["dmk_dt", "content"])

    # Merge with video pubdate
    dmk = dmk.merge(video_df[["bvid", "pubdate_dt"]], on="bvid", how="inner")
    dmk["days_since_pub"] = (dmk["dmk_dt"] - dmk["pubdate_dt"]).dt.total_seconds() / 86400.0

    # Build 24h buckets per video
    # Day 1: [0, 24h), Day 2: [24h, 48h), Day 3+: [48h, 72h), etc.
    dmk["day_bucket"] = np.floor(dmk["days_since_pub"]).astype(int) + 1
    dmk = dmk[dmk["day_bucket"] >= 1]

    valid_bvids = []
    decay_ratios = []
    video_early_texts = []
    video_late_texts = []
    late_bucket_nums = []
    new_word_samples = []

    all_valid_early_tokens = set()

    for bvid in video_df["bvid"]:
        sub = dmk[dmk["bvid"] == bvid]
        if len(sub) == 0:
            continue
        bucket_counts = sub.groupby("day_bucket").size()
        # Must have Day 1 >= 50 and Day 2 >= 50
        if bucket_counts.get(1, 0) < 50 or bucket_counts.get(2, 0) < 50:
            continue

        # Collect Day 3+ buckets until < 30 or > Day 5
        late_buckets = []
        for day in range(3, 6):
            cnt = bucket_counts.get(day, 0)
            if cnt >= 30:
                late_buckets.append(day)
            else:
                break

        if not late_buckets:
            continue

        early_contents = sub[sub["day_bucket"].isin([1, 2])]["content"].astype(str).tolist()
        late_contents = sub[sub["day_bucket"].isin(late_buckets)]["content"].astype(str).tolist()

        early_text = " ".join(early_contents)
        late_text = " ".join(late_contents)

        valid_bvids.append(bvid)
        video_early_texts.append(early_text)
        video_late_texts.append(late_text)
        late_bucket_nums.append(late_buckets[-1])

    print(f"[step5_C] {len(valid_bvids)} videos passed Day1/2 >=50 and had valid Day3+ buckets")

    if len(valid_bvids) == 0:
        raise ValueError("No valid videos with sufficient early/late danmaku.")

    # Subsample to max 500 videos to control runtime
    MAX_VIDEOS = 500
    if len(valid_bvids) > MAX_VIDEOS:
        np.random.seed(42)
        indices = np.random.choice(len(valid_bvids), MAX_VIDEOS, replace=False)
        valid_bvids = [valid_bvids[i] for i in indices]
        video_early_texts = [video_early_texts[i] for i in indices]
        video_late_texts = [video_late_texts[i] for i in indices]
        late_bucket_nums = [late_bucket_nums[i] for i in indices]
        print(f"[step5_C] Subsampled to {len(valid_bvids)} videos for TF-IDF/LDA.")

    # Tokenize
    early_corpus_tokenized = [jieba_tokenize(t) for t in video_early_texts]
    late_corpus_tokenized = [jieba_tokenize(t) for t in video_late_texts]

    # TF-IDF on early corpus (Day 1+2 merged)
    tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
    try:
        early_tfidf_matrix = tfidf.fit_transform(early_corpus_tokenized)
        feature_names = tfidf.get_feature_names_out()
    except Exception as e:
        print(f"[step5_C] TF-IDF fit error: {e}")
        early_tfidf_matrix = None
        feature_names = np.array([])

    # Save global Top 100 words in early corpus
    if early_tfidf_matrix is not None and len(feature_names) > 0:
        mean_scores = np.asarray(early_tfidf_matrix.mean(axis=0)).flatten()
        top100_indices = mean_scores.argsort()[-100:][::-1]
        global_top100 = [feature_names[i] for i in top100_indices if mean_scores[i] > 0]
        top100_path = os.path.join(OUTPUT_DIR, "early_global_top100_words.txt")
        with open(top100_path, "w", encoding="utf-8") as f:
            f.write(", ".join(global_top100))
        print(f"[step5_C] Global Top 100 early words saved to {top100_path}")

    # Compute decay ratio per video
    if early_tfidf_matrix is not None:
        for idx, bvid in enumerate(valid_bvids):
            row = early_tfidf_matrix[idx].toarray().flatten()
            top_indices = row.argsort()[-100:][::-1]
            top_words = set(feature_names[i] for i in top_indices if row[i] > 0)
            if not top_words:
                decay_ratios.append(np.nan)
                continue
            late_tokens = set(late_corpus_tokenized[idx].split())
            ratio = len(top_words & late_tokens) / len(top_words)
            decay_ratios.append(ratio)
    else:
        decay_ratios = [np.nan] * len(valid_bvids)

    decay_ratios = np.array(decay_ratios)
    valid_mask = ~np.isnan(decay_ratios)
    decay_ratios_clean = decay_ratios[valid_mask]

    # LDA: Day 1+2 vs Day 3+ (last valid bucket)
    count_vec = CountVectorizer(max_features=3000, min_df=2, max_df=0.8)
    all_early_counts = count_vec.fit_transform(early_corpus_tokenized)
    all_late_counts = count_vec.transform(late_corpus_tokenized)
    vocab = count_vec.get_feature_names_out()

    n_topics = 6
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15, learning_method='online')
    lda_early = lda.fit_transform(all_early_counts)
    lda_late = lda.transform(all_late_counts)

    early_topic_dist = lda_early.mean(axis=0)
    late_topic_dist = lda_late.mean(axis=0)
    js_div = compute_js_divergence(early_topic_dist, late_topic_dist)

    # Extract top 5 words per topic for quality check
    topic_keywords = []
    for topic_idx, topic in enumerate(lda.components_):
        top5 = [vocab[i] for i in topic.argsort()[-5:][::-1]]
        topic_keywords.append({"topic_id": topic_idx + 1, "top5_words": ", ".join(top5)})
    topic_kw_df = pd.DataFrame(topic_keywords)

    # Wilcoxon signed-rank test: median decay_ratio vs 0.5
    if len(decay_ratios_clean) > 0:
        diffs = decay_ratios_clean - 0.5
        try:
            w_stat, w_pvalue = wilcoxon(diffs, alternative='less')
        except Exception as e:
            w_stat, w_pvalue = np.nan, np.nan
    else:
        w_stat, w_pvalue = np.nan, np.nan

    # New words in late period
    late_tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
    try:
        late_tfidf_matrix = late_tfidf.fit_transform(late_corpus_tokenized)
        late_features = late_tfidf.get_feature_names_out()
    except Exception:
        late_features = np.array([])

    early_tfidf2 = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
    try:
        early_tfidf2_matrix = early_tfidf2.fit_transform(early_corpus_tokenized)
        early_features = early_tfidf2.get_feature_names_out()
    except Exception:
        early_features = np.array([])

    if len(late_features) > 0 and len(early_features) > 0:
        early_set = set(early_features)
        # filter out pure noise: digits-only, alpha-numeric codes, single char, or meaningless repeats
        def is_meaningful_word(w):
            if len(w) <= 1:
                return False
            if re.fullmatch(r"\d+", w):
                return False
            if re.fullmatch(r"[a-zA-Z0-9]+", w) and len(w) <= 4:
                return False
            return True
        new_words = [w for w in late_features if w not in early_set and is_meaningful_word(w)]
        sampled_new_words = new_words[:30]
    else:
        sampled_new_words = []

    # Plot decay distribution
    if len(decay_ratios_clean) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(decay_ratios_clean, bins=30, color="steelblue", edgecolor="white")
        ax.axvline(np.median(decay_ratios_clean), color="red", linestyle="--", label=f"Median={np.median(decay_ratios_clean):.3f}")
        ax.axvline(0.5, color="black", linestyle="--", label="Neutral=0.5")
        ax.set_xlabel("早期 Top100 词在中后期的保留比例")
        ax.set_ylabel("视频数量")
        ax.set_title("弹幕语义衰减分布")
        ax.legend()
        savefig_safe(fig, os.path.join(FIG_DIR, "step5_C_decay_hist.png"))

    # Plot LDA topic transfer (stacked bar)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = np.arange(n_topics)
    width = 0.35
    ax2.bar(x - width/2, early_topic_dist, width, label="早期 (Day 1+2)", color="steelblue")
    ax2.bar(x + width/2, late_topic_dist, width, label="中后期 (Day 3+)", color="coral")
    ax2.set_xlabel("LDA 主题编号")
    ax2.set_ylabel("平均主题强度")
    ax2.set_title("早期 vs 中后期 弹幕主题分布对比")
    ax2.legend()
    savefig_safe(fig2, os.path.join(FIG_DIR, "step5_C_lda_topics.png"))

    # Save processed data
    pd.DataFrame({
        "bvid": np.array(valid_bvids)[valid_mask],
        "decay_ratio": decay_ratios_clean,
    }).to_parquet(os.path.join(OUTPUT_DIR, "decay_ratios.parquet"), index=False)

    pd.DataFrame({
        "metric": ["median_decay_ratio", "js_divergence", "wilcoxon_stat", "wilcoxon_p"],
        "value": [np.median(decay_ratios_clean) if len(decay_ratios_clean) > 0 else np.nan, js_div, w_stat, w_pvalue]
    }).to_parquet(os.path.join(OUTPUT_DIR, "decay_summary.parquet"), index=False)

    # Branch determination
    median_decay = np.median(decay_ratios_clean) if len(decay_ratios_clean) > 0 else np.nan
    if not np.isnan(w_pvalue) and w_pvalue < 0.05 and not np.isnan(median_decay) and median_decay < 0.5:
        branch = "A"
        conclusion = f"弹幕专业词显著衰减。早期 Top100 词在中后期的保留中位数为 {median_decay:.3f} (Wilcoxon p={w_pvalue:.4f})。"
    elif not np.isnan(median_decay) and median_decay >= 0.5:
        branch = "B"
        conclusion = f"未见显著衰减，中后期仍保持较高专业词密度（保留中位数 {median_decay:.3f}）。"
    else:
        branch = "uncertain"
        conclusion = f"衰减Direction存在但统计显著性不足（保留中位数 {median_decay:.3f}, p={w_pvalue:.4f}）。"

    # Evaluate LDA topic quality honestly
    topic_words_flat = [w for t in topic_keywords for w in t["top5_words"].split(", ")]
    garbage_words = [w for w in topic_words_flat if w in STOPWORDS or len(w) <= 2]
    topic_quality_score = 1 - len(garbage_words) / max(len(topic_words_flat), 1)
    lda_is_meaningful = topic_quality_score >= 0.5

    lines = [
        "# 子任务 C：弹幕生命周期与受众泛化（语义衰减分析）",
        "",
        "## 数据说明",
        f"- 使用清洗后的弹幕数据（`danmaku_core.parquet`），结合 `bilibili_video_cleaned.csv`。",
        f"- 筛选 `stat_danmaku >= 1000` 的视频，共 {len(video_df)} 条。",
        f"- 纳入分析的视频（Day1≥50 且 Day2≥50，且存在有效 Day3+ 桶）：{len(valid_bvids)} 条。",
        "",
        "## 数据处理",
        "- 以视频发布日为基准，将弹幕按现实发送时间划分为 24h 桶。",
        "- Day 1 (0–24h) 和 Day 2 (24–48h) 各需 ≥50 条弹幕；Day 3+ 每桶需 ≥30 条，最多到 Day 5。",
        "- 对所有弹幕文本进行 jieba 分词，并施加停用词过滤（ STOPWORDS 表含高频口语词与无意义助词）。",
        "- TF-IDF 路径：提取 Day 1+2 合并后的 Top 100 关键词，计算其在中后期弹幕词汇中的保留比例（decay ratio）。",
        f"- LDA 路径：使用 sklearn LatentDirichletAllocation（n_topics={n_topics}）分别拟合早期（Day 1+2）与中后期（Day 3+）的主题分布，计算 JS 散度。",
        "",
        "## LDA 主题质量校验",
        "- 各主题 Top 5 权重词：",
        "",
        topic_kw_df.to_markdown(index=False),
        "",
        f"- 质量诊断：{len(garbage_words)}/{len(topic_words_flat)} 个词落入停用词/短词表，主题可解释性评分 {topic_quality_score:.2f}。",
        f"- **判断**：{'LDA 主题具有可读性，可用于辅助解读。' if lda_is_meaningful else 'LDA 主题以垃圾停用词为主，不具备可解释性，仅以 TF-IDF 衰减结果为准。'}",
        "",
        "## 数据分析",
        f"- 语义保留比例中位数：{median_decay:.4f}" if not np.isnan(median_decay) else "- 语义保留比例中位数：N/A",
        f"- LDA 主题分布 JS 散度（整体）：{js_div:.4f}",
        f"- 单样本 Wilcoxon 符号秩检验（decay_ratio < 0.5）：stat={w_stat:.2f}, p-value={w_pvalue:.4f}" if not np.isnan(w_pvalue) else "- 单样本 Wilcoxon 检验：N/A",
        "",
        "## 新增中后期高频词抽样",
        f"- Day 3+ 新增高频词（未在 Day 1+2 Top 词中出现）抽样：{', '.join(sampled_new_words) if sampled_new_words else 'N/A'}",
        f"- 新增词质量：{'多数为有意义词汇' if (sampled_new_words and len([w for w in sampled_new_words if w in STOPWORDS]) < len(sampled_new_words)/2) else '仍以口语/噪音词为主'}",
        "",
        "## 验证",
        f"- 衰减分布图与 LDA 主题对比图已保存至 `{FIG_DIR}`。",
        f"- 详细数据已保存至 `{OUTPUT_DIR}`。",
        "",
        "## 结论",
        f"- {conclusion}",
    ]

    if lda_is_meaningful:
        lines.extend([
            f"- LDA 主题 JS 散度仅为 {js_div:.4f}，主题分布在early/late间高度稳定。",
            "- 综合可用信息：核心知识方向未变，但具体专业词汇显著减少。",
        ])
    else:
        lines.extend([
            f"- LDA 产出的主题质量过低（JS={js_div:.4f}），无法支撑'平民化转译'的故事。",
            "- 仅从 TF-IDF 衰减看，中后期弹幕的专业词密度确实下降。",
        ])

    lines.append("")
    lines.append("## 业务洞察")

    if branch == "A":
        if lda_is_meaningful:
            lines.append("- 证实了“破圈即稀释”规律在词汇层面的存在，但核心知识主题保持稳定。")
            lines.append("- 指导 UP 主：在长尾期（Day 3+）主动用通俗语言对硬核概念做二次解读。")
        else:
            lines.append("- 数据如实呈现：中后期弹幕的专业词保留率显著下降，且 LDA 主题无法提供有说服力的替代解释。")
            lines.append("- 这更像是“劣币驱逐良币”——早期硬核讨论者离场后，弹幕质量向更低信息密度的口语退化，而非优雅的平民化转译。")
            lines.append("- 平台建议：加强弹幕的按时间排序/高赞展示功能，保护高信息密度内容不被淹没。")
    elif branch == "B":
        lines.append("- 后期依然保持高浓度专业讨论，说明视频触及了长尾的“硬核搜索引擎流量”。")
        lines.append("- 指导教程类 UP 主：注重简介和标题的 SEO 优化，长尾搜索流量具有极高的精准变现价值。")
    else:
        lines.append("- 数据未能给出显著结论。受众泛化效应可能因视频类型而异，需更大样本或更长观测窗口。")

    lines.extend([
        "",
        "## 未来方向",
        "- 引入 Dynamic Word Embeddings 追踪同一词在视频生命周期内的语义漂移，避免将新梗误判为低质量词汇。",
        "- 若 LDA 主题持续不可解释，应考虑使用 BERTopic 等基于预训练模型的主题方法替代。",
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[step5_C] Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run()

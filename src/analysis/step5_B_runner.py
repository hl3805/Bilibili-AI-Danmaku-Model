import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.step5_utils.tag_norm import load_video_df
from analysis.step5_utils.plot_style import setup_fonts, savefig_safe

BASE_DIR = "D:/Claude_Code/bilibili-ai-tag-analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/step5/step5_B")
REPORT_PATH = os.path.join(BASE_DIR, "results/step5/step5_B_title_tag_consistency_report.md")
FIG_DIR = os.path.join(BASE_DIR, "results/step5/figures")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Collapse excessive repeated chars (e.g., 哈哈哈哈哈 -> 哈哈)
    text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
    # Remove common meaningless emoticons/kaomoji patterns
    text = re.sub(r'[\(\)（）【】]+', '', text)
    return text

def load_model_with_fallback():
    from sentence_transformers import SentenceTransformer
    model_names = [
        "shibing624/text2vec-base-chinese",
        "BAAI/bge-base-zh-v1.5",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ]
    for name in model_names:
        try:
            print(f"[step5_B] Loading model: {name}")
            model = SentenceTransformer(name)
            print(f"[step5_B] Model loaded: {name}")
            return model, name
        except Exception as e:
            print(f"[step5_B] Failed to load {name}: {e}")
            continue
    raise RuntimeError("All sentence-transformer models failed to load.")

def run():
    setup_fonts()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = load_video_df()
    df = df.dropna(subset=["title", "tags"])
    df["title_clean"] = df["title"].apply(clean_text)
    df["tags_clean"] = df["tags"].apply(clean_text)

    model, model_name = load_model_with_fallback()

    # Batch encode
    title_emb = model.encode(df["title_clean"].tolist(), batch_size=256, show_progress_bar=False, convert_to_numpy=True)
    tags_emb = model.encode(df["tags_clean"].tolist(), batch_size=256, show_progress_bar=False, convert_to_numpy=True)

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    S = cosine_similarity(title_emb, tags_emb).diagonal()
    df["S"] = S

    # Mean-centering to eliminate multicollinearity between S and S^2
    S_bar = df["S"].mean()
    df["S_c"] = df["S"] - S_bar
    df["S_sq"] = df["S_c"] ** 2

    df["log_view"] = np.log1p(df["stat_view"])
    df["log_duration"] = np.log1p(df["duration"].clip(lower=1))
    df["pubdate_dt"] = pd.to_datetime(df["pubdate"], errors="coerce")
    df["days_since_publish"] = (pd.Timestamp.now() - df["pubdate_dt"]).dt.days

    # OLS regression
    formula = "log_view ~ S_c + S_sq + log_duration + days_since_publish + C(search_tag)"
    try:
        model_fit = smf.ols(formula=formula, data=df).fit()
        summary_text = model_fit.summary().as_text()
        r2 = model_fit.rsquared
        s_pvalue = model_fit.pvalues.get("S_c", np.nan)
        sq_pvalue = model_fit.pvalues.get("S_sq", np.nan)
        s_coef = model_fit.params.get("S_c", np.nan)
        sq_coef = model_fit.params.get("S_sq", np.nan)
        condition_number = model_fit.condition_number
    except Exception as e:
        summary_text = str(e)
        r2 = s_pvalue = sq_pvalue = s_coef = sq_coef = condition_number = np.nan

    # Save outputs
    df[["bvid", "title", "tags", "S", "S_c", "log_view", "search_tag"]].to_parquet(
        os.path.join(OUTPUT_DIR, "consistency_scores.parquet"), index=False
    )

    # Plot S_c histogram (EDA sentinel)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["S_c"], bins=40, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", label="Mean-centered zero")
    ax.set_xlabel("S_c (Mean-centered Title-Tag Similarity)")
    ax.set_ylabel("Count")
    ax.set_title("S_c Distribution (EDA Sentinel)")
    ax.legend()
    savefig_safe(fig, os.path.join(FIG_DIR, "step5_B_Sc_histogram.png"))

    # Plot S vs log_view
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(df["S"], df["log_view"], alpha=0.3, s=10, color="steelblue")
    ax2.set_xlabel("标题-标签一致性 S")
    ax2.set_ylabel("log(播放量)")
    ax2.set_title("标题-标签一致性与播放量关系")
    savefig_safe(fig2, os.path.join(FIG_DIR, "step5_B_consistency_scatter.png"))

    # Plot residuals if model succeeded
    if not pd.isna(r2):
        df["fitted"] = model_fit.fittedvalues
        df["resid"] = model_fit.resid
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.scatter(df["fitted"], df["resid"], alpha=0.3, s=10, color="coral")
        ax3.axhline(0, color="black", linestyle="--")
        ax3.set_xlabel("Fitted values")
        ax3.set_ylabel("Residuals")
        ax3.set_title("OLS 残差图")
        savefig_safe(fig3, os.path.join(FIG_DIR, "step5_B_residuals.png"))

    # Determine branch
    if not pd.isna(s_pvalue) and s_pvalue < 0.05 and s_coef > 0 and (pd.isna(sq_pvalue) or sq_pvalue >= 0.05 or sq_coef >= 0):
        branch = "A"
        conclusion = "S_c 系数显著为正，说明标题与标签高度一致时播放量更高。"
    elif not pd.isna(sq_pvalue) and sq_pvalue < 0.05 and sq_coef < 0:
        branch = "B"
        conclusion = "发现显著的倒 U 型关系：适度偏离（标题悬念 + 标签老实）可能带来更高播放量。"
    else:
        branch = "C"
        conclusion = "S 与播放量之间无显著线性或二次关系。标题党策略可能仍然有效。"

    lines = [
        "# 子任务 B：标题-标签一致性对推荐机制的调节效应",
        "",
        "## 数据说明",
        f"- 使用数据：`data/cleaned-data/bilibili_video_cleaned.csv`，共 {len(df)} 条视频。",
        f"- 语义模型：`{model_name}`（Sentence-BERT）。",
        "- 文本预处理：清理无意义颜文字、过长重复字符，**未使用 jieba 分词或停用词过滤**。",
        "- 一致性度量：计算标题向量与标签向量的余弦相似度，得到 S ∈ [0, 1]。",
        "",
        "## 核心局限性",
        "- `owner_fans` 全为 0，无法将 UP 主规模作为控制变量纳入回归。我们仅能观察内容层面的调节效应，无法给出“大号 vs 小号”的分层运营建议。",
        "",
        "## 数据处理",
        f"- S 均值：{S_bar:.4f}",
        "- 对 S 进行 Mean-Centering：`S_c = S - mean(S)`，以消除 S 与 S² 之间的多重共线性。",
        f"- OLS 回归模型：`log(stat_view) ~ S_c + S_c^2 + log(duration) + days_since_publish + C(search_tag)`",
        f"- Condition number：{condition_number:.1f}" if not pd.isna(condition_number) else "- Condition number：N/A",
        "",
        "## 数据分析",
        f"- R² = {round(r2, 4) if not pd.isna(r2) else 'N/A'}",
        f"- S_c 系数 = {round(s_coef, 4) if not pd.isna(s_coef) else 'N/A'}，p-value = {round(s_pvalue, 4) if not pd.isna(s_pvalue) else 'N/A'}",
        f"- S_c² 系数 = {round(sq_coef, 4) if not pd.isna(sq_coef) else 'N/A'}，p-value = {round(sq_pvalue, 4) if not pd.isna(sq_pvalue) else 'N/A'}",
        "",
        "### 回归摘要",
        "```text",
        summary_text,
        "```",
        "",
        "## 验证",
        f"- S 的取值范围：[ {df['S'].min():.4f}, {df['S'].max():.4f} ]",
        f"- S_c 的取值范围：[ {df['S_c'].min():.4f}, {df['S_c'].max():.4f} ]",
        "- S_c 直方图、散点图与残差图已保存至 `results/step5/figures/`。",
        "",
        "## 结论",
        f"- {conclusion}",
        "",
        "## 业务洞察",
    ]

    if branch == "A":
        lines.append("- B 站推荐系统具备语义纠偏能力：文题相符是获得稳定推荐流量的最佳策略。UP 主应避免挂羊头卖狗肉。")
    elif branch == "B":
        lines.append("- 推荐策略：标题负责情绪和悬念，标签负责 SEO 和机器定性。UP 主可以采取“双轨制”命名策略。")
    else:
        lines.append("- 在现有数据中，标题党的高点击率足以弥补完播率折损。但这并不意味着推荐系统鼓励文题不符，可能只是样本中存在强标题党的幸存者偏差。")

    lines.extend([
        "",
        "## 未来方向",
        "- 未来可利用 LLM 进行 Few-shot 意图打标（疑问型、震惊型、教程型），在不同标题意图下分别计算一致性的影响。纯 Embedding 无法识别反讽或反差萌式标题。",
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[step5_B] Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run()

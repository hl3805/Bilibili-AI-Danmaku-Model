import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import kruskal

# Guard against OMP duplicate lib issue in bili_gpu env
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.step5_utils.tag_norm import load_video_df
from analysis.step5_utils.plot_style import setup_fonts, savefig_safe

BASE_DIR = "D:/Claude_Code/bilibili-ai-tag-analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/step5/step5_H")
REPORT_PATH = os.path.join(BASE_DIR, "results/step5/step5_H_danmaku_qa_report.md")
FIG_DIR = os.path.join(BASE_DIR, "results/step5/figures")

Q_keywords = ["怎么", "为什么", "求问", "不懂", "请问", "疑问", "如何", "咋办", "求解"]

def is_question(text):
    if not isinstance(text, str):
        return False
    return any(kw in text for kw in Q_keywords)

def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            free = total - allocated
            print(f"[step5_H] GPU total={total:.1f}GB, allocated={allocated:.1f}GB, free={free:.1f}GB")
            if free < 2.5:
                print("[step5_H] GPU free memory < 2.5GB, forcing CPU to avoid shared-memory thrashing.")
                return "cpu"
            return "cuda"
    except Exception as e:
        print(f"[step5_H] GPU check failed: {e}, falling back to CPU.")
    return "cpu"

def load_model_with_fallback():
    from sentence_transformers import SentenceTransformer
    model_names = [
        "shibing624/text2vec-base-chinese",
        "BAAI/bge-base-zh-v1.5",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ]
    device = get_device()
    for name in model_names:
        try:
            print(f"[step5_H] Loading model: {name} on {device}")
            model = SentenceTransformer(name, device=device)
            print(f"[step5_H] Model loaded: {name} on {device}")
            return model, name, device
        except Exception as e:
            print(f"[step5_H] Failed to load {name}: {e}")
            continue
    raise RuntimeError("All sentence-transformer models failed to load.")

def run():
    setup_fonts()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    video_df = load_video_df()
    video_df = video_df[video_df["stat_danmaku"] >= 200].copy()
    video_df = video_df.dropna(subset=["bvid", "search_tag"])
    MAX_V = 500
    if len(video_df) > MAX_V:
        video_df = video_df.sample(n=MAX_V, random_state=42).reset_index(drop=True)
    print(f"[step5_H] {len(video_df)} videos selected (after subsample)")

    dmk_path = os.path.join(BASE_DIR, "data/processed/step5/shared/danmaku_core.parquet")
    dmk = pd.read_parquet(dmk_path, columns=["bvid", "content", "progress", "ctime"])
    dmk = dmk.merge(video_df[["bvid", "search_tag"]], on="bvid", how="inner")
    dmk["ctime"] = pd.to_numeric(dmk["ctime"], errors="coerce")
    dmk = dmk.dropna(subset=["content", "progress", "ctime"])
    dmk["is_q"] = dmk["content"].apply(is_question)

    model, model_name, device = load_model_with_fallback()
    encode_batch = 256 if device == "cuda" else 64

    # ---- Noise baseline: 1000 random unrelated pairs ----
    print("[step5_H] Computing noise baseline (random pairs)...")
    unique_bvids = dmk["bvid"].unique()
    np.random.seed(42)
    random_pairs = []
    for _ in range(1000):
        b1, b2 = np.random.choice(unique_bvids, 2, replace=False)
        c1 = dmk[dmk["bvid"] == b1]["content"].sample(1, random_state=np.random.randint(0, 99999)).iloc[0]
        c2 = dmk[dmk["bvid"] == b2]["content"].sample(1, random_state=np.random.randint(0, 99999)).iloc[0]
        random_pairs.append((c1, c2))

    texts1 = [p[0] for p in random_pairs]
    texts2 = [p[1] for p in random_pairs]
    emb1 = model.encode(texts1, batch_size=encode_batch, convert_to_numpy=True, show_progress_bar=False)
    emb2 = model.encode(texts2, batch_size=encode_batch, convert_to_numpy=True, show_progress_bar=False)
    from sklearn.metrics.pairwise import cosine_similarity
    random_sims = cosine_similarity(emb1, emb2).diagonal()
    random_95th = float(np.percentile(random_sims, 95))
    print(f"[step5_H] Random pairs 95th percentile similarity: {random_95th:.4f}")

    del emb1, emb2
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- QA extraction per video ----
    qa_records = []
    video_stats = []
    pilot_samples = []

    for bvid, group in dmk.groupby("bvid"):
        group = group.sort_values(["progress", "ctime"]).reset_index(drop=True)
        qs = group[group["is_q"]].copy()
        if len(qs) == 0:
            video_stats.append({
                "bvid": bvid, "total_q": 0, "answered_q": 0,
                "response_rate": 0.0, "mean_latency": np.nan,
                "search_tag": group["search_tag"].iloc[0]
            })
            continue
        # Pre-encode all contents in this video
        contents = group["content"].astype(str).tolist()
        embeddings = model.encode(contents, batch_size=encode_batch, convert_to_numpy=True, show_progress_bar=False)

        answered = 0
        latencies = []

        for _, q_row in qs.iterrows():
            q_idx = q_row.name
            q_prog = q_row["progress"]
            q_time = q_row["ctime"]
            q_text = q_row["content"]

            # Candidates within [-5 sec, +15 sec] progress window and strict ctime > q_time
            prog_low = max(0, q_prog - 5)
            prog_high = q_prog + 15
            candidates = group[
                (group["progress"] >= prog_low) &
                (group["progress"] <= prog_high) &
                (group["ctime"] > q_time) &
                (group["ctime"] <= q_time + 15)
            ]
            # Exclude the question itself
            candidates = candidates[candidates.index != q_idx]
            if len(candidates) == 0:
                continue

            # Semantic match (all candidates must pass Sentence-BERT)
            cand_indices = candidates.index.tolist()
            cand_texts = candidates["content"].astype(str).tolist()
            # filter Q=A exact duplicates
            valid_mask = [ct != q_text for ct in cand_texts]
            if not any(valid_mask):
                continue
            cand_indices = [ci for ci, vm in zip(cand_indices, valid_mask) if vm]
            candidates = candidates.loc[cand_indices]
            if len(candidates) == 0:
                continue

            q_vec = embeddings[q_idx].reshape(1, -1)
            c_vecs = embeddings[cand_indices]
            sims = cosine_similarity(q_vec, c_vecs)[0]

            # Pilot sampling: save first 30 highest-sim candidates across all videos
            if len(pilot_samples) < 30:
                for sim_val, (_, cand_row) in zip(sims, candidates.iterrows()):
                    pilot_samples.append({
                        "bvid": bvid,
                        "q_content": q_text,
                        "a_content": cand_row["content"],
                        "similarity": round(float(sim_val), 4),
                    })
                pilot_samples = sorted(pilot_samples, key=lambda x: x["similarity"], reverse=True)[:30]

            best_idx = sims.argmax()
            best_sim = sims[best_idx]
            if best_sim > random_95th:
                best_cand = candidates.iloc[best_idx]
                answered += 1
                latencies.append(best_cand["ctime"] - q_time)
                qa_records.append({
                    "bvid": bvid, "q_content": q_text,
                    "a_content": best_cand["content"],
                    "latency_sec": best_cand["ctime"] - q_time,
                    "progress_q": q_prog, "match_type": "semantic",
                    "similarity": round(float(best_sim), 4)
                })

        total_q = len(qs)
        response_rate = answered / total_q if total_q > 0 else 0.0
        mean_latency = np.mean(latencies) if latencies else np.nan
        video_stats.append({
            "bvid": bvid, "total_q": total_q, "answered_q": answered,
            "response_rate": response_rate, "mean_latency": mean_latency,
            "search_tag": group["search_tag"].iloc[0]
        })

        del embeddings
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"[step5_H] QA extraction done. {len(qa_records)} QA pairs found.")

    stats_df = pd.DataFrame(video_stats)
    stats_df.to_parquet(os.path.join(OUTPUT_DIR, "qa_video_stats.parquet"), index=False)
    qa_df = pd.DataFrame(qa_records) if qa_records else pd.DataFrame(
        columns=["bvid", "q_content", "a_content", "latency_sec", "progress_q", "match_type", "similarity"]
    )
    if not qa_df.empty:
        qa_df.to_parquet(os.path.join(OUTPUT_DIR, "qa_pairs.parquet"), index=False)

    # Save pilot sample
    pilot_df = pd.DataFrame(pilot_samples)
    if not pilot_df.empty:
        pilot_df.to_csv(os.path.join(OUTPUT_DIR, "qa_pilot_sample.txt"), sep="\t", index=False, encoding="utf-8")

    # Kruskal-Wallis across search_tag
    groups = [group["response_rate"].dropna().values for name, group in stats_df.groupby("search_tag") if len(group) >= 5]
    if len(groups) >= 2:
        h_stat, p_kw = kruskal(*groups)
    else:
        h_stat = p_kw = np.nan

    # Plot response rate distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stats_df["response_rate"].dropna(), bins=30, color="steelblue", edgecolor="white")
    ax.axvline(stats_df["response_rate"].median(), color="red", linestyle="--",
               label=f"Median={stats_df['response_rate'].median():.3f}")
    ax.set_xlabel("提问响应率")
    ax.set_ylabel("视频数量")
    ax.set_title("弹幕 QA 响应率分布")
    ax.legend()
    savefig_safe(fig, os.path.join(FIG_DIR, "step5_H_response_rate_hist.png"))

    # Tag-level barplot
    tag_medians = stats_df.groupby("search_tag")["response_rate"].median().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(tag_medians.index, tag_medians.values, color="coral")
    ax2.set_xlabel("Median Response Rate")
    ax2.set_title("不同 search_tag 的弹幕 QA 响应率中位数")
    savefig_safe(fig2, os.path.join(FIG_DIR, "step5_H_tag_response_rate.png"))

    median_rate = stats_df["response_rate"].median()
    if not np.isnan(p_kw) and p_kw < 0.05:
        branch = "A"
        conclusion = f"弹幕 QA 响应率存在显著差异（Kruskal-Wallis H={h_stat:.2f}, p={p_kw:.4f}），不同标签分区的互助生态不均衡。"
    else:
        branch = "B"
        conclusion = f"跨标签响应率无显著差异（Kruskal-Wallis p={p_kw:.4f}），整体中位数响应率为 {median_rate:.3f}。"

    # Check response_rate bounds
    rr_bounds_ok = stats_df["response_rate"].min() >= 0.0 and stats_df["response_rate"].max() <= 1.0

    lines = [
        "# 子任务 H：弹幕空间中的“提问-解答”微观社区生态模型",
        "",
        "## 数据说明",
        f"- 使用 `danmaku_core.parquet` 与视频数据。",
        f"- 筛选 `stat_danmaku >= 200` 的视频，共 {len(video_df)} 条。",
        f"- 语义匹配模型：`{model_name}`（Sentence-BERT，{device.upper()} batch encode，batch={encode_batch}）。",
        "",
        "## 数据处理",
        "- Q 识别：关键词规则（怎么、为什么、求问、不懂、请问、疑问、如何、咋办、求解）。",
        "- A 候选窗口：progress 落在提问弹幕的 [-5秒, +15秒] 区间内，且发送时间严格满足 `A.ctime > Q.ctime` 并在 15 秒内。",
        "- A 确认：废除回复句式规则捷径，所有候选必须经过 Sentence-BERT 语义相似度检验，且阈值必须严格大于噪音基线 95 分位数。",
        f"- **噪音基线校准**：在全库随机抽取 1000 对无关弹幕，95 分位数相似度为 **{random_95th:.4f}**。真实 QA 匹配阈值 > {random_95th:.4f}。",
        "- **Q=A 过滤**：内容完全相同的候选直接跳过，避免自问自答。",
        f"- Pilot 样本：前 30 对高相似度候选已保存至 `{os.path.join(OUTPUT_DIR, 'qa_pilot_sample.txt')}`。",
        "",
        "## 数据分析",
        f"- 提取到的总 QA 对数：{len(qa_df)}",
        f"- 各视频中位数响应率：{median_rate:.4f}",
        f"- Kruskal-Wallis 检验：H = {h_stat:.2f}, p-value = {p_kw:.4f}" if not np.isnan(p_kw) else "- Kruskal-Wallis 检验：样本不足，无法计算。",
        f"- 响应率范围校验：{'通过 [0,1]' if rr_bounds_ok else '警告：存在越界值'}",
        "",
        "## 验证",
        "- QA 对表与视频级统计表已保存至 `data/processed/step5/step5_H/`。",
        "- 响应率分布图与分标签对比图已保存。",
        "",
        "## 结论",
        f"- {conclusion}",
        "",
        "## 业务洞察",
    ]

    if branch == "A":
        lines.append("- 弹幕不仅是情绪工具，更是高效的分布式知识注脚系统。")
        lines.append("- 向平台建议：开发“弹幕高亮追问”或“问题悬赏弹幕”功能，将自发互助行为产品化，提升学习社区壁垒。")
    else:
        lines.append("- 弹幕中充斥问题但无人作答，或者回答与问题无关。")
        lines.append("- 指导 UP 主：不要指望观众自治。应定期挖掘视频弹幕中的“未解答高频 Q”，作为下一期视频的选题来源。")

    lines.extend([
        "",
        "## 局限与未来方向",
        "- `user_hash` 仅为视频内唯一标识，无法追踪全网“热心解答者”的专家画像。未来若能获取全局用户 ID，可构建异质图神经网络挖掘“隐形课代表”。",
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[step5_H] Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run()

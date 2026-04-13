import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import ttest_1samp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.step5_utils.plot_style import setup_fonts, savefig_safe

BASE_DIR = "D:/Claude_Code/bilibili-ai-tag-analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/step5/step5_F")
REPORT_PATH = os.path.join(BASE_DIR, "results/step5/step5_F_ai_event_impact_report.md")
FIG_DIR = os.path.join(BASE_DIR, "results/step5/figures")

EVENTS = [
    {"name": "DeepSeek-R1发布", "date": "2025-02-03"},
    {"name": "DeepSeek登顶美区AppStore", "date": "2025-02-17"},
    {"name": "Sora发布", "date": "2024-02-19"},
    {"name": "文心一言发布", "date": "2024-06-17"},
    {"name": "豆包年末更新", "date": "2024-12-23"},
    {"name": "豆包年度更新", "date": "2025-12-29"},
]

SPRING_FESTIVAL_RANGES = [
    ("2024-02-09", "2024-02-17"),
    ("2025-01-28", "2025-02-04"),
]

def is_near_spring_festival(event_date):
    event_dt = pd.to_datetime(event_date)
    for start, end in SPRING_FESTIVAL_RANGES:
        if pd.to_datetime(start) <= event_dt <= pd.to_datetime(end) + pd.Timedelta(days=3):
            return True
    return False

def format_date_axis(ax, dates):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    if len(dates) > 0:
        first_year = pd.to_datetime(dates.min()).year
        ax.text(0.02, -0.18, f"{str(first_year)[-2:]}年", transform=ax.transAxes,
                fontsize=10, ha='left', va='top')

def event_study(df_daily, event_date, event_name, est_pre=-30, est_post=-10, evt_pre=-2, evt_post=7):
    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    event_dt = pd.to_datetime(event_date)

    est_mask = (df["date"] >= event_dt + pd.Timedelta(days=est_pre)) & (df["date"] <= event_dt + pd.Timedelta(days=est_post))
    evt_mask = (df["date"] >= event_dt + pd.Timedelta(days=evt_pre)) & (df["date"] <= event_dt + pd.Timedelta(days=evt_post))

    est = df.loc[est_mask].copy()
    evt = df.loc[evt_mask].copy()

    if len(est) < 5 or len(evt) < 3:
        return None

    est = est.sort_values("date").reset_index(drop=True)
    est["time_idx"] = np.arange(len(est))
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    X = sm.add_constant(est["time_idx"])
    model = OLS(est["mean_anxiety"], X).fit()

    evt = evt.sort_values("date").reset_index(drop=True)
    evt["time_idx"] = np.arange(len(evt)) + len(est)
    X_evt = sm.add_constant(evt["time_idx"])
    pred = model.predict(X_evt)
    evt["predicted"] = pred
    evt["abnormal"] = evt["mean_anxiety"] - evt["predicted"]
    evt["cum_abnormal"] = evt["abnormal"].cumsum()

    t_stat, p_value = ttest_1samp(evt["abnormal"].dropna(), 0)

    # Cohen's d
    abnormal_vals = evt["abnormal"].dropna().values
    cohen_d = abnormal_vals.mean() / abnormal_vals.std(ddof=1) if len(abnormal_vals) > 1 and abnormal_vals.std(ddof=1) > 0 else np.nan

    near_cny = is_near_spring_festival(event_date)

    return {
        "event_name": event_name,
        "event_date": event_date,
        "est_n": len(est),
        "evt_n": len(evt),
        "mean_abnormal": evt["abnormal"].mean(),
        "max_cum_abnormal": evt["cum_abnormal"].max(),
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": cohen_d,
        "near_spring_festival": near_cny,
    }, evt

def run():
    setup_fonts()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    dmk_path = os.path.join(BASE_DIR, "data/processed/step5/shared/danmaku_daily_metrics.parquet")
    df_daily = pd.read_parquet(dmk_path)
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    results = []
    for ev in EVENTS:
        out = event_study(df_daily, ev["date"], ev["name"])
        if out is None:
            continue
        res, evt = out
        results.append(res)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(evt["date"], evt["cum_abnormal"], marker="o", color="coral")
        event_dt = pd.to_datetime(ev["date"])
        ax.axvline(event_dt, color="red", linestyle="--", label="事件日")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title(f"{ev['name']} 累积异常焦虑 (CAR)")
        ax.set_ylabel("累积异常焦虑值")
        ax.legend()
        format_date_axis(ax, evt["date"])
        fig.tight_layout()
        savefig_safe(fig, os.path.join(FIG_DIR, f"step5_F_car_{ev['name']}.png".replace("/", "_")))

    res_df = pd.DataFrame(results)
    res_df.to_parquet(os.path.join(OUTPUT_DIR, "event_study_results.parquet"), index=False)

    valid = res_df.dropna(subset=["p_value"])
    significant = valid[valid["p_value"] < 0.05]
    n_pos = (significant["mean_abnormal"] > 0).sum()
    n_neg = (significant["mean_abnormal"] < 0).sum()

    if n_pos >= 2:
        branch = "A"
        conclusion = f"在 {len(significant)} 个显著事件中，{n_pos} 个显示事件后焦虑值显著上升，存在脉冲式冲击。"
    elif len(significant) == 0:
        branch = "C"
        conclusion = "未检测到 AI 重大事件对 B站弹幕焦虑情绪的显著影响。"
    else:
        branch = "mixed"
        conclusion = f"结果混杂：{n_pos} 个正向显著，{n_neg} 个负向显著。"

    lines = [
        "# 子任务 F：重大 AI 事件对社会心理（弹幕情感）的冲击效应",
        "",
        "## 数据说明",
        "- 使用 `data/processed/step5/shared/danmaku_daily_metrics.parquet` 中的日级平均焦虑率。",
        "- 事件定义来源于 `results/baidu_index_event_report.md`。",
        "",
        "## 数据处理",
        "- 估计窗：[-30, -10] 天，拟合线性基线。",
        "- 事件窗：[-2, +7] 天，计算异常焦虑值（实际 - 预测）。",
        "- 单样本 t 检验验证事件窗内异常焦虑是否显著异于 0。",
        "- **Cohen's d**：计算显著事件的异常焦虑效应量 `mean(abnormal) / std(abnormal)`。",
        "",
        "## 数据分析",
        "### 事件研究结果",
    ]

    if not valid.empty:
        lines.append(valid[["event_name", "mean_abnormal", "max_cum_abnormal", "t_stat", "p_value", "cohens_d", "near_spring_festival"]].to_markdown(index=False))
    else:
        lines.append("- 无有效事件研究回归结果。")

    cny_events = valid[valid["near_spring_festival"] == True]["event_name"].tolist()
    if cny_events:
        lines.append(f"- 注：{', '.join(cny_events)} 发生在春节前后，基线可能受到节假日用户结构变化的系统性偏移。")

    lines.extend([
        "",
        "## 验证",
        "- CAR（累积异常焦虑）图已保存至 `results/step5/figures/`。",
        "",
        "## 结论",
        f"- {conclusion}",
        "",
        "## 业务洞察",
    ])

    if branch == "A":
        lines.append("- 证实了“技术焦虑的脉冲式爆发规律”。当重大 AI 事件发生时，平台算法应同步放量推送“AI 变现/提效”类实用视频，以抚平焦虑情绪。")
        lines.append("- **时效建议**：建议高权重标签下的 UP 主在事件爆发后 48 小时内，将选题从“吃瓜解析”快速转向“工具化/教程化”，抢占流量并缓解焦虑。")
    elif branch == "C":
        lines.append("- 用户对“颠覆性 AI”的宏大叙事已产生免疫。UP 主应放弃空泛吹捧/贩卖焦虑，回归微观、实用的手把手教学。")
    else:
        lines.append("- 不同事件的情绪冲击方向不一致，建议平台根据事件类型（技术突破 vs 产品发布）动态调整内容推荐策略。")
        lines.append("- **时效建议**：建议高权重标签下的 UP 主在事件爆发后 48 小时内，将选题从“吃瓜解析”快速转向“工具化/教程化”，抢占流量并缓解焦虑。")

    lines.extend([
        "",
        "## 局限与未来方向",
        "- 当前焦虑词表较为粗糙。未来应引入基于 Ekman 六大基本情绪的细粒度情绪图谱，区分恐惧、惊奇、愤怒等维度。",
        "- 春节期间的用户画像偏移（学生党比例上升）可能对焦虑基线产生干扰，事件研究法的结果应谨慎外推到非节假日时段。",
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[step5_F] Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run()

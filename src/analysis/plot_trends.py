"""
百度指数趋势可视化与峰值分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(filepath):
    """加载百度指数数据"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def detect_peaks(df, keyword, window=7, threshold=2.0):
    """
    使用Z-Score方法检测峰值

    Parameters:
    -----------
    df : DataFrame
        百度指数数据
    keyword : str
        关键词
    window : int
        移动平均窗口大小
    threshold : float
        Z-Score阈值，高于此值视为峰值

    Returns:
    --------
    peaks : DataFrame
        峰值数据
    """
    # 筛选关键词数据
    kw_data = df[df['keyword'] == keyword].copy().sort_values('date')

    if len(kw_data) == 0:
        return pd.DataFrame()

    # 计算移动平均和标准差
    kw_data['ma'] = kw_data['index'].rolling(window=window, center=True, min_periods=1).mean()
    kw_data['std'] = kw_data['index'].rolling(window=window, center=True, min_periods=1).std()

    # 计算Z-Score
    kw_data['zscore'] = (kw_data['index'] - kw_data['ma']) / (kw_data['std'] + 1e-10)

    # 识别峰值（Z-Score超过阈值且为局部最大值）
    kw_data['is_peak'] = (kw_data['zscore'] > threshold) & \
                         (kw_data['index'] > kw_data['index'].shift(1)) & \
                         (kw_data['index'] > kw_data['index'].shift(-1))

    peaks = kw_data[kw_data['is_peak']].copy()

    # 合并相邻峰值（3天内）
    if len(peaks) > 1:
        merged_peaks = []
        current_peak = peaks.iloc[0]

        for i in range(1, len(peaks)):
            next_peak = peaks.iloc[i]
            days_diff = (next_peak['date'] - current_peak['date']).days

            if days_diff <= 3:
                # 保留较高的峰值
                if next_peak['index'] > current_peak['index']:
                    current_peak = next_peak
            else:
                merged_peaks.append(current_peak)
                current_peak = next_peak

        merged_peaks.append(current_peak)
        peaks = pd.DataFrame(merged_peaks)

    return peaks[['date', 'keyword', 'index', 'zscore']]


def get_event_annotation(date_str):
    """
    根据日期返回对应的事件标注
    """
    events = {
        # Sora 相关
        "2024-02-15": "Sora发布",
        "2024-02-16": "Sora发布",
        "2024-02-14": "Sora发布",
        "2024-03-15": "Sora技术报告",

        # ChatGPT/OpenAI 相关
        "2024-05-14": "GPT-4o发布",
        "2024-05-13": "GPT-4o发布",
        "2024-05-15": "GPT-4o发布",
        "2024-09-13": "OpenAI o1发布",
        "2024-09-12": "OpenAI o1发布",
        "2024-09-14": "OpenAI o1发布",

        # DeepSeek 相关
        "2024-12-26": "DeepSeek-V3发布",
        "2025-01-20": "DeepSeek-R1开源",
        "2025-01-19": "DeepSeek-R1开源",
        "2025-01-21": "DeepSeek-R1开源",
        "2025-01-27": "DeepSeek登顶美区App Store",
        "2025-01-26": "DeepSeek登顶美区App Store",
        "2025-01-28": "DeepSeek登顶美区App Store",

        # 行业大会
        "2024-03-18": "英伟达GTC大会",
        "2024-03-19": "英伟达GTC大会",
        "2024-07-04": "WAIC大会",
        "2024-09-10": "阿里巴巴云栖大会",

        # AI焦虑相关
        "2024-03-28": "AI取代工作讨论",
    }

    # 尝试精确匹配
    if date_str in events:
        return events[date_str]

    # 尝试前后1天匹配
    date = datetime.strptime(date_str, "%Y-%m-%d")
    for delta in [-1, 1]:
        nearby_date = (date + pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
        if nearby_date in events:
            return events[nearby_date]

    return None


def plot_trends(df, output_dir):
    """
    绘制各关键词趋势图并标注峰值事件
    """
    keywords = df['keyword'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(keywords)))

    # 创建大图
    fig, axes = plt.subplots(len(keywords), 1, figsize=(16, 20))
    fig.suptitle('百度指数趋势分析 (2024-01 至 2026-01)', fontsize=16, fontweight='bold')

    all_peaks_info = []

    for idx, (keyword, color) in enumerate(zip(keywords, colors)):
        ax = axes[idx] if len(keywords) > 1 else axes

        # 获取关键词数据
        kw_data = df[df['keyword'] == keyword].sort_values('date')

        # 绘制趋势线
        ax.plot(kw_data['date'], kw_data['index'], color=color, linewidth=1.5, label=keyword)

        # 检测峰值
        peaks = detect_peaks(df, keyword, window=7, threshold=1.5)

        # 标注峰值
        for _, peak in peaks.iterrows():
            date_str = peak['date'].strftime('%Y-%m-%d')
            event = get_event_annotation(date_str)

            if event:
                # 绘制峰值点
                ax.scatter(peak['date'], peak['index'], color='red', s=100, zorder=5)

                # 添加标注
                ax.annotate(
                    event,
                    xy=(peak['date'], peak['index']),
                    xytext=(10, 20),
                    textcoords='offset points',
                    fontsize=8,
                    ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1)
                )

                all_peaks_info.append({
                    'date': date_str,
                    'keyword': keyword,
                    'index': int(peak['index']),
                    'event': event
                })

        # 设置图表属性
        ax.set_title(f'{keyword}', fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel('搜索指数', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # 添加统计信息
        mean_val = kw_data['index'].mean()
        max_val = kw_data['index'].max()
        ax.axhline(y=mean_val, color='gray', linestyle='--', alpha=0.5, label=f'均值: {mean_val:.0f}')

    plt.xlabel('日期', fontsize=10)
    plt.tight_layout()

    # 保存图表
    output_path = Path(output_dir) / 'baidu_trend_with_events.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")

    plt.close()

    return all_peaks_info


def plot_comparison(df, output_dir):
    """
    绘制所有关键词的对比图
    """
    plt.figure(figsize=(16, 8))

    keywords = ['人工智能', 'AI', 'ChatGPT', 'DeepSeek', 'Sora', '机器学习', 'AI焦虑']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for keyword, color in zip(keywords, colors):
        kw_data = df[df['keyword'] == keyword].sort_values('date')
        if len(kw_data) > 0:
            plt.plot(kw_data['date'], kw_data['index'], label=keyword, color=color, linewidth=1.5)

    plt.title('百度指数趋势对比 (2024-01 至 2026-01)', fontsize=14, fontweight='bold')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('搜索指数', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)

    # 设置x轴格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()

    output_path = Path(output_dir) / 'baidu_trend_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {output_path}")

    plt.close()


def analyze_peak_duration(df, peaks_info):
    """
    分析每个峰值的热度持续时间
    """
    results = []

    for peak in peaks_info:
        keyword = peak['keyword']
        peak_date = pd.to_datetime(peak['date'])

        # 获取该关键词的数据
        kw_data = df[df['keyword'] == keyword].sort_values('date')

        # 计算该关键词的平均值作为基准
        baseline = kw_data['index'].mean()

        # 向前查找起始点（超过基准的1.2倍）
        start_date = peak_date
        for i in range(30):  # 最多向前查找30天
            check_date = peak_date - pd.Timedelta(days=i)
            row = kw_data[kw_data['date'] == check_date]
            if len(row) > 0 and row['index'].values[0] < baseline * 1.2:
                start_date = check_date + pd.Timedelta(days=1)
                break

        # 向后查找结束点（回落到基准的1.2倍以下）
        end_date = peak_date
        for i in range(30):  # 最多向后查找30天
            check_date = peak_date + pd.Timedelta(days=i)
            row = kw_data[kw_data['date'] == check_date]
            if len(row) > 0 and row['index'].values[0] < baseline * 1.2:
                end_date = check_date - pd.Timedelta(days=1)
                break

        duration = (end_date - start_date).days + 1

        results.append({
            'date': peak['date'],
            'keyword': peak['keyword'],
            'event': peak['event'],
            'peak_index': peak['index'],
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'duration_days': duration
        })

    return results


def main():
    """主函数"""
    data_path = Path("D:/Claude_Code/bilibili-ai-tag-analysis/data/raw/baidu_index_2401_2601.csv")
    output_dir = Path("D:/Claude_Code/bilibili-ai-tag-analysis/results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载数据...")
    df = load_data(data_path)

    print("绘制趋势图...")
    peaks_info = plot_trends(df, output_dir)

    print("绘制对比图...")
    plot_comparison(df, output_dir)

    print("分析峰值持续时间...")
    peak_durations = analyze_peak_duration(df, peaks_info)

    # 保存峰值信息
    peaks_df = pd.DataFrame(peak_durations)
    peaks_csv_path = Path("D:/Claude_Code/bilibili-ai-tag-analysis/data/raw/baidu_peaks_analysis.csv")
    peaks_df.to_csv(peaks_csv_path, index=False, encoding='utf-8')
    print(f"峰值分析已保存至: {peaks_csv_path}")

    print(f"\n共识别 {len(peaks_info)} 个显著峰值")
    return peaks_info, peak_durations


if __name__ == "__main__":
    main()

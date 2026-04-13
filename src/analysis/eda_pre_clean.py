"""
第3步：数据预处理前的探索性分析 (Pre-clean EDA)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('results/figures', exist_ok=True)

# 读取数据
df = pd.read_csv('data/raw/bilibili_video_info.csv', encoding='utf-8')

# 基本统计
total_records = len(df)
unique_videos = df['bvid'].nunique()

print(f"数据总量: {total_records} 条")
print(f"去重后视频数: {unique_videos} 条")
print(f"重复视频数: {total_records - unique_videos} 条")

# ========== EDA 1: 按 search_tag 统计 ==========
print("\n=== EDA 1: 按 search_tag 统计 ===")
search_tag_counts = df.groupby('search_tag')['bvid'].nunique().sort_values(ascending=False)
print(search_tag_counts)

# 绘制 search_tag 分布图
fig, ax = plt.subplots(figsize=(12, 6))
search_tag_counts.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('各搜索标签视频数量分布', fontsize=14)
ax.set_xlabel('搜索标签', fontsize=12)
ax.set_ylabel('视频数量', fontsize=12)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('results/figures/search_tag_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ========== EDA 2: 视频真实标签统计 (tags) ==========
print("\n=== EDA 2: 视频真实标签统计 ===")

# 处理 tags 字段，拆分并平铺
all_tags = []
for tags_str in df['tags'].dropna():
    # 以 | 或逗号分隔
    if '|' in str(tags_str):
        tags_list = [t.strip() for t in str(tags_str).split('|') if t.strip()]
    else:
        tags_list = [t.strip() for t in str(tags_str).split(',') if t.strip()]
    all_tags.extend(tags_list)

tag_counter = Counter(all_tags)
print(f"独立标签总数: {len(tag_counter)}")
print(f"标签出现总次数: {sum(tag_counter.values())}")

# Top 20 标签
top_20_tags = tag_counter.most_common(20)
others_count = sum(count for _, count in tag_counter.most_common()[20:])

print("\nTop 20 标签:")
for tag, count in top_20_tags:
    print(f"  {tag}: {count}")
print(f"  Others: {others_count}")

# 可视化 Top 20 + Others
top_20_names = [tag for tag, _ in top_20_tags]
top_20_counts = [count for _, count in top_20_tags]
top_20_names.append('Others')
top_20_counts.append(others_count)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(top_20_names)))
bars = ax.barh(range(len(top_20_names)), top_20_counts, color=colors)
ax.set_yticks(range(len(top_20_names)))
ax.set_yticklabels(top_20_names)
ax.invert_yaxis()
ax.set_xlabel('出现频次', fontsize=12)
ax.set_title('视频标签分布 (Top 20 + Others)', fontsize=14)

# 添加数值标注
for i, (bar, count) in enumerate(zip(bars, top_20_counts)):
    ax.text(count + max(top_20_counts)*0.01, i, str(count),
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/tag_distribution_top20.png', dpi=150, bbox_inches='tight')
plt.close()

# ========== EDA 3: 播放量分箱统计 ==========
print("\n=== EDA 3: 播放量分箱统计 ===")

# 去重后统计
df_unique = df.drop_duplicates(subset=['bvid']).copy()
views = df_unique['stat_view']

# 分箱
bins = [0, 50000, 200000, 1000000, float('inf')]
labels = ['< 5万', '5万-20万', '20万-100万', '> 100万']
df_unique['view_bin'] = pd.cut(views, bins=bins, labels=labels, right=False)

view_bin_counts = df_unique['view_bin'].value_counts().sort_index()
view_bin_pct = (view_bin_counts / len(df_unique) * 100).round(2)

print("播放量分箱统计:")
for label, count, pct in zip(labels, view_bin_counts, view_bin_pct):
    print(f"  {label}: {count} 条 ({pct}%)")

# 检查 <5万 的数据
if view_bin_counts.iloc[0] > 0:
    print(f"\n[WARNING] 发现 {view_bin_counts.iloc[0]} 条播放量 <5万的视频")
    print("  标记为: 异常漏网/早期规则变动数据")

# 可视化播放量分布
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
bars = ax.bar(labels, view_bin_counts, color=colors, edgecolor='white', linewidth=1.5)
ax.set_xlabel('播放量区间', fontsize=12)
ax.set_ylabel('视频数量', fontsize=12)
ax.set_title('播放量层级分布', fontsize=14)

# 添加数值和百分比标注
for bar, count, pct in zip(bars, view_bin_counts, view_bin_pct):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct}%)',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/view_distribution_bins.png', dpi=150, bbox_inches='tight')
plt.close()

# 保存统计数据
stats = {
    'total_records': total_records,
    'unique_videos': unique_videos,
    'duplicate_records': total_records - unique_videos,
    'search_tag_counts': search_tag_counts.to_dict(),
    'top_20_tags': top_20_tags,
    'others_count': others_count,
    'total_unique_tags': len(tag_counter),
    'view_bins': {label: {'count': int(count), 'pct': float(pct)}
                  for label, count, pct in zip(labels, view_bin_counts, view_bin_pct)}
}

print("\n=== EDA 完成 ===")
print("图表已保存至 results/figures/")

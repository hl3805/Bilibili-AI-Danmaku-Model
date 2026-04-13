"""
第4步：数据深度清理与探索性分析 (Clean & EDA)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from collections import Counter
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 2.1 数据清理与双向匹配 ==========
print("=== 2.1 数据清理与双向匹配 ===")

# 1. 读取视频基础表
df_video = pd.read_csv('data/raw/bilibili_video_info.csv', encoding='utf-8')
print(f"原始视频数据: {len(df_video)} 条")

# 2. 严格去重（以bvid为基准）
df_video = df_video.drop_duplicates(subset=['bvid'])
print(f"去重后视频数据: {len(df_video)} 条")

# 3. 阈值硬过滤
# 剔除播放量 < 50,000
# 剔除弹幕量 < 50
df_video = df_video[(df_video['stat_view'] >= 50000) & (df_video['stat_danmaku'] >= 50)]
print(f"阈值过滤后(播放>=5万, 弹幕>=50): {len(df_video)} 条")

# 4. 弹幕文件扫描，提取BVID
danmaku_files = glob.glob('data/raw/danmaku/*.csv')
danmaku_bvids = set()
for f in danmaku_files:
    # 提取文件名（去掉路径和.csv后缀）
    bvid = os.path.basename(f).replace('.csv', '')
    danmaku_bvids.add(bvid)
print(f"弹幕文件数量: {len(danmaku_bvids)} 个")

# 5. 双向交集匹配
video_bvids = set(df_video['bvid'].tolist())
valid_bvids = video_bvids & danmaku_bvids
print(f"双向匹配后有效BVID: {len(valid_bvids)} 个")

# 仅保留有对应弹幕文件的视频
df_video_cleaned = df_video[df_video['bvid'].isin(valid_bvids)].copy()
print(f"清洗后视频数据: {len(df_video_cleaned)} 条")

# ========== 2.2 数据安全落盘 ==========
print("\n=== 2.2 数据安全落盘 ===")

# 创建目录
os.makedirs('cleaned-data/danmaku', exist_ok=True)

# 保存清洗后的视频数据
df_video_cleaned.to_csv('cleaned-data/bilibili_video_cleaned.csv', index=False, encoding='utf-8')
print("视频数据已保存至: cleaned-data/bilibili_video_cleaned.csv")

# 复制匹配的弹幕文件
copied_count = 0
for bvid in valid_bvids:
    src = f'data/raw/danmaku/{bvid}.csv'
    dst = f'cleaned-data/danmaku/{bvid}.csv'
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied_count += 1
print(f"弹幕文件已复制: {copied_count} 个")

# ========== 2.3 探索性数据分析 ==========
print("\n=== 2.3 探索性数据分析 ===")

# 1. 基础概览
print("\n--- 基础概览 ---")
print("字段说明:")
print("  bvid: 视频唯一标识")
print("  title: 视频标题")
print("  pubdate: 发布时间")
print("  stat_view: 播放量")
print("  stat_danmaku: 弹幕数")
print("  stat_reply: 评论数")
print("  stat_like: 点赞数")
print("  owner_fans: UP主粉丝数")
print("  tags: 视频标签(竖线分隔)")
print("  search_tag: 采集所用搜索标签")
print(f"\n有效视频总量: {len(df_video_cleaned)} 条")
print("\n前5条样本:")
print(df_video_cleaned[['bvid', 'title', 'stat_view', 'stat_danmaku', 'tags']].head())

# 2. 播放量阶梯分布分析
print("\n--- 播放量阶梯分布 ---")
views = df_video_cleaned['stat_view']
bins = [0, 50000, 200000, 1000000, float('inf')]
labels = ['< 5万', '5万-20万', '20万-100万', '> 100万']
df_video_cleaned['view_bin'] = pd.cut(views, bins=bins, labels=labels, right=False)

view_bin_counts = df_video_cleaned['view_bin'].value_counts().sort_index()
view_bin_pct = (view_bin_counts / len(df_video_cleaned) * 100).round(2)

print("播放量分箱统计:")
for label, count, pct in zip(labels, view_bin_counts, view_bin_pct):
    print(f"  {label}: {count} 条 ({pct}%)")

# 可视化播放量分布
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
bars = ax.bar(labels, view_bin_counts, color=colors, edgecolor='white', linewidth=1.5)
ax.set_xlabel('播放量区间', fontsize=12)
ax.set_ylabel('视频数量', fontsize=12)
ax.set_title('播放量层级分布 (清洗后)', fontsize=14)

for bar, count, pct in zip(bars, view_bin_counts, view_bin_pct):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct}%)',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/cleaned-view_distribution_bins.png', dpi=150, bbox_inches='tight')
plt.close()
print("播放量分布图已保存")

# 3. 标签大盘统计
print("\n--- 标签大盘统计 ---")

# 无标签视频数量
no_tags_count = df_video_cleaned['tags'].isna().sum() + (df_video_cleaned['tags'].str.strip() == '').sum()
print(f"无标签视频数量: {no_tags_count} 条")

# 提取所有标签并展平统计
all_tags = []
for tags_str in df_video_cleaned['tags'].dropna():
    if '|' in str(tags_str):
        tags_list = [t.strip() for t in str(tags_str).split('|') if t.strip()]
    else:
        tags_list = [t.strip() for t in str(tags_str).split(',') if t.strip()]
    all_tags.extend(tags_list)

tag_counter = Counter(all_tags)
print(f"独立标签总数: {len(tag_counter)}")

# Top 30 标签
top_30_tags = tag_counter.most_common(30)
top_30_tag_names = [tag for tag, _ in top_30_tags]
print("\nTop 30 标签明细:")
for i, (tag, count) in enumerate(top_30_tags, 1):
    print(f"  {i}. {tag}: {count}")

# Others视频数量：没有任何标签属于Top 30的视频
def has_top30_tag(tags_str):
    if pd.isna(tags_str) or str(tags_str).strip() == '':
        return False
    if '|' in str(tags_str):
        video_tags = [t.strip() for t in str(tags_str).split('|') if t.strip()]
    else:
        video_tags = [t.strip() for t in str(tags_str).split(',') if t.strip()]
    return len(set(video_tags) & set(top_30_tag_names)) > 0

df_video_cleaned['has_top30_tag'] = df_video_cleaned['tags'].apply(has_top30_tag)
others_video_count = (~df_video_cleaned['has_top30_tag']).sum()
print(f"\nOthers视频数量(无Top30标签): {others_video_count} 条")

# 可视化 Top 30 + Others
top_30_names = [tag for tag, _ in top_30_tags]
top_30_counts = [count for _, count in top_30_tags]
top_30_names.append('Others')
top_30_counts.append(others_video_count)

fig, ax = plt.subplots(figsize=(12, 10))
colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(top_30_names)))
bars = ax.barh(range(len(top_30_names)), top_30_counts, color=colors)
ax.set_yticks(range(len(top_30_names)))
ax.set_yticklabels(top_30_names)
ax.invert_yaxis()
ax.set_xlabel('出现频次', fontsize=12)
ax.set_title('视频标签分布 (Top 30 + Others)', fontsize=14)

for i, (bar, count) in enumerate(zip(bars, top_30_counts)):
    ax.text(count + max(top_30_counts)*0.01, i, str(count),
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/cleaned-tag_distribution_top30.png', dpi=150, bbox_inches='tight')
plt.close()
print("标签分布图已保存")

# 4. 标签共现与头部视频洞察
print("\n--- 标签共现与头部视频洞察 ---")

cooccurrence_results = []

for tag_name in top_30_tag_names:
    # 获取包含该标签的视频
    def has_tag(tags_str, target=tag_name):
        if pd.isna(tags_str):
            return False
        if '|' in str(tags_str):
            tags_list = [t.strip() for t in str(tags_str).split('|') if t.strip()]
        else:
            tags_list = [t.strip() for t in str(tags_str).split(',') if t.strip()]
        return target in tags_list

    mask = df_video_cleaned['tags'].apply(has_tag)
    tag_videos = df_video_cleaned[mask]

    if len(tag_videos) == 0:
        continue

    tag_count = len(tag_videos)

    # 共现分析：统计与该标签共现的其他标签
    cooccurrence_counter = Counter()
    for tags_str in tag_videos['tags'].dropna():
        if '|' in str(tags_str):
            tags_list = [t.strip() for t in str(tags_str).split('|') if t.strip()]
        else:
            tags_list = [t.strip() for t in str(tags_str).split(',') if t.strip()]
        for t in tags_list:
            if t != tag_name:
                cooccurrence_counter[t] += 1

    # 计算百分比
    cooccurrence_pct = [(t, count, count/tag_count*100) for t, count in cooccurrence_counter.items()]
    cooccurrence_pct.sort(key=lambda x: x[2], reverse=True)

    # 分级输出
    high_co = [f"{t}({pct:.1f}%)" for t, c, pct in cooccurrence_pct if pct > 50]
    mid_co = [f"{t}({pct:.1f}%)" for t, c, pct in cooccurrence_pct if 20 <= pct <= 50]
    low_co = [f"{t}({pct:.1f}%)" for t, c, pct in cooccurrence_pct if 10 <= pct < 20]

    # 头部视频
    top3_view = tag_videos.nlargest(3, 'stat_view')[['bvid', 'stat_view', 'stat_danmaku', 'pubdate']]
    top3_danmaku = tag_videos.nlargest(3, 'stat_danmaku')[['bvid', 'stat_view', 'stat_danmaku', 'pubdate']]

    cooccurrence_results.append({
        'tag': tag_name,
        'count': tag_count,
        'high_co': high_co,
        'mid_co': mid_co,
        'low_co': low_co,
        'top3_view': top3_view,
        'top3_danmaku': top3_danmaku
    })

# 输出共现分析结果
print("\n标签共现分析结果:")
for result in cooccurrence_results[:5]:  # 先输出前5个作为预览
    print(f"\n【{result['tag']}】 ({result['count']}条视频)")
    if result['high_co']:
        print(f"  >50%共现: {', '.join(result['high_co'][:5])}")
    if result['mid_co']:
        print(f"  20-49%共现: {', '.join(result['mid_co'][:5])}")

print("\n=== EDA 完成 ===")

# 保存统计结果供报告生成使用
import json
eda_summary = {
    'total_videos': len(df_video_cleaned),
    'view_bins': {label: {'count': int(count), 'pct': float(pct)}
                  for label, count, pct in zip(labels, view_bin_counts, view_bin_pct)},
    'no_tags_count': int(no_tags_count),
    'top_30_tags': top_30_tags,
    'others_video_count': int(others_video_count),
    'cooccurrence': [{k: v if k not in ['top3_view', 'top3_danmaku'] else v.to_dict('records')
                     for k, v in r.items()} for r in cooccurrence_results]
}
with open('cleaned-data/eda_summary.json', 'w', encoding='utf-8') as f:
    json.dump(eda_summary, f, ensure_ascii=False, indent=2)
print("统计结果已保存至: cleaned-data/eda_summary.json")

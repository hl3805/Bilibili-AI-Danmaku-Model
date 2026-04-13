"""
B站视频与弹幕核心数据采集爬虫
支持断点续传、实时保存、按比例采样
"""

import asyncio
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

# 设置编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import aiohttp
import aiofiles

# 尝试导入bilibili_api
try:
    from bilibili_api import search, video, user
    from bilibili_api.exceptions import ResponseCodeException, NetworkException
    BILI_API_AVAILABLE = True
except ImportError:
    BILI_API_AVAILABLE = False
    print("警告: bilibili-api-python 未安装，将使用备用方案")


# ============ 配置常量 ============
DATA_DIR = Path("D:/Claude_Code/bilibili-ai-tag-analysis/data/raw")
DANMAKU_DIR = DATA_DIR / "danmaku"
PROGRESS_FILE = DATA_DIR / "crawler_progress.json"
VIDEO_INFO_FILE = DATA_DIR / "bilibili_video_info.csv"

# 目标标签列表
TARGET_TAGS = [
    "人工智能",
    "科技",
    "机器学习",
    "AI",
    "ChatGPT",
    "Sora",
    "DeepSeek",
    "AI焦虑",
    "AI绘画",
    "AI编程"
]

# 采集参数
MIN_VIEW = 100000  # 最小播放量
MIN_DANMAKU = 200  # 最小弹幕数
TARGET_COUNT_PER_TAG = 1000  # 每个标签目标数量
MAX_CONCURRENT = 3  # 最大并发数
REQUEST_DELAY = (1, 3)  # 请求延迟范围（秒）

# 时间窗口
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)


class BilibiliCrawler:
    """B站视频数据采集器（支持断点续传）"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.progress: Dict = {}
        self.collected_bvids: Set[str] = set()  # 已采集的BVID集合
        self.tag_counts: Dict[str, int] = {tag: 0 for tag in TARGET_TAGS}

        # 确保目录存在
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DANMAKU_DIR.mkdir(parents=True, exist_ok=True)

        # 加载进度
        self.load_progress()
        self.load_existing_data()

    def load_progress(self):
        """加载采集进度"""
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                    self.progress = json.load(f)
                print(f"✓ 已加载进度文件: {PROGRESS_FILE}")
            except Exception as e:
                print(f"✗ 加载进度文件失败: {e}")
                self.progress = {tag: {"page": 1, "count": 0} for tag in TARGET_TAGS}
        else:
            self.progress = {tag: {"page": 1, "count": 0} for tag in TARGET_TAGS}

    def save_progress(self):
        """保存采集进度"""
        try:
            with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"✗ 保存进度失败: {e}")

    def load_existing_data(self):
        """加载已采集的数据（用于断点续传）"""
        if VIDEO_INFO_FILE.exists():
            try:
                with open(VIDEO_INFO_FILE, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        bvid = row.get('bvid')
                        if bvid:
                            self.collected_bvids.add(bvid)
                            # 统计各标签数量
                            search_tag = row.get('search_tag', '')
                            if search_tag in self.tag_counts:
                                self.tag_counts[search_tag] += 1
                print(f"✓ 已加载现有数据: {len(self.collected_bvids)} 个视频")
                print(f"  各标签进度: {self.tag_counts}")
            except Exception as e:
                print(f"✗ 加载现有数据失败: {e}")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
        self.save_progress()

    async def random_delay(self):
        """随机延迟"""
        delay = random.uniform(*REQUEST_DELAY)
        await asyncio.sleep(delay)

    async def safe_api_call(self, func, max_retries=3, *args, **kwargs):
        """带重试机制的API调用"""
        for attempt in range(max_retries):
            try:
                async with self.semaphore:
                    await self.random_delay()
                    return await func(*args, **kwargs)

            except ResponseCodeException as e:
                if "-799" in str(e) or "风控" in str(e):
                    wait_time = min((2 ** attempt) * 5, 60)  # 最大60秒
                    print(f"  [!] 风控触发，等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  [!] API错误: {e}，{2 ** attempt}秒后重试...")
                    await asyncio.sleep(2 ** attempt)

            except NetworkException as e:
                wait_time = 2 ** attempt
                print(f"  [!] 网络错误: {e}，{wait_time}秒后重试...")
                await asyncio.sleep(wait_time)

            except Exception as e:
                print(f"  [!] 未知错误: {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise

        return None

    async def search_videos(self, keyword: str, page: int = 1, page_size: int = 20) -> List[Dict]:
        """搜索视频"""
        if not BILI_API_AVAILABLE:
            print("✗ bilibili_api 不可用")
            return []

        try:
            # 使用 search_by_type 进行类型筛选
            result = await search.search_by_type(
                keyword=keyword,
                search_type=search.SearchObjectType.VIDEO,
                page=page
            )
            return result.get('result', [])
        except Exception as e:
            print(f"[!] search_by_type 失败: {e}, 尝试普通搜索...")
            try:
                # 备用方案：普通搜索
                result = await search.search(keyword=keyword)
                videos = result.get('videos', [])
                # 过滤只保留视频
                return [v for v in videos if v.get('type', '') == 'video']
            except Exception as e2:
                print(f"✗ 搜索失败: {e2}")
                return []

    async def get_video_info(self, bvid: str) -> Optional[Dict]:
        """获取视频详细信息"""
        if not BILI_API_AVAILABLE:
            return None

        try:
            v = video.Video(bvid=bvid)
            info = await v.get_info()
            return info
        except Exception as e:
            print(f"  [!] 获取视频信息失败 {bvid}: {e}")
            return None

    async def get_video_tags(self, bvid: str) -> List[str]:
        """获取视频标签"""
        if not BILI_API_AVAILABLE:
            return []

        try:
            v = video.Video(bvid=bvid)
            tags = await v.get_tags()
            return [tag.get('tag_name', '') for tag in tags if tag.get('tag_name')]
        except Exception as e:
            print(f"  [!] 获取标签失败 {bvid}: {e}")
            return []

    async def get_danmaku(self, bvid: str, cid: int) -> List[Dict]:
        """获取弹幕数据"""
        if not BILI_API_AVAILABLE:
            return []

        danmaku_list = []
        try:
            # 先创建Video对象，然后通过get_cid设置cid
            v = video.Video(bvid=bvid)
            # 设置cid
            v.set_cid(cid)
            page = 1
            while page <= 5:  # 最多获取5页弹幕
                try:
                    dms = await v.get_danmakus(page_index=page)
                    if not dms:
                        break

                    for dm in dms:
                        if hasattr(dm, 'text'):
                            danmaku_list.append({
                                'danmaku_id': getattr(dm, 'dmid', ''),
                                'content': getattr(dm, 'text', ''),
                                'progress': getattr(dm, 'dm_time', 0),
                                'ctime': getattr(dm, 'send_time', 0),
                                'user_hash': getattr(dm, 'crc32_id', '')
                            })

                    page += 1
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"  [!] 获取弹幕页失败: {e}")
                    break
        except Exception as e:
            print(f"  [!] 获取弹幕失败 {bvid}: {e}")

        return danmaku_list

    async def get_user_fans(self, mid: int) -> int:
        """获取UP主粉丝数"""
        if not BILI_API_AVAILABLE:
            return 0

        try:
            u = user.User(uid=mid)
            info = await u.get_user_info()
            return info.get('fans', 0)
        except Exception as e:
            print(f"  [!] 获取UP主信息失败 {mid}: {e}")
            return 0

    def filter_video(self, info: Dict) -> bool:
        """过滤视频（硬性标准）"""
        if not info:
            return False

        stat = info.get('stat', {})
        view = stat.get('view', 0)
        danmaku = stat.get('danmaku', 0)

        # 硬性过滤条件
        if view < MIN_VIEW:
            return False
        if danmaku < MIN_DANMAKU:
            return False

        return True

    async def process_video(self, bvid: str, search_tag: str) -> bool:
        """处理单个视频（采集信息+弹幕）"""
        # 检查是否已采集
        if bvid in self.collected_bvids:
            print(f"  [跳过] {bvid} 已采集")
            return True

        print(f"  [处理] {bvid} ...")

        # 1. 获取视频信息
        info = await self.get_video_info(bvid)
        if not info:
            return False

        # 2. 过滤检查
        if not self.filter_video(info):
            stat = info.get('stat', {})
            print(f"  [过滤] 不满足条件 (播放:{stat.get('view',0)}, 弹幕:{stat.get('danmaku',0)})")
            return False

        # 3. 获取标签
        tags = await self.get_video_tags(bvid)

        # 4. 获取UP主粉丝数
        owner_mid = info.get('owner', {}).get('mid', 0)
        fans = await self.get_user_fans(owner_mid) if owner_mid else 0

        # 5. 构建数据记录
        stat = info.get('stat', {})
        record = {
            'bvid': bvid,
            'aid': info.get('aid', ''),
            'cid': info.get('cid', ''),
            'title': info.get('title', '').replace('\n', ' ').replace('\r', ' '),
            'pubdate': datetime.fromtimestamp(info.get('pubdate', 0)).strftime('%Y-%m-%d %H:%M:%S'),
            'tname': info.get('tname', ''),
            'owner_mid': owner_mid,
            'owner_fans': fans,
            'duration': info.get('duration', 0),
            'desc': info.get('desc', '').replace('\n', ' ').replace('\r', ' ')[:500],
            'stat_view': stat.get('view', 0),
            'stat_danmaku': stat.get('danmaku', 0),
            'stat_reply': stat.get('reply', 0),
            'stat_like': stat.get('like', 0),
            'stat_coin': stat.get('coin', 0),
            'stat_favorite': stat.get('favorite', 0),
            'stat_share': stat.get('share', 0),
            'tags': '|'.join(tags),
            'search_tag': search_tag
        }

        # 6. 实时保存视频信息
        await self.save_video_info(record)

        # 7. 获取并保存弹幕
        cid = info.get('cid', 0)
        if cid:
            danmaku_list = await self.get_danmaku(bvid, cid)
            await self.save_danmaku(bvid, danmaku_list)
            print(f"  [弹幕] {len(danmaku_list)} 条")

        # 8. 更新进度
        self.collected_bvids.add(bvid)
        self.tag_counts[search_tag] += 1

        print(f"  [完成] {bvid} | 播放:{record['stat_view']} | 弹幕:{record['stat_danmaku']}")
        return True

    async def save_video_info(self, record: Dict):
        """实时保存视频信息到CSV"""
        file_exists = VIDEO_INFO_FILE.exists()

        async with asyncio.Lock():
            # 使用同步方式写入（CSV不支持真正的异步）
            with open(VIDEO_INFO_FILE, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)

    async def save_danmaku(self, bvid: str, danmaku_list: List[Dict]):
        """保存弹幕到单独文件"""
        if not danmaku_list:
            return

        danmaku_file = DANMAKU_DIR / f"{bvid}.csv"

        with open(danmaku_file, 'w', encoding='utf-8', newline='') as f:
            if danmaku_list:
                writer = csv.DictWriter(f, fieldnames=danmaku_list[0].keys())
                writer.writeheader()
                writer.writerows(danmaku_list)

    async def crawl_tag(self, tag: str):
        """采集单个标签的视频"""
        print(f"\n{'='*60}")
        print(f"开始采集标签: {tag}")
        print(f"{'='*60}")

        target_count = TARGET_COUNT_PER_TAG
        current_count = self.tag_counts.get(tag, 0)
        start_page = self.progress.get(tag, {}).get('page', 1)

        print(f"目标数量: {target_count} | 已有: {current_count} | 起始页: {start_page}")

        page = start_page
        consecutive_empty = 0  # 连续空页计数

        while current_count < target_count and consecutive_empty < 5:
            print(f"\n[标签:{tag}] 第 {page} 页 | 已采集 {current_count}/{target_count}")

            # 搜索视频
            videos = await self.safe_api_call(
                self.search_videos,
                max_retries=3,
                keyword=tag,
                page=page
            )

            if not videos:
                consecutive_empty += 1
                print(f"  [!] 第 {page} 页无数据 (连续空页: {consecutive_empty})")
                page += 1
                continue

            consecutive_empty = 0

            # 处理每个视频
            for video_item in videos:
                bvid = video_item.get('bvid')
                if not bvid:
                    continue

                # 处理视频
                success = await self.process_video(bvid, tag)

                if success:
                    current_count += 1
                    self.progress[tag] = {"page": page, "count": current_count}
                    self.save_progress()

                # 检查是否达到目标
                if current_count >= target_count:
                    print(f"\n[完成] 标签 '{tag}' 已达到目标数量 {target_count}")
                    break

                # 每10个视频保存一次进度
                if current_count % 10 == 0:
                    self.save_progress()

            page += 1

            # 页数限制（防止无限循环）
            if page > 50:
                print(f"[!] 达到最大页数限制 (50)")
                break

        # 最终保存
        self.progress[tag] = {"page": page, "count": current_count}
        self.save_progress()

        print(f"\n[标签:{tag}] 采集结束 | 最终数量: {current_count}/{target_count}")

    async def run(self):
        """运行完整采集任务"""
        print("="*60)
        print("B站视频数据采集任务")
        print("="*60)
        print(f"目标标签: {TARGET_TAGS}")
        print(f"目标数量: 每标签 {TARGET_COUNT_PER_TAG} 个")
        print(f"过滤条件: 播放≥{MIN_VIEW}, 弹幕≥{MIN_DANMAKU}")
        print(f"并发数: {MAX_CONCURRENT}")
        print("="*60)

        for tag in TARGET_TAGS:
            await self.crawl_tag(tag)

        # 最终统计
        print("\n" + "="*60)
        print("采集任务完成!")
        print("="*60)
        print("各标签采集数量:")
        for tag, count in self.tag_counts.items():
            print(f"  {tag}: {count}")
        print(f"\n总计: {len(self.collected_bvids)} 个视频")
        print(f"视频信息: {VIDEO_INFO_FILE}")
        print(f"弹幕目录: {DANMAKU_DIR}")


async def main():
    """主函数"""
    async with BilibiliCrawler() as crawler:
        await crawler.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] 用户中断，进度已保存")
    except Exception as e:
        print(f"\n[错误] {e}")
        raise

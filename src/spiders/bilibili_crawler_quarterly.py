"""
B站视频与弹幕采集 - 按季度分批策略
每季度采集前10页视频，然后采集弹幕
"""

import asyncio
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

from bilibili_api import search, video, user, request_settings as bili_settings
from bilibili_api.exceptions import ResponseCodeException, NetworkException

# ============ 配置 ============
DATA_DIR = Path("D:/Claude_Code/bilibili-ai-tag-analysis/data/raw")
DANMAKU_DIR = DATA_DIR / "danmaku"
PROGRESS_FILE = DATA_DIR / "crawler_quarterly_progress.json"
VIDEO_INFO_FILE = DATA_DIR / "bilibili_video_info.csv"

# 目标标签
TARGET_TAGS = {
    "人工智能": {"fuzzy": False},
    "科技": {"fuzzy": False},
    "AI焦虑": {"fuzzy": True},
    "DeepSeek": {"fuzzy": True},
    "机器学习": {"fuzzy": True},
    "AI编程": {"fuzzy": True},
    "ChatGPT": {"fuzzy": True},
    "AI绘画": {"fuzzy": True},
    "Sora": {"fuzzy": True},
    "AI": {"fuzzy": False},
}

# 模糊匹配关键词
FUZZY_KEYWORDS = {
    "机器学习": ["机器学习", "machine learning", "ML算法", "神经网络入门"],
    "ChatGPT": ["ChatGPT", "GPT-4", "OpenAI", "ChatGPT教程"],
    "Sora": ["Sora", "OpenAI Sora", "AI视频生成", "文生视频"],
    "DeepSeek": ["DeepSeek", "DeepSeek教程", "AI大模型"],
    "AI焦虑": ["AI焦虑", "AI取代工作", "AI失业"],
    "AI绘画": ["AI绘画", "Stable Diffusion", "Midjourney", "文生图"],
    "AI编程": ["AI编程", "AI写代码", "Copilot", "AI代码助手"],
}

# 季度定义 (2024-2025共8个季度)
QUARTERS = [
    ("2024-01-01", "2024-03-31", "2024Q1"),
    ("2024-04-01", "2024-06-30", "2024Q2"),
    ("2024-07-01", "2024-09-30", "2024Q3"),
    ("2024-10-01", "2024-12-31", "2024Q4"),
    ("2025-01-01", "2025-03-31", "2025Q1"),
    ("2025-04-01", "2025-06-30", "2025Q2"),
    ("2025-07-01", "2025-09-30", "2025Q3"),
    ("2025-10-01", "2025-12-31", "2025Q4"),
]

# 采集参数
MIN_VIEW = 50000
MIN_DANMAKU = 50
MAX_PAGES_PER_QUARTER = 10  # 每季度最多10页
MAX_CONCURRENT = 2
REQUEST_DELAY = (2, 4)

PROXY_LIST = [None, "http://127.0.0.1:7890"]


class ProxyRotator:
    def __init__(self, proxy_list):
        self.proxy_list = proxy_list
        self.current_index = 0
        self.failed_proxies: Set[str] = set()

    def get_next_proxy(self) -> Optional[str]:
        attempts = 0
        while attempts < len(self.proxy_list):
            proxy = self.proxy_list[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxy_list)
            if proxy in self.failed_proxies and len(self.failed_proxies) < len(self.proxy_list):
                attempts += 1
                continue
            return proxy
        self.failed_proxies.clear()
        return self.proxy_list[0]

    def mark_failed(self, proxy):
        if proxy:
            self.failed_proxies.add(proxy)


class QuarterlyCrawler:
    def __init__(self):
        self.session = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.proxy_rotator = ProxyRotator(PROXY_LIST)
        self.progress: Dict = {}
        self.collected_bvids: Set[str] = set()
        self.tag_counts: Dict[str, int] = {tag: 0 for tag in TARGET_TAGS}

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DANMAKU_DIR.mkdir(parents=True, exist_ok=True)

        self.load_progress()
        self.load_existing_data()

    def load_progress(self):
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                    self.progress = json.load(f)
                print(f"✓ 已加载进度: {PROGRESS_FILE}")
            except Exception as e:
                print(f"✗ 加载进度失败: {e}")
                self._init_progress()
        else:
            self._init_progress()

    def _init_progress(self):
        self.progress = {}
        for tag in TARGET_TAGS:
            self.progress[tag] = {
                "current_quarter_idx": 0,
                "completed_quarters": [],
                "total": 0
            }

    def save_progress(self):
        try:
            with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"✗ 保存进度失败: {e}")

    def load_existing_data(self):
        if VIDEO_INFO_FILE.exists():
            try:
                with open(VIDEO_INFO_FILE, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        bvid = row.get('bvid')
                        if bvid:
                            self.collected_bvids.add(bvid)
                            tag = row.get('search_tag', '')
                            if tag in self.tag_counts:
                                self.tag_counts[tag] += 1
                print(f"✓ 已加载现有数据: {len(self.collected_bvids)} 个视频")
            except Exception as e:
                print(f"✗ 加载现有数据失败: {e}")

    def setup_proxy(self, proxy):
        if proxy:
            try:
                bili_settings.set_proxy(proxy)
            except:
                pass

    async def random_delay(self):
        await asyncio.sleep(random.uniform(*REQUEST_DELAY))

    async def safe_api_call(self, func, max_retries=3, *args, **kwargs):
        for attempt in range(max_retries):
            proxy = self.proxy_rotator.get_next_proxy()
            self.setup_proxy(proxy)
            try:
                async with self.semaphore:
                    await self.random_delay()
                    return await func(*args, **kwargs)
            except ResponseCodeException as e:
                error_msg = str(e)
                if "-799" in error_msg or "风控" in error_msg:
                    self.proxy_rotator.mark_failed(proxy)
                    wait_time = min((2 ** attempt) * 10, 120)
                    print(f"  [!] 风控触发，等待 {wait_time} 秒...")
                    await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(2 ** attempt)
            except NetworkException as e:
                self.proxy_rotator.mark_failed(proxy)
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
        return None

    async def search_videos_with_time(self, keyword: str, time_start: str, time_end: str, page: int = 1) -> List[Dict]:
        """按时间范围搜索视频"""
        try:
            result = await search.search_by_type(
                keyword=keyword,
                search_type=search.SearchObjectType.VIDEO,
                page=page,
                time_start=time_start,
                time_end=time_end
            )
            return result.get('result', [])
        except Exception as e:
            print(f"[!] 搜索失败: {e}")
            return []

    async def get_video_info(self, bvid: str) -> Optional[Dict]:
        try:
            v = video.Video(bvid=bvid)
            return await v.get_info()
        except Exception as e:
            return None

    async def get_video_tags(self, bvid: str) -> List[str]:
        try:
            v = video.Video(bvid=bvid)
            tags = await v.get_tags()
            return [t.get('tag_name', '') for t in tags if t.get('tag_name')]
        except:
            return []

    async def get_user_fans(self, mid: int) -> int:
        try:
            u = user.User(uid=mid)
            info = await u.get_user_info()
            return info.get('fans', 0)
        except:
            return 0

    async def get_danmaku(self, bvid: str, cid: int) -> List[Dict]:
        danmaku_list = []
        try:
            v = video.Video(bvid=bvid)
            for page in range(1, 21):
                try:
                    dms = await v.get_danmakus(cid=cid, page_index=page)
                    if not dms:
                        break
                    for dm in dms:
                        if hasattr(dm, 'text'):
                            danmaku_list.append({
                                'danmaku_id': getattr(dm, 'id_str', getattr(dm, 'id_', '')),
                                'content': getattr(dm, 'text', ''),
                                'progress': getattr(dm, 'dm_time', 0),
                                'ctime': getattr(dm, 'send_time', 0),
                                'user_hash': getattr(dm, 'crc32_id', '')
                            })
                    await asyncio.sleep(0.3)
                except ResponseCodeException as e:
                    if "-799" in str(e) or "风控" in str(e):
                        await asyncio.sleep(30)
                        continue
                    break
                except:
                    break
        except:
            pass
        return danmaku_list

    def filter_video(self, info: Dict) -> bool:
        if not info:
            return False
        stat = info.get('stat', {})
        view = stat.get('view', 0)
        danmaku = stat.get('danmaku', 0)
        if view < MIN_VIEW or danmaku < MIN_DANMAKU:
            return False
        # 检查时间
        pubdate = info.get('pubdate', 0)
        if pubdate:
            video_date = datetime.fromtimestamp(pubdate)
            if video_date.year < 2024 or video_date.year > 2025:
                return False
        return True

    async def save_video_info(self, record: Dict):
        file_exists = VIDEO_INFO_FILE.exists()
        with open(VIDEO_INFO_FILE, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    async def save_danmaku(self, bvid: str, danmaku_list: List[Dict]):
        if not danmaku_list:
            return
        danmaku_file = DANMAKU_DIR / f"{bvid}.csv"
        with open(danmaku_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=danmaku_list[0].keys())
            writer.writeheader()
            writer.writerows(danmaku_list)

    async def process_video(self, bvid: str, search_tag: str) -> bool:
        if bvid in self.collected_bvids:
            return True

        info = await self.get_video_info(bvid)
        if not info or not self.filter_video(info):
            return False

        tags = await self.get_video_tags(bvid)
        owner_mid = info.get('owner', {}).get('mid', 0)
        fans = await self.get_user_fans(owner_mid) if owner_mid else 0

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

        await self.save_video_info(record)

        # 采集弹幕
        cid = info.get('cid', 0)
        if cid:
            dms = await self.get_danmaku(bvid, cid)
            await self.save_danmaku(bvid, dms)
            print(f"  [弹幕] {len(dms)} 条")

        self.collected_bvids.add(bvid)
        self.tag_counts[search_tag] += 1
        self.progress[search_tag]['total'] += 1

        print(f"  [完成] {bvid} | 播放:{record['stat_view']} | 弹幕:{record['stat_danmaku']}")
        return True

    async def crawl_quarter(self, tag: str, time_start: str, time_end: str, quarter_name: str):
        """采集单个季度的数据"""
        print(f"\n  [{quarter_name}] {time_start} ~ {time_end}")

        keywords = [tag]
        if TARGET_TAGS[tag]['fuzzy'] and tag in FUZZY_KEYWORDS:
            keywords = FUZZY_KEYWORDS[tag][:3]  # 取前3个关键词

        quarter_videos = 0

        for keyword in keywords:
            for page in range(1, MAX_PAGES_PER_QUARTER + 1):
                videos = await self.safe_api_call(
                    self.search_videos_with_time,
                    max_retries=3,
                    keyword=keyword,
                    time_start=time_start,
                    time_end=time_end,
                    page=page
                )

                if not videos:
                    break

                for v in videos:
                    bvid = v.get('bvid')
                    if not bvid:
                        continue

                    success = await self.process_video(bvid, tag)
                    if success:
                        quarter_videos += 1
                        self.save_progress()

                print(f"    关键词'{keyword}' 第{page}页完成 | 本季度累计:{quarter_videos}")

        print(f"  [{quarter_name}] 完成 | 新增 {quarter_videos} 个视频")
        return quarter_videos

    async def run(self):
        """运行完整采集任务"""
        print("="*60)
        print("B站视频季度分批采集任务")
        print("="*60)
        print(f"目标标签: {list(TARGET_TAGS.keys())}")
        print(f"时间范围: 2024-2025 (共{len(QUARTERS)}个季度)")
        print(f"每季度页数: {MAX_PAGES_PER_QUARTER} 页")
        print(f"过滤条件: 播放≥{MIN_VIEW}, 弹幕≥{MIN_DANMAKU}")
        print("="*60)

        for tag in TARGET_TAGS:
            print(f"\n{'='*60}")
            print(f"开始采集标签: {tag}")
            print(f"{'='*60}")

            current_q_idx = self.progress[tag].get('current_quarter_idx', 0)
            completed = self.progress[tag].get('completed_quarters', [])

            for idx in range(current_q_idx, len(QUARTERS)):
                time_start, time_end, q_name = QUARTERS[idx]

                if q_name in completed:
                    print(f"\n  [{q_name}] 已跳过（已完成）")
                    continue

                await self.crawl_quarter(tag, time_start, time_end, q_name)

                # 标记完成
                completed.append(q_name)
                self.progress[tag]['current_quarter_idx'] = idx + 1
                self.progress[tag]['completed_quarters'] = completed
                self.save_progress()

            print(f"\n[标签:{tag}] 所有季度采集完成 | 总计: {self.progress[tag]['total']}")

        # 最终统计
        print("\n" + "="*60)
        print("采集任务完成!")
        print("="*60)
        for tag, count in self.tag_counts.items():
            print(f"  {tag}: {count}")
        print(f"\n总计: {len(self.collected_bvids)} 个视频")
        print(f"视频信息: {VIDEO_INFO_FILE}")
        print(f"弹幕目录: {DANMAKU_DIR}")

        print("\n已完成所有指定标签的B站视频与弹幕核心数据采集任务。")
        print("视频基础数据已存入 `data/raw/bilibili_video_info.csv`")
        print("弹幕详情已分类存入 `data/raw/danmaku/` 目录。")


async def main():
    crawler = QuarterlyCrawler()
    await crawler.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] 用户中断，进度已保存")
    except Exception as e:
        print(f"\n[错误] {e}")
        raise

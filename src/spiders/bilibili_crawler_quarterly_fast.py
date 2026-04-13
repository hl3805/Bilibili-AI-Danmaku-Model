"""
B站视频与弹幕采集 - 高速版（带智能IP调度和错误感知）
- REQUEST_DELAY = (1, 2)
- MAX_CONCURRENT = 3
- IP小黑屋机制
- 错误感知自适应延迟
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

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

from bilibili_api import search, video, user, request_settings as bili_settings
from bilibili_api.exceptions import ResponseCodeException, NetworkException

# ============ 配置 ============
DATA_DIR = Path("D:/Claude_Code/bilibili-ai-tag-analysis/data/raw")
DANMAKU_DIR = DATA_DIR / "danmaku"
PROGRESS_FILE = DATA_DIR / "crawler_quarterly_progress.json"  # 共享进度文件
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

FUZZY_KEYWORDS = {
    "机器学习": ["机器学习", "machine learning", "ML算法", "神经网络入门"],
    "ChatGPT": ["ChatGPT", "GPT-4", "OpenAI", "ChatGPT教程"],
    "Sora": ["Sora", "OpenAI Sora", "AI视频生成", "文生视频"],
    "DeepSeek": ["DeepSeek", "DeepSeek教程", "AI大模型"],
    "AI焦虑": ["AI焦虑", "AI取代工作", "AI失业"],
    "AI绘画": ["AI绘画", "Stable Diffusion", "Midjourney", "文生图"],
    "AI编程": ["AI编程", "AI写代码", "Copilot", "AI代码助手"],
}

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
MAX_PAGES_PER_QUARTER = 10
MAX_CONCURRENT = 3
REQUEST_DELAY = (1, 2)  # 基础延迟
REQUEST_DELAY_SLOW = (2, 4)  # 减速模式

PROXY_LIST = [None, "http://127.0.0.1:7890"]

# IP稳定性追踪（新速率启用时间）
IP_STABILITY_TRACK = {
    None: None,  # 无代理IP
    "http://127.0.0.1:7890": datetime.now(),  # 代理IP先用新速率
}
NEW_RATE_TRIAL_HOURS = 2  # 2小时稳定期


class SmartProxyRotator:
    """智能代理轮换器 - 带小黑屋机制"""

    def __init__(self, proxy_list):
        self.proxy_list = proxy_list
        self.current_index = 0
        self.failed_proxies: Dict[Optional[str], datetime] = {}  # 失败时间
        self.cooldown_seconds = 60  # 小黑屋时间
        self.error_counts: Dict[Optional[str], int] = {}  # 错误计数
        self.last_used: Dict[Optional[str], datetime] = {}  # 最后使用时间

    def get_next_proxy(self) -> Optional[str]:
        """获取下一个可用代理（跳过小黑屋中的IP）"""
        now = datetime.now()
        attempts = 0
        proxy_order = list(self.proxy_list)

        # 优先使用未在小黑屋且已过稳定期的IP
        available_proxies = []
        for proxy in proxy_order:
            # 检查小黑屋
            if proxy in self.failed_proxies:
                cooldown_end = self.failed_proxies[proxy] + timedelta(seconds=self.cooldown_seconds)
                if now < cooldown_end:
                    continue  # 还在小黑屋
                else:
                    del self.failed_proxies[proxy]  # 小黑屋到期

            # 检查新速率稳定期（只有代理IP先用新速率）
            if proxy is not None and IP_STABILITY_TRACK[proxy] is not None:
                stable_time = IP_STABILITY_TRACK[proxy] + timedelta(hours=NEW_RATE_TRIAL_HOURS)
                if now < stable_time:
                    available_proxies.insert(0, proxy)  # 优先使用代理IP
                    continue

            available_proxies.append(proxy)

        if not available_proxies:
            # 所有IP都在小黑屋，强制使用第一个
            print("  [!] 警告：所有IP都在小黑屋，强制使用")
            return self.proxy_list[0]

        # 轮询选择
        selected = available_proxies[self.current_index % len(available_proxies)]
        self.current_index += 1
        self.last_used[selected] = now
        return selected

    def send_to_cooldown(self, proxy: Optional[str], reason: str = ""):
        """将IP送入小黑屋"""
        if proxy is not None:
            self.failed_proxies[proxy] = datetime.now()
            self.error_counts[proxy] = self.error_counts.get(proxy, 0) + 1
            print(f"  [!] IP {proxy} 送入小黑屋 60秒 ({reason})")

    def mark_412_error(self, proxy: Optional[str]):
        """标记412错误，增加小黑屋时间"""
        if proxy is not None:
            self.failed_proxies[proxy] = datetime.now()
            self.error_counts[proxy] = self.error_counts.get(proxy, 0) + 1
            # 412错误增加更长的冷却
            extended_cooldown = 60 * (2 ** min(self.error_counts[proxy], 3))
            print(f"  [!] IP {proxy} 触发412错误，送入小黑屋 {extended_cooldown}秒")


class AdaptiveDelay:
    """自适应延迟管理器"""

    def __init__(self, base_delay, slow_delay):
        self.base_delay = base_delay
        self.slow_delay = slow_delay
        self.current_delay = base_delay
        self.error_count = 0
        self.last_error_time = None
        self.recovery_threshold = 10  # 连续成功10次后恢复
        self.success_count = 0

    def get_delay(self):
        """获取当前延迟"""
        # 检查是否可以恢复
        if self.current_delay == self.slow_delay and self.success_count >= self.recovery_threshold:
            self.current_delay = self.base_delay
            self.success_count = 0
            self.error_count = 0
            print(f"  [延迟恢复] 恢复到快速模式 {self.base_delay}")

        return random.uniform(*self.current_delay)

    def on_error(self, error_code=None):
        """发生错误时调用"""
        self.success_count = 0
        self.error_count += 1

        # 412错误立即切换到慢速
        if error_code == 412:
            self.current_delay = self.slow_delay
            print(f"  [延迟调整] 检测到412错误，切换到慢速模式 {self.slow_delay}")

        # 其他错误累积3次也切换
        elif self.error_count >= 3 and self.current_delay == self.base_delay:
            self.current_delay = self.slow_delay
            print(f"  [延迟调整] 错误累积{self.error_count}次，切换到慢速模式")

    def on_success(self):
        """成功时调用"""
        if self.current_delay == self.slow_delay:
            self.success_count += 1


class QuarterlyCrawlerFast:
    """高速版B站采集器"""

    def __init__(self):
        self.session = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.proxy_rotator = SmartProxyRotator(PROXY_LIST)
        self.delay_manager = AdaptiveDelay(REQUEST_DELAY, REQUEST_DELAY_SLOW)
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

    async def safe_api_call(self, func, max_retries=3, *args, **kwargs):
        """带智能重试的API调用"""
        last_error = None

        for attempt in range(max_retries):
            proxy = self.proxy_rotator.get_next_proxy()
            self.setup_proxy(proxy)

            try:
                async with self.semaphore:
                    delay = self.delay_manager.get_delay()
                    await asyncio.sleep(delay)
                    result = await func(*args, **kwargs)
                    self.delay_manager.on_success()
                    return result

            except ResponseCodeException as e:
                last_error = e
                error_msg = str(e)

                if "-412" in error_msg:
                    # 412错误 - IP被限制
                    self.proxy_rotator.mark_412_error(proxy)
                    self.delay_manager.on_error(412)
                    # 不重试，换IP
                    continue

                elif "-799" in error_msg or "风控" in error_msg:
                    # 799错误 - 风控
                    self.proxy_rotator.send_to_cooldown(proxy, "风控-799")
                    if attempt < max_retries - 1:
                        continue

                else:
                    # 其他API错误
                    self.delay_manager.on_error()
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)

            except NetworkException as e:
                last_error = e
                self.proxy_rotator.send_to_cooldown(proxy, "网络错误")
                self.delay_manager.on_error()
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise

        print(f"  ✗ API调用失败: {last_error}")
        return None

    async def search_videos_with_time(self, keyword: str, time_start: str, time_end: str, page: int = 1) -> List[Dict]:
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
                    await asyncio.sleep(0.2)  # 弹幕分页更快
                except ResponseCodeException as e:
                    if "-799" in str(e) or "风控" in str(e):
                        await asyncio.sleep(20)
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
        print(f"\n  [{quarter_name}] {time_start} ~ {time_end}")

        keywords = [tag]
        if TARGET_TAGS[tag]['fuzzy'] and tag in FUZZY_KEYWORDS:
            keywords = FUZZY_KEYWORDS[tag][:3]

        quarter_videos = 0

        # 科技标签只采集前5页
        max_pages = 5 if tag == "科技" else MAX_PAGES_PER_QUARTER

        for keyword in keywords:
            for page in range(1, max_pages + 1):
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
        print("="*60)
        print("B站视频季度分批采集任务 - 高速版")
        print("="*60)
        print(f"目标标签: {list(TARGET_TAGS.keys())}")
        print(f"时间范围: 2024-2025 (共{len(QUARTERS)}个季度)")
        print(f"每季度页数: {MAX_PAGES_PER_QUARTER} 页")
        print(f"基础延迟: {REQUEST_DELAY} 秒 (自适应)")
        print(f"慢速延迟: {REQUEST_DELAY_SLOW} 秒 (错误时)")
        print(f"并发数: {MAX_CONCURRENT}")
        print(f"IP小黑屋: 60秒 (412错误加倍)")
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

                completed.append(q_name)
                self.progress[tag]['current_quarter_idx'] = idx + 1
                self.progress[tag]['completed_quarters'] = completed
                self.save_progress()

            print(f"\n[标签:{tag}] 所有季度采集完成 | 总计: {self.progress[tag]['total']}")

        print("\n" + "="*60)
        print("采集任务完成!")
        print("="*60)
        for tag, count in self.tag_counts.items():
            print(f"  {tag}: {count}")
        print(f"\n总计: {len(self.collected_bvids)} 个视频")
        print(f"视频信息: {VIDEO_INFO_FILE}")
        print(f"弹幕目录: {DANMAKU_DIR}")

        print("\n已完成所有指定标签的B站视频与弹幕核心数据采集任务。")


async def main():
    crawler = QuarterlyCrawlerFast()
    await crawler.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] 用户中断，进度已保存")
    except Exception as e:
        print(f"\n[错误] {e}")
        raise

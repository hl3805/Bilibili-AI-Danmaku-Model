"""
B站视频与弹幕核心数据采集爬虫 v2
支持：断点续传、实时保存、代理轮换、模糊匹配
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

# 导入bilibili_api
from bilibili_api import search, video, user, request_settings as bili_settings
from bilibili_api.exceptions import ResponseCodeException, NetworkException
BILI_API_AVAILABLE = True


# ============ 配置常量 ============
DATA_DIR = Path("D:/Claude_Code/bilibili-ai-tag-analysis/data/raw")
DANMAKU_DIR = DATA_DIR / "danmaku"
PROGRESS_FILE = DATA_DIR / "crawler_progress_v2.json"
VIDEO_INFO_FILE = DATA_DIR / "bilibili_video_info.csv"

# 目标标签列表（带模糊匹配配置）
TARGET_TAGS = {
    "人工智能": {"fuzzy": False, "min_count": 1000},
    "科技": {"fuzzy": False, "min_count": 1000},
    "机器学习": {"fuzzy": True, "min_count": 1000},  # 数量可能少，开模糊
    "AI": {"fuzzy": False, "min_count": 1000},
    "ChatGPT": {"fuzzy": True, "min_count": 1000},  # 具体产品，开模糊
    "Sora": {"fuzzy": True, "min_count": 1000},      # 具体产品，开模糊
    "DeepSeek": {"fuzzy": True, "min_count": 1000},  # 具体产品，开模糊
    "AI焦虑": {"fuzzy": True, "min_count": 1000},    # 情感标签，开模糊
    "AI绘画": {"fuzzy": True, "min_count": 1000},    # 细分场景，开模糊
    "AI编程": {"fuzzy": True, "min_count": 1000},    # 细分场景，开模糊
}

# 模糊匹配关键词扩展
FUZZY_KEYWORDS = {
    "机器学习": ["机器学习", "machine learning", "ML算法", "监督学习", "神经网络入门"],
    "ChatGPT": ["ChatGPT", "chatgpt", "GPT-4", "OpenAI", "ChatGPT教程", "ChatGPT应用"],
    "Sora": ["Sora", "OpenAI Sora", "AI视频生成", "sora视频", "文生视频"],
    "DeepSeek": ["DeepSeek", "deepseek", "DeepSeek教程", "DeepSeek使用", "AI大模型"],
    "AI焦虑": ["AI焦虑", "人工智能焦虑", "AI取代工作", "AI失业", "AI恐惧"],
    "AI绘画": ["AI绘画", "AI画图", "Stable Diffusion", "Midjourney", "文生图", "AI插画"],
    "AI编程": ["AI编程", "AI写代码", "Copilot", "AI程序员", "AI代码助手"],
}

# 采集参数
MIN_VIEW = 50000   # 最小播放量
MIN_DANMAKU = 50   # 最小弹幕数
TARGET_COUNT_PER_TAG = 1000  # 每个标签目标数量
MAX_CONCURRENT = 2  # 最大并发数（降低避免风控）
REQUEST_DELAY = (2, 5)  # 请求延迟范围（秒，增加延迟）

# 代理配置
PROXY_LIST = [
    None,  # 无代理
    "http://127.0.0.1:7890",  # 本地代理（如Clash/V2Ray）
]

# 时间窗口
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)


class ProxyRotator:
    """代理轮换器"""

    def __init__(self, proxy_list: List[Optional[str]]):
        self.proxy_list = proxy_list
        self.current_index = 0
        self.failed_proxies: Set[str] = set()
        self.last_proxy: Optional[str] = None

    def get_next_proxy(self) -> Optional[str]:
        """获取下一个代理（轮询+失败跳过）"""
        attempts = 0
        while attempts < len(self.proxy_list):
            proxy = self.proxy_list[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxy_list)

            # 跳过已知失败的代理（除非全部失败）
            if proxy in self.failed_proxies and len(self.failed_proxies) < len(self.proxy_list):
                attempts += 1
                continue

            self.last_proxy = proxy
            return proxy

        # 所有代理都失败了，重置
        self.failed_proxies.clear()
        self.last_proxy = self.proxy_list[0]
        return self.last_proxy

    def mark_proxy_failed(self, proxy: Optional[str]):
        """标记代理失败"""
        if proxy:
            self.failed_proxies.add(proxy)
        print(f"  [!] 代理 {proxy} 标记为失败，切换到下一个代理")


class BilibiliCrawlerV2:
    """B站视频数据采集器 v2（支持代理轮换+模糊匹配）"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.proxy_rotator = ProxyRotator(PROXY_LIST)
        self.progress: Dict = {}
        self.collected_bvids: Set[str] = set()  # 已采集的BVID集合
        self.collected_bvids_with_danmaku: Set[str] = set()  # 已采集弹幕的BVID
        self.tag_counts: Dict[str, int] = {tag: 0 for tag in TARGET_TAGS}
        self.use_bili_api = True  # 是否使用bilibili_api库

        # 确保目录存在
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DANMAKU_DIR.mkdir(parents=True, exist_ok=True)

        # 加载进度和数据
        self.load_progress()
        self.load_existing_data()
        self.load_existing_danmaku()

    def load_progress(self):
        """加载采集进度"""
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                    self.progress = json.load(f)
                print(f"✓ 已加载进度文件: {PROGRESS_FILE}")
            except Exception as e:
                print(f"✗ 加载进度文件失败: {e}")
                self._init_progress()
        else:
            self._init_progress()

    def _init_progress(self):
        """初始化进度"""
        self.progress = {}
        for tag in TARGET_TAGS:
            self.progress[tag] = {
                "page": 1,
                "count": 0,
                "fuzzy_keyword_index": 0,
                "fuzzy_page": 1
            }

    def save_progress(self):
        """保存采集进度"""
        try:
            with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"✗ 保存进度失败: {e}")

    def load_existing_data(self):
        """加载已采集的视频数据"""
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
                print(f"✓ 已加载现有视频数据: {len(self.collected_bvids)} 个视频")
                print(f"  各标签进度: {self.tag_counts}")
            except Exception as e:
                print(f"✗ 加载现有数据失败: {e}")

    def load_existing_danmaku(self):
        """加载已采集的弹幕列表"""
        if DANMAKU_DIR.exists():
            for f in DANMAKU_DIR.iterdir():
                if f.suffix == '.csv':
                    bvid = f.stem
                    self.collected_bvids_with_danmaku.add(bvid)
            print(f"✓ 已采集弹幕的视频: {len(self.collected_bvids_with_danmaku)} 个")

    def _setup_bili_api_proxy(self, proxy: Optional[str]):
        """设置bilibili_api的代理"""
        if BILI_API_AVAILABLE and proxy:
            try:
                bili_settings.set_proxy(proxy)
                print(f"  [代理] bilibili_api 使用代理: {proxy}")
            except Exception as e:
                print(f"  [!] 设置代理失败: {e}")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://search.bilibili.com/',
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
        """带重试机制+代理轮换的API调用"""
        last_error = None

        for attempt in range(max_retries):
            # 每次重试换一个代理
            proxy = self.proxy_rotator.get_next_proxy()
            self._setup_bili_api_proxy(proxy)

            try:
                async with self.semaphore:
                    await self.random_delay()
                    return await func(*args, **kwargs)

            except ResponseCodeException as e:
                last_error = e
                error_msg = str(e)

                if "-799" in error_msg or "风控" in error_msg or "frequently" in error_msg.lower():
                    # 风控错误 - 切换代理并延长等待
                    self.proxy_rotator.mark_proxy_failed(proxy)
                    wait_time = min((2 ** attempt) * 10, 120)  # 最大120秒
                    print(f"  [!] 风控触发(-799)，切换代理，等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)

                elif "-412" in error_msg or "precondition" in error_msg.lower():
                    # 预检失败 - 通常是IP被限制
                    self.proxy_rotator.mark_proxy_failed(proxy)
                    wait_time = 60
                    print(f"  [!] IP被限制(-412)，切换代理，等待 {wait_time} 秒...")
                    await asyncio.sleep(wait_time)

                else:
                    # 其他API错误
                    wait_time = min((2 ** attempt) * 5, 30)
                    print(f"  [!] API错误({error_msg})，{wait_time}秒后重试...")
                    await asyncio.sleep(wait_time)

            except NetworkException as e:
                last_error = e
                self.proxy_rotator.mark_proxy_failed(proxy)
                wait_time = 2 ** attempt
                print(f"  [!] 网络错误: {e}，切换代理，{wait_time}秒后重试...")
                await asyncio.sleep(wait_time)

            except Exception as e:
                last_error = e
                print(f"  [!] 未知错误: {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise

        print(f"  ✗ 达到最大重试次数，放弃请求: {last_error}")
        return None

    async def search_videos(self, keyword: str, page: int = 1, page_size: int = 20) -> List[Dict]:
        """搜索视频"""
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
                return [v for v in videos if v.get('type', '') == 'video']
            except Exception as e2:
                print(f"✗ 搜索失败: {e2}")
                return []

    async def get_video_info(self, bvid: str) -> Optional[Dict]:
        """获取视频详细信息"""
        try:
            v = video.Video(bvid=bvid)
            info = await v.get_info()
            return info
        except Exception as e:
            print(f"  [!] 获取视频信息失败 {bvid}: {e}")
            return None

    async def get_video_tags(self, bvid: str) -> List[str]:
        """获取视频标签"""
        try:
            v = video.Video(bvid=bvid)
            tags = await v.get_tags()
            return [tag.get('tag_name', '') for tag in tags if tag.get('tag_name')]
        except Exception as e:
            print(f"  [!] 获取标签失败 {bvid}: {e}")
            return []

    async def get_danmaku(self, bvid: str, cid: int) -> List[Dict]:
        """获取弹幕数据"""
        danmaku_list = []
        try:
            v = video.Video(bvid=bvid)
            page = 1
            total_pages = 0

            while page <= 20:  # 最多获取20页弹幕
                try:
                    # 使用cid参数获取弹幕
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

                    total_pages = page
                    page += 1
                    await asyncio.sleep(0.5)  # 增加延迟

                except ResponseCodeException as e:
                    error_msg = str(e)
                    if "-799" in error_msg or "风控" in error_msg or "frequently" in error_msg.lower():
                        print(f"  [!] 弹幕接口风控，等待60秒...")
                        await asyncio.sleep(60)
                        continue
                    else:
                        print(f"  [!] 获取弹幕页API错误: {e}")
                        break
                except Exception as e:
                    print(f"  [!] 获取弹幕页失败: {e}")
                    break

            if total_pages > 0:
                print(f"  [弹幕] 获取了 {total_pages} 页，共 {len(danmaku_list)} 条弹幕")

        except Exception as e:
            print(f"  [!] 获取弹幕失败 {bvid}: {e}")

        return danmaku_list

    async def get_user_fans(self, mid: int) -> int:
        """获取UP主粉丝数"""
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

        # 时间窗口检查
        pubdate = info.get('pubdate', 0)
        if pubdate:
            video_date = datetime.fromtimestamp(pubdate)
            if video_date < START_DATE or video_date > END_DATE:
                return False

        return True

    async def process_video(self, bvid: str, search_tag: str) -> bool:
        """处理单个视频（采集信息+弹幕）"""
        # 检查是否已采集
        if bvid in self.collected_bvids:
            # 检查是否已有弹幕
            if bvid not in self.collected_bvids_with_danmaku:
                # 尝试补充弹幕
                await self.supplement_danmaku(bvid)
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
            self.collected_bvids_with_danmaku.add(bvid)

        # 8. 更新进度
        self.collected_bvids.add(bvid)
        self.tag_counts[search_tag] += 1

        print(f"  [完成] {bvid} | 播放:{record['stat_view']} | 弹幕:{record['stat_danmaku']}")
        return True

    async def supplement_danmaku(self, bvid: str):
        """为已采集的视频补充弹幕"""
        try:
            # 从CSV中读取cid
            cid = None
            if VIDEO_INFO_FILE.exists():
                with open(VIDEO_INFO_FILE, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('bvid') == bvid:
                            cid = row.get('cid')
                            if cid:
                                cid = int(cid)
                            break

            if cid:
                print(f"  [补弹幕] {bvid} ...")
                danmaku_list = await self.get_danmaku(bvid, cid)
                if danmaku_list:
                    await self.save_danmaku(bvid, danmaku_list)
                    self.collected_bvids_with_danmaku.add(bvid)
                    print(f"  [补弹幕完成] {bvid} 共 {len(danmaku_list)} 条")
        except Exception as e:
            print(f"  [!] 补充弹幕失败 {bvid}: {e}")

    async def save_video_info(self, record: Dict):
        """实时保存视频信息到CSV"""
        file_exists = VIDEO_INFO_FILE.exists()

        # 使用锁确保并发安全
        async with asyncio.Lock():
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

        try:
            with open(danmaku_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=danmaku_list[0].keys())
                writer.writeheader()
                writer.writerows(danmaku_list)
        except Exception as e:
            print(f"  [!] 保存弹幕失败 {bvid}: {e}")

    async def crawl_tag(self, tag: str, config: Dict):
        """采集单个标签的视频（支持模糊匹配）"""
        print(f"\n{'='*60}")
        print(f"开始采集标签: {tag}")
        print(f"模糊匹配: {'开启' if config['fuzzy'] else '关闭'}")
        print(f"{'='*60}")

        target_count = config['min_count']
        current_count = self.tag_counts.get(tag, 0)
        fuzzy_enabled = config['fuzzy']

        if current_count >= target_count:
            print(f"[完成] 标签 '{tag}' 已达到目标数量 {target_count}")
            return

        # 模糊匹配逻辑
        if fuzzy_enabled and tag in FUZZY_KEYWORDS:
            keywords = FUZZY_KEYWORDS[tag]
            keyword_index = self.progress.get(tag, {}).get('fuzzy_keyword_index', 0)
            fuzzy_page = self.progress.get(tag, {}).get('fuzzy_page', 1)

            while current_count < target_count and keyword_index < len(keywords):
                keyword = keywords[keyword_index]
                print(f"\n[模糊匹配] 使用关键词: '{keyword}' ({keyword_index+1}/{len(keywords)})")

                consecutive_empty = 0
                page = fuzzy_page

                while current_count < target_count and consecutive_empty < 3:
                    print(f"[标签:{tag}] 关键词'{keyword}' 第 {page} 页 | 已采集 {current_count}/{target_count}")

                    videos = await self.safe_api_call(
                        self.search_videos,
                        max_retries=3,
                        keyword=keyword,
                        page=page
                    )

                    if not videos:
                        consecutive_empty += 1
                        print(f"  [!] 第 {page} 页无数据 (连续空页: {consecutive_empty})")
                        page += 1
                        continue

                    consecutive_empty = 0

                    for video_item in videos:
                        bvid = video_item.get('bvid')
                        if not bvid:
                            continue

                        success = await self.process_video(bvid, tag)

                        if success:
                            current_count += 1
                            self.progress[tag] = {
                                "page": 1,
                                "count": current_count,
                                "fuzzy_keyword_index": keyword_index,
                                "fuzzy_page": page
                            }
                            self.save_progress()

                        if current_count >= target_count:
                            print(f"\n[完成] 标签 '{tag}' 已达到目标数量 {target_count}")
                            break

                        if current_count % 10 == 0:
                            self.save_progress()

                    page += 1

                    if page > 100:  # 每个关键词最多100页
                        break

                # 切换到下一个关键词
                keyword_index += 1
                fuzzy_page = 1
                self.progress[tag] = {
                    "page": 1,
                    "count": current_count,
                    "fuzzy_keyword_index": keyword_index,
                    "fuzzy_page": fuzzy_page
                }
                self.save_progress()

        else:
            # 普通采集逻辑
            start_page = self.progress.get(tag, {}).get('page', 1)
            page = start_page
            consecutive_empty = 0

            while current_count < target_count and consecutive_empty < 5:
                print(f"\n[标签:{tag}] 第 {page} 页 | 已采集 {current_count}/{target_count}")

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

                for video_item in videos:
                    bvid = video_item.get('bvid')
                    if not bvid:
                        continue

                    success = await self.process_video(bvid, tag)

                    if success:
                        current_count += 1
                        self.progress[tag] = {
                            "page": page,
                            "count": current_count,
                            "fuzzy_keyword_index": 0,
                            "fuzzy_page": 1
                        }
                        self.save_progress()

                    if current_count >= target_count:
                        print(f"\n[完成] 标签 '{tag}' 已达到目标数量 {target_count}")
                        break

                    if current_count % 10 == 0:
                        self.save_progress()

                page += 1

                if page > 200:  # 每标签最多200页
                    print(f"[!] 达到最大页数限制 (200)")
                    break

        self.progress[tag]["count"] = current_count
        self.save_progress()
        print(f"\n[标签:{tag}] 采集结束 | 最终数量: {current_count}/{target_count}")

    async def batch_supplement_danmaku(self):
        """批量补充所有已采集视频的弹幕"""
        print("\n" + "="*60)
        print("批量补充弹幕数据")
        print("="*60)

        # 找出需要补充弹幕的视频
        need_danmaku = self.collected_bvids - self.collected_bvids_with_danmaku
        print(f"需要补充弹幕的视频: {len(need_danmaku)} 个")

        if not need_danmaku:
            print("所有视频都已有弹幕数据！")
            return

        # 构建bvid到cid的映射
        bvid_cid_map = {}
        if VIDEO_INFO_FILE.exists():
            with open(VIDEO_INFO_FILE, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bvid = row.get('bvid')
                    cid = row.get('cid')
                    if bvid and cid:
                        try:
                            bvid_cid_map[bvid] = int(cid)
                        except:
                            pass

        success_count = 0
        fail_count = 0

        for i, bvid in enumerate(need_danmaku, 1):
            print(f"\n[{i}/{len(need_danmaku)}] 补充弹幕: {bvid}")

            cid = bvid_cid_map.get(bvid)
            if not cid:
                print(f"  [!] 无法获取 {bvid} 的cid")
                fail_count += 1
                continue

            try:
                danmaku_list = await self.get_danmaku(bvid, cid)
                if danmaku_list:
                    await self.save_danmaku(bvid, danmaku_list)
                    self.collected_bvids_with_danmaku.add(bvid)
                    success_count += 1
                    print(f"  [完成] {bvid} 共 {len(danmaku_list)} 条弹幕")
                else:
                    print(f"  [!] {bvid} 无弹幕数据")
                    fail_count += 1

                # 每10个保存一次进度
                if i % 10 == 0:
                    self.save_progress()

                # 间隔休息
                await asyncio.sleep(1)

            except Exception as e:
                print(f"  [错误] {bvid}: {e}")
                fail_count += 1

        self.save_progress()
        print(f"\n[弹幕补充完成] 成功: {success_count}, 失败: {fail_count}")

    async def run(self, mode: str = "all"):
        """
        运行采集任务
        mode: "all"=采集视频+弹幕, "danmaku_only"=仅补充弹幕, "video_only"=仅采集视频
        """
        print("="*60)
        print("B站视频数据采集任务 v2")
        print("="*60)
        print(f"目标标签: {list(TARGET_TAGS.keys())}")
        print(f"目标数量: 每标签 {TARGET_COUNT_PER_TAG} 个")
        print(f"过滤条件: 播放≥{MIN_VIEW}, 弹幕≥{MIN_DANMAKU}")
        print(f"并发数: {MAX_CONCURRENT}")
        print(f"代理配置: {PROXY_LIST}")
        print(f"模式: {mode}")
        print("="*60)

        if mode in ["all", "danmaku_only"]:
            # 首先补充现有视频的弹幕
            await self.batch_supplement_danmaku()

        if mode in ["all", "video_only"]:
            # 采集视频数据
            for tag, config in TARGET_TAGS.items():
                await self.crawl_tag(tag, config)

        # 最终统计
        print("\n" + "="*60)
        print("采集任务完成!")
        print("="*60)
        print("各标签采集数量:")
        for tag, count in self.tag_counts.items():
            target = TARGET_TAGS[tag]['min_count']
            status = "✓" if count >= target else "○"
            print(f"  {status} {tag}: {count}/{target}")
        print(f"\n总计视频: {len(self.collected_bvids)} 个")
        print(f"总计弹幕: {len(self.collected_bvids_with_danmaku)} 个视频")
        print(f"视频信息: {VIDEO_INFO_FILE}")
        print(f"弹幕目录: {DANMAKU_DIR}")

        # 输出指定回复
        print("\n" + "="*60)
        print("已完成所有指定标签的B站视频与弹幕核心数据采集任务。")
        print("视频基础数据已存入 `data/raw/bilibili_video_info.csv`")
        print("弹幕详情已分类存入 `data/raw/danmaku/` 目录。")
        print("等待下一步数据清洗与UP主信息补充的指示。")
        print("="*60)


async def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='B站视频与弹幕采集器 v2')
    parser.add_argument('--mode', choices=['all', 'danmaku_only', 'video_only'], default='all',
                        help='运行模式: all=全部, danmaku_only=仅弹幕, video_only=仅视频')
    args = parser.parse_args()

    async with BilibiliCrawlerV2() as crawler:
        await crawler.run(mode=args.mode)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] 用户中断，进度已保存")
    except Exception as e:
        print(f"\n[错误] {e}")
        raise

"""
B站视频与弹幕核心数据采集爬虫 - 顺序执行版
按优先级逐个标签完成，每个标签至少1000个视频
支持断点续传、实时保存
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

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: 安装 tqdm 可获得更好的进度显示体验 (pip install tqdm)")

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

from bilibili_api import search, video, user, request_settings
from bilibili_api.exceptions import ResponseCodeException, NetworkException

# 设置代理
request_settings.set_proxy("http://127.0.0.1:7890")
print("✓ 代理已设置: http://127.0.0.1:7890")

# ============ 配置常量 ============
DATA_DIR = Path("D:/Claude_Code/bilibili-ai-tag-analysis/data/raw")
DANMAKU_DIR = DATA_DIR / "danmaku"
PROGRESS_FILE = DATA_DIR / "crawler_progress_sequential.json"
VIDEO_INFO_FILE = DATA_DIR / "bilibili_video_info.csv"

# 按优先级排序的目标标签列表
TARGET_TAGS = [
    "人工智能",  # 最高优先级
    "AI",        # 第二优先级
    "DeepSeek",  # 第三优先级
    "ChatGPT",   # 第四优先级
    "Sora",      # 第五优先级
    "科技",      # 第六优先级
    "机器学习",  # 第七优先级
    "AI焦虑",    # 第八优先级
    "AI绘画",    # 第九优先级
    "AI编程"     # 第十优先级
]

# 采集参数（降低频率避免风控）
MIN_VIEW = 10000  # 最小播放量（降低以加快采集）
MIN_DANMAKU = 50  # 最小弹幕数（降低以加快采集）
TARGET_COUNT_PER_TAG = 1000  # 每个标签目标数量
MAX_CONCURRENT = 1  # 最大并发数（降低为1，避免并发触发风控）
REQUEST_DELAY = (5, 10)  # 请求延迟范围（秒，进一步增加以避免-1200错误）
BATCH_REST_COUNT = 50  # 每采集50个视频休息一次
BATCH_REST_TIME = 60  # 休息时间（秒）


class SequentialBilibiliCrawler:
    """B站视频数据采集器（顺序执行版）"""

    def __init__(self):
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.progress: Dict = {}
        self.collected_bvids: Set[str] = set()  # 已采集的BVID集合
        self.tag_counts: Dict[str, int] = {tag: 0 for tag in TARGET_TAGS}
        self.current_tag_index = 0  # 当前正在采集的标签索引

        # 确保目录存在
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DANMAKU_DIR.mkdir(parents=True, exist_ok=True)

        # 加载进度
        self.load_progress()
        self.load_existing_data()

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        self.save_progress()
        return None

    def load_progress(self):
        """加载采集进度"""
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.progress = data.get('progress', {})
                    self.current_tag_index = data.get('current_tag_index', 0)
                print(f"✓ 已加载进度文件: {PROGRESS_FILE}")
                print(f"  当前标签索引: {self.current_tag_index} ({TARGET_TAGS[self.current_tag_index] if self.current_tag_index < len(TARGET_TAGS) else '已完成'})")
            except Exception as e:
                print(f"✗ 加载进度文件失败: {e}")
                self._init_progress()
        else:
            self._init_progress()

    def _init_progress(self):
        """初始化进度"""
        self.progress = {tag: {"page": 1, "count": 0, "completed": False} for tag in TARGET_TAGS}
        self.current_tag_index = 0

    def save_progress(self):
        """保存采集进度"""
        try:
            data = {
                'progress': self.progress,
                'current_tag_index': self.current_tag_index,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
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
                # 更新进度中的count
                for tag in TARGET_TAGS:
                    if tag in self.progress:
                        self.progress[tag]['count'] = self.tag_counts[tag]
            except Exception as e:
                print(f"✗ 加载现有数据失败: {e}")

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
                error_str = str(e)
                if "-799" in error_str or "风控" in error_str:
                    wait_time = min((2 ** attempt) * 5, 60)
                    print(f"  [!] 风控触发，等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                elif "412" in error_str:
                    # 412错误需要更长的等待时间
                    wait_time = min((2 ** attempt) * 30, 300)  # 30, 60, 120秒
                    print(f"  [!] 412风控拦截，等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                elif "-1200" in error_str or "降级" in error_str:
                    # -1200错误需要长时间等待
                    wait_time = min((2 ** attempt) * 120, 600)  # 120, 240, 480秒
                    print(f"  [!] API请求被降级(-1200)，等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  [!] API错误: {e}，{2 ** attempt}秒后重试...")
                    await asyncio.sleep(2 ** attempt)

            except NetworkException as e:
                error_str = str(e)
                if "412" in error_str:
                    # 412错误需要更长的等待时间
                    wait_time = min((2 ** attempt) * 60, 600)  # 60, 120, 240秒
                    print(f"  [!] 412风控拦截，等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    wait_time = 2 ** attempt
                    print(f"  [!] 网络错误: {e}，{wait_time}秒后重试...")
                    await asyncio.sleep(wait_time)

            except Exception as e:
                error_str = str(e)
                if "412" in error_str:
                    wait_time = min((2 ** attempt) * 60, 600)
                    print(f"  [!] 412风控拦截，等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  [!] 未知错误: {type(e).__name__}: {e}")
                    if attempt == max_retries - 1:
                        raise

        return None

    async def search_videos(self, keyword: str, page: int = 1) -> List[Dict]:
        """搜索视频"""
        try:
            result = await search.search_by_type(
                keyword=keyword,
                search_type=search.SearchObjectType.VIDEO,
                page=page
            )
            return result.get('result', [])
        except Exception as e:
            error_str = str(e)
            # -1200错误需要特殊处理，不要立即尝试备用方案
            if "-1200" in error_str or "降级" in error_str:
                print(f"[!] search_by_type 被降级过滤(-1200)，跳过此页")
                return []
            print(f"[!] search_by_type 失败: {e}, 尝试普通搜索...")
            try:
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
            return await v.get_info()
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

    async def get_danmaku(self, bvid: str, cid: int, total_danmaku: int = 0) -> List[Dict]:
        """获取弹幕数据 - 按比例随机采样

        Args:
            bvid: 视频ID
            cid: 视频cid
            total_danmaku: 视频总弹幕数（用于计算采集数量）
        """
        import random
        danmaku_list = []

        # 计算目标采集数量：80%或全部(不满100条时)，最多1000条
        if total_danmaku < 100:
            target_count = total_danmaku  # 不满100条全采集
        else:
            target_count = min(int(total_danmaku * 0.8), 1000)  # 80%，最多1000

        if target_count <= 0:
            return []

        try:
            # 创建Video对象获取弹幕
            v = video.Video(bvid=bvid)

            # 先获取所有弹幕（多页）
            all_danmaku = []
            page = 1
            max_pages = 20  # 最多获取20页

            while page <= max_pages:
                try:
                    dms = await v.get_danmakus(page_index=page)
                    if not dms:
                        break

                    for dm in dms:
                        if hasattr(dm, 'text'):
                            all_danmaku.append({
                                'danmaku_id': getattr(dm, 'dmid', ''),
                                'content': getattr(dm, 'text', ''),
                                'progress': getattr(dm, 'dm_time', 0),
                                'ctime': getattr(dm, 'send_time', 0),
                                'user_hash': getattr(dm, 'crc32_id', '')
                            })

                    # 如果已经采集够目标数量，停止获取
                    if len(all_danmaku) >= target_count:
                        break

                    page += 1
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"  [!] 获取弹幕页失败: {e}")
                    break

            # 随机采样：如果获取的弹幕多于目标数，随机选择
            if len(all_danmaku) > target_count:
                danmaku_list = random.sample(all_danmaku, target_count)
                print(f"  [弹幕采样] 共{len(all_danmaku)}条，随机采样{target_count}条")
            else:
                danmaku_list = all_danmaku
                print(f"  [弹幕全采] 共{len(all_danmaku)}条")

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

        if view < MIN_VIEW:
            return False
        if danmaku < MIN_DANMAKU:
            return False

        return True

    async def process_video(self, bvid: str, search_tag: str) -> bool:
        """处理单个视频（采集信息+弹幕）"""
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
            'title': info.get('title', '').replace('\n', ' ').replace('\r', ' ')[:200],
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
            'tags': '|'.join(tags[:20]),
            'search_tag': search_tag
        }

        # 6. 实时保存视频信息
        await self.save_video_info(record)

        # 7. 获取并保存弹幕（传入总弹幕数以计算采集数量）
        cid = info.get('cid', 0)
        total_danmaku = stat.get('danmaku', 0)
        if cid:
            danmaku_list = await self.get_danmaku(bvid, cid, total_danmaku)
            await self.save_danmaku(bvid, danmaku_list)

        # 8. 更新进度
        self.collected_bvids.add(bvid)
        self.tag_counts[search_tag] += 1

        print(f"  [完成] {bvid} | 播放:{record['stat_view']} | 弹幕:{record['stat_danmaku']}")
        return True

    async def save_video_info(self, record: Dict):
        """实时保存视频信息到CSV"""
        file_exists = VIDEO_INFO_FILE.exists()

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
            writer = csv.DictWriter(f, fieldnames=danmaku_list[0].keys())
            writer.writeheader()
            writer.writerows(danmaku_list)

    def print_progress_bar(self, current: int, total: int, tag: str, eta: str = "计算中..."):
        """打印进度条"""
        percent = current / total * 100
        bar_width = 30
        filled = int(bar_width * current / total)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"\r  [{tag}] [{bar}] {current}/{total} ({percent:.1f}%) | ETA: {eta}", end='', flush=True)

    async def crawl_tag(self, tag: str) -> bool:
        """采集单个标签的视频，直到达到目标数量"""
        print(f"\n{'='*70}")
        print(f"开始采集标签: {tag}")
        print(f"{'='*70}")

        target_count = TARGET_COUNT_PER_TAG
        current_count = self.tag_counts.get(tag, 0)
        start_page = self.progress.get(tag, {}).get('page', 1)
        batch_counter = 0  # 批量计数器
        tag_start_time = time.time()

        # 检查是否已完成
        if current_count >= target_count:
            print(f"✓ 标签 '{tag}' 已完成 ({current_count}/{target_count})")
            self.progress[tag]['completed'] = True
            return True

        print(f"目标数量: {target_count} | 已有: {current_count} | 还需: {target_count - current_count} | 起始页: {start_page}")
        print(f"并发数: {MAX_CONCURRENT} | 请求延迟: {REQUEST_DELAY[0]}-{REQUEST_DELAY[1]}秒")

        page = start_page
        consecutive_empty = 0

        while current_count < target_count and consecutive_empty < 5:
            need = target_count - current_count
            # 计算ETA
            elapsed = time.time() - tag_start_time
            if current_count > 0:
                avg_time_per_video = elapsed / (current_count - self.tag_counts.get(tag, 0) + 1)
                eta_seconds = avg_time_per_video * (target_count - current_count)
                eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m"
            else:
                eta_str = "计算中..."
            self.print_progress_bar(current_count, target_count, tag, eta_str)
            print(f"\n[{tag}] 第 {page} 页 | 已采集 {current_count}/{target_count} | 还需 {need}")

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
            print(f"  找到 {len(videos)} 个视频")

            # 处理每个视频
            for video_item in videos:
                bvid = video_item.get('bvid')
                if not bvid:
                    continue

                success = await self.process_video(bvid, tag)

                if success:
                    current_count += 1
                    batch_counter += 1
                    self.progress[tag] = {"page": page, "count": current_count, "completed": current_count >= target_count}
                    self.save_progress()

                    # 更新进度条
                    elapsed = time.time() - tag_start_time
                    if current_count > 0:
                        avg_time_per_video = elapsed / (current_count - self.tag_counts.get(tag, 0) + 1)
                        eta_seconds = avg_time_per_video * (target_count - current_count)
                        eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m"
                    else:
                        eta_str = "计算中..."
                    self.print_progress_bar(current_count, target_count, tag, eta_str)

                    # 批量休息检查
                    if batch_counter >= BATCH_REST_COUNT:
                        print(f"\n[!] 已采集 {BATCH_REST_COUNT} 个视频，休息 {BATCH_REST_TIME} 秒...")
                        await asyncio.sleep(BATCH_REST_TIME)
                        batch_counter = 0

                # 检查是否达到目标
                if current_count >= target_count:
                    self.print_progress_bar(current_count, target_count, tag, "完成!")
                    elapsed_total = time.time() - tag_start_time
                    print(f"\n[完成] 标签 '{tag}' 已达到目标数量 {target_count}")
                    print(f"  用时: {int(elapsed_total//60)}分{int(elapsed_total%60)}秒 | 平均: {elapsed_total/target_count:.1f}秒/视频")
                    self.progress[tag]['completed'] = True
                    self.save_progress()
                    return True

                # 每10个视频保存一次进度
                if current_count % 10 == 0:
                    self.save_progress()

            page += 1

            # 页数限制（防止无限循环）
            if page > 150:
                print(f"[!] 达到最大页数限制 (150)，标签 '{tag}' 采集结束")
                break

        # 保存最终进度
        self.progress[tag] = {"page": page, "count": current_count, "completed": current_count >= target_count}
        self.save_progress()

        print(f"\n[{tag}] 采集结束 | 最终数量: {current_count}/{target_count}")
        return current_count >= target_count

    async def run(self):
        """运行完整采集任务（按优先级顺序）"""
        print("="*70)
        print("B站视频数据采集任务 - 顺序执行版")
        print("="*70)
        print(f"目标标签（按优先级）: {TARGET_TAGS}")
        print(f"每个标签目标: {TARGET_COUNT_PER_TAG} 个")
        print(f"过滤条件: 播放≥{MIN_VIEW}, 弹幕≥{MIN_DANMAKU}")
        print(f"并发数: {MAX_CONCURRENT}")
        print("="*70)

        # 从当前标签索引开始
        for i in range(self.current_tag_index, len(TARGET_TAGS)):
            self.current_tag_index = i
            tag = TARGET_TAGS[i]

            completed = await self.crawl_tag(tag)

            if completed:
                print(f"\n✓✓✓ 标签 '{tag}' 完成！进入下一个标签...")
            else:
                print(f"\n[!] 标签 '{tag}' 未完成，但已尽力采集")

            # 标签间休息
            if i < len(TARGET_TAGS) - 1:
                next_tag = TARGET_TAGS[i + 1]
                print(f"\n[休息] 5秒后进入下一个标签 '{next_tag}'...")
                await asyncio.sleep(5)

        # 最终统计
        print("\n" + "="*70)
        print("采集任务完成!")
        print("="*70)

        # 计算整体进度
        total_target = len(TARGET_TAGS) * TARGET_COUNT_PER_TAG
        total_collected = sum(self.tag_counts.get(tag, 0) for tag in TARGET_TAGS)
        overall_percent = total_collected / total_target * 100

        # 打印整体进度条
        bar_width = 40
        filled = int(bar_width * total_collected / total_target)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"\n整体进度: [{bar}] {total_collected}/{total_target} ({overall_percent:.1f}%)")

        print("\n各标签采集数量:")
        total = 0
        for tag in TARGET_TAGS:
            count = self.tag_counts.get(tag, 0)
            status = "✓" if count >= TARGET_COUNT_PER_TAG else "○"
            tag_bar_width = 20
            tag_filled = int(tag_bar_width * min(count, TARGET_COUNT_PER_TAG) / TARGET_COUNT_PER_TAG)
            tag_bar = '█' * tag_filled + '░' * (tag_bar_width - tag_filled)
            print(f"  {status} {tag:10s} [{tag_bar}] {count}/{TARGET_COUNT_PER_TAG}")
            total += count
        print(f"\n总计: {len(self.collected_bvids)} 个唯一视频")
        print(f"视频信息: {VIDEO_INFO_FILE}")
        print(f"弹幕目录: {DANMAKU_DIR}")


async def main():
    """主函数"""
    async with SequentialBilibiliCrawler() as crawler:
        await crawler.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] 用户中断，进度已保存")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()

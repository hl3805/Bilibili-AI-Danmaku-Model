"""
小规模测试采集（每个标签只采5个视频验证流程）"""
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

from bilibili_api import search, video, user
from bilibili_api.exceptions import ResponseCodeException, NetworkException

# 配置
DATA_DIR = Path("D:/Claude_Code/bilibili-ai-tag-analysis/data/raw")
DANMAKU_DIR = DATA_DIR / "danmaku"
TEST_TAG = "DeepSeek"  # 只测试一个标签
MIN_VIEW = 100000
MIN_DANMAKU = 200
MAX_CONCURRENT = 2

# 确保目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)
DANMAKU_DIR.mkdir(parents=True, exist_ok=True)


class TestCrawler:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.collected_count = 0
        self.target_count = 5  # 只采集5个

    async def random_delay(self):
        await asyncio.sleep(random.uniform(1, 2))

    async def search_videos(self, keyword: str, page: int = 1):
        try:
            result = await search.search_by_type(
                keyword=keyword,
                search_type=search.SearchObjectType.VIDEO,
                page=page
            )
            return result.get('result', [])
        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    async def get_video_info(self, bvid: str):
        try:
            v = video.Video(bvid=bvid)
            return await v.get_info()
        except Exception as e:
            print(f"  获取信息失败: {e}")
            return None

    async def get_video_tags(self, bvid: str):
        try:
            v = video.Video(bvid=bvid)
            tags = await v.get_tags()
            return [tag.get('tag_name', '') for tag in tags if tag.get('tag_name')]
        except Exception as e:
            return []

    async def get_danmaku(self, bvid: str, cid: int):
        try:
            v = video.Video(bvid=bvid, cid=cid)
            danmaku_list = []
            page = 1
            while page <= 3:  # 最多3页
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
                    await asyncio.sleep(0.3)
                except:
                    break
            return danmaku_list
        except Exception as e:
            return []

    async def process_video(self, bvid: str, search_tag: str) -> bool:
        async with self.semaphore:
            print(f"\n[处理] {bvid} ...")
            await self.random_delay()

            # 获取视频信息
            info = await self.get_video_info(bvid)
            if not info:
                return False

            # 检查过滤条件
            stat = info.get('stat', {})
            view = stat.get('view', 0)
            danmaku_count = stat.get('danmaku', 0)

            print(f"  播放: {view}, 弹幕: {danmaku_count}")

            if view < MIN_VIEW or danmaku_count < MIN_DANMAKU:
                print(f"  [过滤] 不满足条件")
                return False

            # 获取标签
            tags = await self.get_video_tags(bvid)

            # 构建记录
            record = {
                'bvid': bvid,
                'aid': info.get('aid', ''),
                'cid': info.get('cid', ''),
                'title': info.get('title', '').replace('\n', ' ')[:100],
                'pubdate': datetime.fromtimestamp(info.get('pubdate', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'tname': info.get('tname', ''),
                'owner_mid': info.get('owner', {}).get('mid', 0),
                'duration': info.get('duration', 0),
                'stat_view': stat.get('view', 0),
                'stat_danmaku': stat.get('danmaku', 0),
                'stat_like': stat.get('like', 0),
                'stat_coin': stat.get('coin', 0),
                'stat_favorite': stat.get('favorite', 0),
                'tags': '|'.join(tags[:10]),
                'search_tag': search_tag
            }

            # 保存到CSV
            video_file = DATA_DIR / "test_video_info.csv"
            file_exists = video_file.exists()
            with open(video_file, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)

            # 获取弹幕
            cid = info.get('cid', 0)
            if cid:
                danmaku_list = await self.get_danmaku(bvid, cid)
                if danmaku_list:
                    danmaku_file = DANMAKU_DIR / f"{bvid}.csv"
                    with open(danmaku_file, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=danmaku_list[0].keys())
                        writer.writeheader()
                        writer.writerows(danmaku_list)
                    print(f"  [弹幕] {len(danmaku_list)} 条")

            self.collected_count += 1
            print(f"  [完成] 第 {self.collected_count}/{self.target_count} 个")
            return True

    async def run(self):
        print("=" * 60)
        print("小规模测试采集")
        print("=" * 60)
        print(f"测试标签: {TEST_TAG}")
        print(f"目标数量: {self.target_count}")
        print(f"过滤条件: 播放≥{MIN_VIEW}, 弹幕≥{MIN_DANMAKU}")

        page = 1
        while self.collected_count < self.target_count and page <= 10:
            print(f"\n[搜索] 第 {page} 页...")
            videos = await self.search_videos(TEST_TAG, page)

            if not videos:
                print("  无更多视频")
                break

            print(f"  找到 {len(videos)} 个视频")

            for video_item in videos:
                if self.collected_count >= self.target_count:
                    break

                bvid = video_item.get('bvid')
                if not bvid:
                    continue

                success = await self.process_video(bvid, TEST_TAG)
                if success:
                    await asyncio.sleep(1)  # 采集间隔

            page += 1

        print("\n" + "=" * 60)
        print(f"测试完成! 采集了 {self.collected_count} 个视频")
        print(f"数据文件: {DATA_DIR / 'test_video_info.csv'}")
        print(f"弹幕目录: {DANMAKU_DIR}")


async def main():
    crawler = TestCrawler()
    await crawler.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

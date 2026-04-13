"""
简化版B站API测试
"""
import asyncio
import sys
import os

# 设置环境变量避免编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'

from bilibili_api import search, video

TEST_BVID = "BV1GJ411x7h7"

async def test_search():
    """测试搜索"""
    print("测试搜索功能...")
    try:
        # 新版API用法
        result = await search.search_by_type(
            keyword="人工智能",
            search_type=search.SearchObjectType.VIDEO,
            page=1
        )
        videos = result.get('result', [])
        print(f"搜索成功，找到 {len(videos)} 个视频")
        if videos:
            print(f"第一个视频: {videos[0].get('title', 'N/A')[:50]}")
        return True
    except Exception as e:
        print(f"搜索失败: {e}")
        # 尝试旧版API
        try:
            result = await search.search(keyword="人工智能")
            print(f"旧版API搜索成功")
            return True
        except Exception as e2:
            print(f"旧版API也失败: {e2}")
            return False

async def test_video_info():
    """测试视频信息获取"""
    print("\n测试视频信息获取...")
    try:
        v = video.Video(bvid=TEST_BVID)
        info = await v.get_info()
        print(f"视频标题: {info.get('title', 'N/A')[:50]}")
        print(f"播放量: {info.get('stat', {}).get('view', 0)}")
        print(f"弹幕数: {info.get('stat', {}).get('danmaku', 0)}")
        return True
    except Exception as e:
        print(f"获取视频信息失败: {e}")
        return False

async def test_video_tags():
    """测试视频标签获取"""
    print("\n测试视频标签获取...")
    try:
        v = video.Video(bvid=TEST_BVID)
        tags = await v.get_tags()
        print(f"获取到 {len(tags)} 个标签")
        for tag in tags[:5]:
            print(f"  - {tag.get('tag_name', 'N/A')}")
        return True
    except Exception as e:
        print(f"获取标签失败: {e}")
        return False

async def main():
    print("=" * 60)
    print("B站API简单测试")
    print("=" * 60)

    results = []
    results.append(("搜索", await test_search()))
    results.append(("视频信息", await test_video_info()))
    results.append(("视频标签", await test_video_tags()))

    print("\n" + "=" * 60)
    print("测试结果:")
    for name, success in results:
        status = "通过" if success else "失败"
        print(f"  {name}: {status}")

if __name__ == "__main__":
    asyncio.run(main())

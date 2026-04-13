"""
B站 API 测试脚本
基于 bilibili-api-python 库
测试原则：先测试、后运行，单次异步请求验证各接口
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

# 导入 bilibili_api 相关模块
try:
    from bilibili_api import video, search, user, comment, Credential
    from bilibili_api.exceptions import (
        ResponseCodeException,
        NetworkException,
        ApiException
    )
except ImportError as e:
    print(f"导入失败: {e}")
    print("请安装依赖: pip install bilibili-api-python")
    exit(1)


# 测试用的凭证（如需访问会员专享内容，需要填写）
# 当前测试使用游客模式
credential = None  # 无凭证模式


# 测试数据
TEST_BVID = "BV1GJ411x7h7"  # 测试视频（知名视频，数据稳定）
TEST_MID = 208259  # 测试UP主（官方账号）
TEST_CID = None  # 将通过视频信息获取


async def test_video_info():
    """
    测试1: 获取视频基础信息
    需求：播放量、弹幕数、点赞/投币/收藏、发布时间、UP主ID、CID
    """
    print("\n" + "="*60)
    print("测试1: 视频基础信息获取")
    print("="*60)

    try:
        # 创建 Video 对象
        v = video.Video(bvid=TEST_BVID, credential=credential)

        # 获取视频信息
        info = await v.get_info()

        print(f"✓ 视频信息获取成功")
        print(f"  BVID: {info.get('bvid')}")
        print(f"  标题: {info.get('title', 'N/A')[:50]}...")
        print(f"  播放量: {info.get('stat', {}).get('view', 'N/A')}")
        print(f"  弹幕数: {info.get('stat', {}).get('danmaku', 'N/A')}")
        print(f"  点赞: {info.get('stat', {}).get('like', 'N/A')}")
        print(f"  投币: {info.get('stat', {}).get('coin', 'N/A')}")
        print(f"  收藏: {info.get('stat', {}).get('favorite', 'N/A')}")
        print(f"  分享: {info.get('stat', {}).get('share', 'N/A')}")
        print(f"  发布时间: {info.get('pubdate', 'N/A')}")
        print(f"  UP主ID: {info.get('owner', {}).get('mid', 'N/A')}")
        print(f"  CID: {info.get('cid', 'N/A')}")

        # 保存CID供后续测试使用
        global TEST_CID
        TEST_CID = info.get('cid')

        return True, info

    except ResponseCodeException as e:
        print(f"✗ API返回错误: {e}")
        return False, None
    except NetworkException as e:
        print(f"✗ 网络错误: {e}")
        return False, None
    except Exception as e:
        print(f"✗ 未知错误: {type(e).__name__}: {e}")
        return False, None


async def test_video_tags():
    """
    测试2: 获取视频标签
    需求：视频挂载的具体标签
    """
    print("\n" + "="*60)
    print("测试2: 视频标签获取")
    print("="*60)

    try:
        v = video.Video(bvid=TEST_BVID, credential=credential)
        tags = await v.get_tags()

        print(f"✓ 标签获取成功，共 {len(tags)} 个标签")
        for i, tag in enumerate(tags[:5], 1):  # 只显示前5个
            print(f"  {i}. {tag.get('tag_name', 'N/A')} (ID: {tag.get('tag_id', 'N/A')})")

        if len(tags) > 5:
            print(f"  ... 还有 {len(tags) - 5} 个标签")

        return True, tags

    except Exception as e:
        print(f"✗ 错误: {type(e).__name__}: {e}")
        return False, None


async def test_danmaku():
    """
    测试3: 获取弹幕数据
    需求：根据 CID 获取弹幕内容、时间戳
    """
    print("\n" + "="*60)
    print("测试3: 弹幕数据获取")
    print("="*60)

    if not TEST_CID:
        print("✗ 未获取到CID，跳过弹幕测试")
        return False, None

    try:
        # 注意：新版 bilibili-api 弹幕获取方式可能不同
        # 尝试使用 video 对象的弹幕方法
        v = video.Video(bvid=TEST_BVID, credential=credential)

        # 获取弹幕
        danmakus = await v.get_danmakus(page_index=1)

        print(f"✓ 弹幕获取成功")
        print(f"  弹幕数量: {len(danmakus) if isinstance(danmakus, list) else 'N/A'}")

        if isinstance(danmakus, list) and len(danmakus) > 0:
            print(f"  第一条弹幕示例:")
            print(f"    内容: {str(danmakus[0])[:50]}...")

        return True, danmakus

    except Exception as e:
        print(f"✗ 错误: {type(e).__name__}: {e}")
        return False, None


async def test_user_info():
    """
    测试4: 获取UP主信息
    需求：根据 MID 获取粉丝数
    """
    print("\n" + "="*60)
    print("测试4: UP主信息获取")
    print("="*60)

    try:
        u = user.User(uid=TEST_MID, credential=credential)
        info = await u.get_user_info()

        print(f"✓ UP主信息获取成功")
        print(f"  MID: {info.get('mid', 'N/A')}")
        print(f"  昵称: {info.get('name', 'N/A')}")
        print(f"  粉丝数: {info.get('fans', 'N/A')}")
        print(f"  关注数: {info.get('attention', 'N/A')}")
        print(f"  获赞数: {info.get('likes', 'N/A')}")

        return True, info

    except Exception as e:
        print(f"✗ 错误: {type(e).__name__}: {e}")
        return False, None


async def test_search():
    """
    测试5: 视频搜索
    需求：按标签搜索视频，获取 BVID 列表
    """
    print("\n" + "="*60)
    print("测试5: 视频搜索")
    print("="*60)

    try:
        # 搜索关键词
        keyword = "人工智能"

        # 执行搜索
        result = await search.search(
            keyword=keyword,
            search_type=search.SearchObjectType.VIDEO,
            page=1
        )

        print(f"✓ 搜索成功")
        print(f"  关键词: {keyword}")

        # 解析结果
        videos = result.get('result', [])
        print(f"  结果数量: {len(videos)}")

        for i, v in enumerate(videos[:3], 1):
            print(f"  {i}. {v.get('title', 'N/A')[:40]}... (BVID: {v.get('bvid', 'N/A')})")

        return True, result

    except Exception as e:
        print(f"✗ 错误: {type(e).__name__}: {e}")
        return False, None


async def test_error_handling():
    """
    测试6: 异常处理测试
    测试各种异常情况下的错误捕获
    """
    print("\n" + "="*60)
    print("测试6: 异常处理测试")
    print("="*60)

    # 测试1: 无效BVID
    print("\n测试6.1: 无效BVID")
    try:
        v = video.Video(bvid="BV_invalid", credential=credential)
        info = await v.get_info()
        print("✗ 应该抛出异常但没有")
    except ResponseCodeException as e:
        print(f"✓ 正确捕获 ResponseCodeException: {e}")
    except Exception as e:
        print(f"✓ 捕获异常: {type(e).__name__}: {e}")

    # 测试2: 无效用户ID
    print("\n测试6.2: 无效用户ID")
    try:
        u = user.User(uid=999999999, credential=credential)
        info = await u.get_user_info()
        print(f"✓ 获取结果: {info.get('name', 'N/A')}")
    except Exception as e:
        print(f"✓ 捕获异常: {type(e).__name__}: {e}")

    return True, None


async def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("B站 API 测试脚本")
    print("基于 bilibili-api-python")
    print("="*60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试BVID: {TEST_BVID}")
    print(f"测试MID: {TEST_MID}")

    results = {}

    # 执行各项测试
    results['video_info'] = await test_video_info()
    results['video_tags'] = await test_video_tags()
    results['danmaku'] = await test_danmaku()
    results['user_info'] = await test_user_info()
    results['search'] = await test_search()
    results['error_handling'] = await test_error_handling()

    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    for test_name, (success, _) in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {test_name}: {status}")

    passed = sum(1 for s, _ in results.values() if s)
    total = len(results)
    print(f"\n总计: {passed}/{total} 项测试通过")

    return results


if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(run_all_tests())

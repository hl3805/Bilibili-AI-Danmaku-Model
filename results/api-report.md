# B站 API 使用调研报告

> 调研对象: [bilibili-api-python](https://github.com/nemo2011/bilibili-api)  
> 调研时间: 2026年4月7日  
> 报告目的: 为后续B站数据采集提供技术规范参考

---

## 一、库概述

### 1.1 基本信息

| 属性 | 内容 |
|-----|------|
| 库名称 | `bilibili-api-python` |
| GitHub | https://github.com/nemo2011/bilibili-api |
| 安装命令 | `pip install bilibili-api-python` |
| 依赖要求 | Python 3.9+ |
| HTTP客户端 | curl_cffi / httpx / aiohttp |

### 1.2 核心特性

- **全异步设计**: 基于 `asyncio` 的异步API调用
- **类型注解**: 完整的类型提示支持
- **异常处理**: 详细的异常分类和错误码
- **凭证管理**: 支持登录态凭证（Credential）管理
- **无头浏览器**: 部分接口支持 Playwright 模拟

---

## 二、异步编程模式

### 2.1 基础异步调用模式

```python
import asyncio
from bilibili_api import video, Credential

# 创建 Credential（可选，用于需要登录的接口）
credential = Credential(
    sessdata="your_sessdata",
    bili_jct="your_bili_jct",
    buvid3="your_buvid3"
)

# 定义异步函数
async def get_video_info(bvid: str):
    """获取视频信息"""
    v = video.Video(bvid=bvid, credential=credential)
    info = await v.get_info()
    return info

# 运行异步函数
if __name__ == "__main__":
    info = asyncio.run(get_video_info("BV1xx411c7mD"))
    print(info)
```

### 2.2 并发请求模式

```python
import asyncio
from bilibili_api import video

async def fetch_multiple_videos(bvids: list):
    """并发获取多个视频信息"""
    
    async def fetch_one(bvid):
        v = video.Video(bvid=bvid)
        try:
            return await v.get_info()
        except Exception as e:
            return {"bvid": bvid, "error": str(e)}
    
    # 使用 asyncio.gather 并发执行
    tasks = [fetch_one(bvid) for bvid in bvids]
    results = await asyncio.gather(*tasks)
    return results

# 控制并发数（推荐）
async def fetch_with_limit(bvids: list, max_concurrent: int = 5):
    """带并发限制的批量获取"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_one(bvid):
        async with semaphore:
            v = video.Video(bvid=bvid)
            await asyncio.sleep(0.5)  # 请求间隔
            return await v.get_info()
    
    tasks = [fetch_one(bvid) for bvid in bvids]
    return await asyncio.gather(*tasks)
```

---

## 三、核心数据需求与API映射

### 3.1 API 映射总表

| 数据需求 | API类/方法 | 关键参数 | 返回值要点 |
|---------|-----------|---------|-----------|
| 视频搜索 | `search.search()` | keyword, search_type | result列表 |
| 视频信息 | `video.Video.get_info()` | bvid | stat, pubdate, owner, cid |
| 视频标签 | `video.Video.get_tags()` | - | tag_name, tag_id |
| 弹幕数据 | `video.Video.get_danmakus()` | page_index | 弹幕列表 |
| UP主信息 | `user.User.get_user_info()` | uid | name, fans |
| 评论数据 | `comment.get_comments()` | oid, type | replies列表 |

### 3.2 详细API说明

#### 3.2.1 视频搜索

```python
from bilibili_api import search

async def search_videos(keyword: str, page: int = 1):
    """
    搜索视频
    
    Args:
        keyword: 搜索关键词（如"人工智能"）
        page: 页码
        
    Returns:
        搜索结果字典
    """
    result = await search.search(
        keyword=keyword,
        search_type=search.SearchObjectType.VIDEO,
        page=page
    )
    
    videos = result.get('result', [])
    for v in videos:
        bvid = v.get('bvid')
        title = v.get('title')
        pubdate = v.get('pubdate')
        # 提取BVID用于后续详细获取
    
    return videos
```

**关键字段映射**:
| 需求字段 | API返回字段 | 说明 |
|---------|------------|------|
| BVID | `bvid` | 视频唯一标识 |
| 标题 | `title` | 视频标题（含HTML标签需清洗） |
| 发布时间 | `pubdate` | Unix时间戳 |
| UP主ID | `mid` | UP主MID |

#### 3.2.2 视频基础信息

```python
from bilibili_api import video

async def get_video_details(bvid: str):
    """
    获取视频详细信息
    
    Args:
        bvid: 视频BVID
        
    Returns:
        视频详细信息字典
    """
    v = video.Video(bvid=bvid)
    info = await v.get_info()
    
    return {
        'bvid': info.get('bvid'),
        'title': info.get('title'),
        'pubdate': info.get('pubdate'),
        'cid': info.get('cid'),
        'owner_mid': info.get('owner', {}).get('mid'),
        'owner_name': info.get('owner', {}).get('name'),
        'view': info.get('stat', {}).get('view'),
        'danmaku': info.get('stat', {}).get('danmaku'),
        'reply': info.get('stat', {}).get('reply'),
        'like': info.get('stat', {}).get('like'),
        'coin': info.get('stat', {}).get('coin'),
        'favorite': info.get('stat', {}).get('favorite'),
        'share': info.get('stat', {}).get('share'),
        'duration': info.get('duration'),
    }
```

**关键字段映射**:
| 需求字段 | API返回字段路径 | 类型 |
|---------|----------------|------|
| 播放量 | `stat.view` | int |
| 弹幕数 | `stat.danmaku` | int |
| 点赞数 | `stat.like` | int |
| 投币数 | `stat.coin` | int |
| 收藏数 | `stat.favorite` | int |
| 分享数 | `stat.share` | int |
| CID | `cid` | int (用于弹幕获取) |
| UP主ID | `owner.mid` | int |

#### 3.2.3 视频标签

```python
async def get_video_tags(bvid: str):
    """
    获取视频标签
    
    Args:
        bvid: 视频BVID
        
    Returns:
        标签列表
    """
    v = video.Video(bvid=bvid)
    tags = await v.get_tags()
    
    return [
        {
            'tag_id': tag.get('tag_id'),
            'tag_name': tag.get('tag_name'),
        }
        for tag in tags
    ]
```

#### 3.2.4 弹幕数据

```python
async def get_video_danmaku(bvid: str, cid: int = None):
    """
    获取视频弹幕
    
    Args:
        bvid: 视频BVID
        cid: 分P CID（如不提供会自动获取）
        
    Returns:
        弹幕列表
    """
    v = video.Video(bvid=bvid, cid=cid)
    
    # 获取所有分页弹幕
    danmakus = []
    page = 1
    while True:
        page_danmaku = await v.get_danmakus(page_index=page)
        if not page_danmaku:
            break
        danmakus.extend(page_danmaku)
        page += 1
        
        # 限制分页数，避免请求过多
        if page > 10:
            break
    
    return danmakus
```

**弹幕字段**:
| 字段 | 说明 |
|-----|------|
| `text` | 弹幕内容 |
| `dm_time` | 弹幕出现时间（秒） |
| `send_time` | 发送时间（Unix时间戳） |
| `crc32_id` | 用户Hash（非真实ID） |

#### 3.2.5 UP主信息

```python
from bilibili_api import user

async def get_user_fans(uid: int):
    """
    获取UP主粉丝数
    
    Args:
        uid: UP主MID
        
    Returns:
        粉丝数
    """
    u = user.User(uid=uid)
    info = await u.get_user_info()
    
    return {
        'mid': info.get('mid'),
        'name': info.get('name'),
        'fans': info.get('fans'),
        'attention': info.get('attention'),
        'likes': info.get('likes'),
    }
```

---

## 四、异常处理与风控

### 4.1 异常类型梳理

```python
from bilibili_api.exceptions import (
    ResponseCodeException,  # API返回非0状态码
    NetworkException,       # 网络连接错误
    ArgsException,          # 参数错误
    ApiException,           # 通用API错误
)
```

| 异常类型 | 触发场景 | 处理建议 |
|---------|---------|---------|
| `ResponseCodeException` | API返回错误码（如-799风控） | 重试+退避 |
| `NetworkException` | 网络超时、连接失败 | 指数退避重试 |
| `ArgsException` | 参数格式错误 | 检查参数 |
| `ApiException` | 其他API错误 | 记录日志，人工检查 |

### 4.2 异常处理最佳实践

```python
import asyncio
from bilibili_api.exceptions import ResponseCodeException, NetworkException

async def safe_api_call(func, max_retries: int = 3, *args, **kwargs):
    """
    带重试机制的API调用包装器
    
    Args:
        func: 异步API函数
        max_retries: 最大重试次数
        *args, **kwargs: 传递给func的参数
        
    Returns:
        API调用结果
        
    Raises:
        最后一次异常
    """
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
            
        except ResponseCodeException as e:
            # 风控错误 (-799)，需要更长退避
            if "-799" in str(e) or "风控" in str(e):
                wait_time = (2 ** attempt) * 5  # 5, 10, 20秒
                print(f"风控触发，等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                # 其他API错误，标准退避
                wait_time = 2 ** attempt
                print(f"API错误: {e}，{wait_time}秒后重试...")
                await asyncio.sleep(wait_time)
                
        except NetworkException as e:
            # 网络错误，指数退避
            wait_time = 2 ** attempt
            print(f"网络错误: {e}，{wait_time}秒后重试...")
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            # 未知错误，直接抛出
            print(f"未知错误: {type(e).__name__}: {e}")
            raise
    
    raise Exception(f"超过最大重试次数 ({max_retries})")
```

### 4.3 风控规避策略

```python
class BilibiliCrawler:
    """B站爬虫类（含风控规避）"""
    
    def __init__(self, credential=None, delay_range=(1, 3)):
        self.credential = credential
        self.delay_range = delay_range
        self.request_count = 0
        
    async def _random_delay(self):
        """随机延迟"""
        delay = random.uniform(*self.delay_range)
        await asyncio.sleep(delay)
        
    async def get_video_info_safe(self, bvid: str):
        """安全获取视频信息"""
        await self._random_delay()
        
        v = video.Video(bvid=bvid, credential=self.credential)
        return await safe_api_call(v.get_info, max_retries=3)
        
    async def batch_get_videos(self, bvids: list, max_concurrent: int = 3):
        """
        批量获取视频（带并发控制）
        
        注意：即使使用异步，B站API也有IP级别频率限制
        建议: 
        1. 单IP并发不超过3
        2. 请求间隔1-3秒
        3. 使用代理池轮换IP
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_one(bvid):
            async with semaphore:
                return await self.get_video_info_safe(bvid)
        
        tasks = [fetch_one(bvid) for bvid in bvids]
        return await asyncio.gather(*tasks)
```

---

## 五、凭证配置（Credential）

### 5.1 Credential 类说明

```python
from bilibili_api import Credential

# 创建凭证（从浏览器Cookie获取）
credential = Credential(
    sessdata="你的SESSDATA",
    bili_jct="你的bili_jct",
    buvid3="你的buvid3"
)

# 检查凭证有效性
is_valid = await credential.check_valid()
print(f"凭证有效: {is_valid}")
```

### 5.2 Cookie获取方法

1. 登录B站网页版
2. 打开浏览器开发者工具 (F12)
3. 切换到 Application/Storage 标签
4. 找到 Cookies -> https://www.bilibili.com
5. 提取以下字段：
   - `SESSDATA`
   - `bili_jct`
   - `buvid3`

### 5.3 凭证使用场景

| 接口 | 是否需要Credential | 说明 |
|-----|-------------------|------|
| 视频搜索 | 否 | 公开接口 |
| 视频信息 | 否 | 公开接口 |
| 视频标签 | 否 | 公开接口 |
| 弹幕获取 | 否 | 公开接口 |
| UP主信息 | 否 | 公开接口 |
| 用户关注列表 | 是 | 需登录 |
| 个人收藏夹 | 是 | 需登录 |
| 大会员专享内容 | 是 | 需大会员 |

**本项目建议**：视频搜索和基础信息获取不需要Credential，使用游客模式即可。

---

## 六、测试代码示例

### 6.1 单次测试代码

```python
# test_bili_api.py
import asyncio
from bilibili_api import video, user, search

TEST_BVID = "BV1GJ411x7h7"
TEST_MID = 208259

async def test_all():
    """测试所有核心接口"""
    
    # 1. 视频信息
    print("=" * 60)
    print("测试1: 视频信息")
    v = video.Video(bvid=TEST_BVID)
    info = await v.get_info()
    print(f"标题: {info.get('title')}")
    print(f"播放量: {info.get('stat', {}).get('view')}")
    
    # 2. 视频标签
    print("\n" + "=" * 60)
    print("测试2: 视频标签")
    tags = await v.get_tags()
    for tag in tags[:3]:
        print(f"  - {tag.get('tag_name')}")
    
    # 3. UP主信息
    print("\n" + "=" * 60)
    print("测试3: UP主信息")
    u = user.User(uid=TEST_MID)
    user_info = await u.get_user_info()
    print(f"昵称: {user_info.get('name')}")
    print(f"粉丝: {user_info.get('fans')}")
    
    # 4. 搜索
    print("\n" + "=" * 60)
    print("测试4: 视频搜索")
    result = await search.search(
        keyword="人工智能",
        search_type=search.SearchObjectType.VIDEO
    )
    videos = result.get('result', [])
    print(f"找到 {len(videos)} 个视频")
    for v in videos[:3]:
        print(f"  - {v.get('title')[:30]}...")

if __name__ == "__main__":
    asyncio.run(test_all())
```

---

## 七、数据采集架构建议

### 7.1 整体流程

```
┌─────────────────┐
│  1. 关键词搜索   │  ← search.search()
│  (获取BVID列表) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. 视频详情获取 │  ← video.Video.get_info()
│  (基础数据+CID) │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌─────────┐
│3a.标签获取│ │3b.弹幕获取│  ← get_tags() / get_danmakus()
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
┌─────────────────┐
│  4. UP主信息获取 │  ← user.User.get_user_info()
│  (粉丝数等)     │
└─────────────────┘
```

### 7.2 数据库表设计建议

```sql
-- 视频信息表
CREATE TABLE videos (
    bvid VARCHAR(20) PRIMARY KEY,
    title TEXT,
    pubdate TIMESTAMP,
    cid BIGINT,
    owner_mid BIGINT,
    view_count INT,
    danmaku_count INT,
    like_count INT,
    coin_count INT,
    favorite_count INT,
    share_count INT,
    duration INT,
    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 标签表
CREATE TABLE video_tags (
    id SERIAL PRIMARY KEY,
    bvid VARCHAR(20),
    tag_id BIGINT,
    tag_name VARCHAR(100)
);

-- UP主信息表
CREATE TABLE up_users (
    mid BIGINT PRIMARY KEY,
    name VARCHAR(100),
    fans_count INT,
    crawled_at TIMESTAMP
);
```

---

## 八、注意事项与限制

### 8.1 API限制

| 限制类型 | 说明 | 建议 |
|---------|------|------|
| 频率限制 | IP级别风控，超频会-799 | 单IP QPS < 1 |
| 数据限制 | 搜索结果最多50页 | 使用关键词细分 |
| 时间限制 | 历史弹幕有保存期限 | 尽早采集 |
| 反爬升级 | B站定期更新风控策略 | 关注库更新 |

### 8.2 合规建议

1. **遵守robots.txt**: B站允许搜索引擎爬取，但需遵守频率限制
2. **用户隐私**: 弹幕中的用户Hash不可反查真实用户
3. **版权声明**: 视频内容版权归UP主所有，仅采集元数据
4. **免责声明**: 本项目仅供学术研究使用

---

## 九、参考资源

- [bilibili-api-python 文档](https://nemo2011.github.io/bilibili-api/)
- [B站API官方文档（非公开）](https://github.com/SocialSisterYi/bilibili-API-collect)
- [GitHub 仓库](https://github.com/nemo2011/bilibili-api)

---

*报告生成时间: 2026年4月7日*  
*报告版本: v1.0*  
*适用库版本: bilibili-api-python >= 17.0.0*

"""
百度指数爬虫
采集关键词：人工智能、AI、ChatGPT、Sora、DeepSeek、机器学习、AI焦虑
时间范围：2024-01-01 至 2026-01-31
"""

import time
import random
import json
import csv
from datetime import datetime, timedelta
from urllib.parse import quote
import requests
from pathlib import Path


class BaiduIndexScraper:
    """百度指数爬虫类"""

    def __init__(self, cookies_dict=None):
        """
        初始化爬虫
        cookies_dict: 包含BDUSS等Cookie的字典
        """
        self.cookies_dict = cookies_dict or {}
        self.cookie_str = self._build_cookie_string()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Cookie': self.cookie_str,
            'Referer': 'https://index.baidu.com/v2/main/index.html',
            'Host': 'index.baidu.com',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.base_url = "https://index.baidu.com/v2/api/index"

    def _build_cookie_string(self):
        """构建Cookie字符串"""
        cookie_parts = []
        for key, value in self.cookies_dict.items():
            cookie_parts.append(f"{key}={value}")
        return '; '.join(cookie_parts)

    def get_ptbk(self, uniqid):
        """获取解密用的ptbk"""
        url = f"https://index.baidu.com/v2/api/ptbk?uniqid={uniqid}"
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json().get('data', '')
        except Exception as e:
            print(f"获取ptbk失败: {e}")
        return None

    def decrypt_data(self, encrypted_data, ptbk):
        """解密百度指数数据"""
        if not ptbk or not encrypted_data:
            return []

        # 百度指数解密算法
        char_map = {}
        half_len = len(ptbk) // 2
        for i in range(half_len):
            char_map[ptbk[i]] = ptbk[half_len + i]

        result = ""
        for char in encrypted_data:
            result += char_map.get(char, char)

        return result.split(',')

    def fetch_index_data(self, keyword, start_date, end_date):
        """
        获取单个关键词的百度指数数据
        """
        area = "0"  # 全国
        word = quote(json.dumps([[{'name': keyword, 'wordType': 1}]]))

        url = (f"{self.base_url}?area={area}&word={word}"
               f"&startDate={start_date}&endDate={end_date}")

        try:
            print(f"  正在请求: {keyword} ({start_date} ~ {end_date})")
            resp = self.session.get(url, timeout=15)

            if resp.status_code != 200:
                print(f"  请求失败: HTTP {resp.status_code}")
                return None

            data = resp.json()

            if data.get('status') != 0:
                print(f"  API返回错误: {data.get('message', '未知错误')}")
                return None

            result = data.get('data', {})
            if not result:
                print(f"  无数据返回")
                return None

            # 获取解密key
            uniqid = result.get('uniqid')
            if not uniqid:
                print(f"  未获取到uniqid")
                return None

            ptbk = self.get_ptbk(uniqid)
            if not ptbk:
                print(f"  未获取到ptbk")
                return None

            # 解密数据
            user_indexes = result.get('userIndexes', [])
            if not user_indexes:
                print(f"  无userIndexes数据")
                return None

            index_data = user_indexes[0]
            all_data = index_data.get('all', {})
            encrypted = all_data.get('data', '')
            dates = all_data.get('dates', [])

            if not encrypted or not dates:
                print(f"  数据为空")
                return None

            decrypted = self.decrypt_data(encrypted, ptbk)

            if len(decrypted) != len(dates):
                print(f"  数据长度不匹配: dates={len(dates)}, decrypted={len(decrypted)}")
                return None

            print(f"  成功获取 {len(dates)} 天数据")
            return list(zip(dates, decrypted))

        except Exception as e:
            print(f"  获取数据失败 {keyword}: {e}")
            return None

    def fetch_all_keywords(self, keywords, start_date, end_date):
        """
        获取所有关键词的数据
        由于百度指数限制，需要分批次获取（每次最多365天）
        """
        all_data = []

        # 计算需要分多少段
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end - start).days + 1

        print(f"时间跨度: {total_days}天，预计需要分段获取")

        for keyword in keywords:
            print(f"\n正在获取关键词: {keyword}")

            current_start = start
            keyword_data = []

            while current_start <= end:
                # 计算本次查询的结束日期（最多365天）
                current_end = min(current_start + timedelta(days=364), end)

                start_str = current_start.strftime("%Y-%m-%d")
                end_str = current_end.strftime("%Y-%m-%d")

                # 获取数据
                result = self.fetch_index_data(keyword, start_str, end_str)

                if result:
                    for date_str, value in result:
                        keyword_data.append({
                            'date': date_str,
                            'keyword': keyword,
                            'index': int(value) if value.isdigit() else 0
                        })

                # 随机延迟，避免被限制
                time.sleep(random.uniform(1, 3))

                # 移动到下一段
                current_start = current_end + timedelta(days=1)

            all_data.extend(keyword_data)
            print(f"  关键词 {keyword} 共获取 {len(keyword_data)} 条记录")

        return all_data


def save_to_csv(data, filepath):
    """保存数据到CSV"""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'keyword', 'index'])
        writer.writeheader()
        writer.writerows(data)
    print(f"\n数据已保存至: {filepath}")


def main():
    """主函数 - 使用真实Cookie获取数据"""
    output_path = Path("D:/Claude_Code/bilibili-ai-tag-analysis/data/raw/baidu_index_2401_2601.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 用户提供的Cookie
    cookies = {
        'BDUSS': 'mdyakFKVGRUNU92RVBGUm5wOHN1SVhFOE0zWG9RVVlWVUQ1QU1NZW8tcFpJUEpwSUFBQUFBJCQAAAAAAAAAAAEAAABTfMgpMTUzMTI5ODAzegAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFmTymlZk8ppS',
        'BDUSS_BFESS': 'mdyakFKVGRUNU92RVBGUm5wOHN1SVhFOE0zWG9RVVlWVUQ1QU1NZW8tcFpJUEpwSUFBQUFBJCQAAAAAAAAAAAEAAABTfMgpMTUzMTI5ODAzegAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFmTymlZk8ppS',
        'BIDUPSID': 'E5D43470D09B19A245815838B60B751E',
        'PSTM': '1773553548',
        'BAIDUID': '50608C6C5D539A6A22E609078FB70B3A:FG=1',
    }

    keywords = ["人工智能", "AI", "ChatGPT", "Sora", "DeepSeek", "机器学习", "AI焦虑"]
    start_date = "2024-01-01"
    end_date = "2026-01-31"

    print("=" * 60)
    print("开始获取百度指数真实数据")
    print("=" * 60)

    scraper = BaiduIndexScraper(cookies)
    data = scraper.fetch_all_keywords(keywords, start_date, end_date)

    if data:
        save_to_csv(data, output_path)
        print(f"\n共获取 {len(data)} 条记录")
        print(f"覆盖关键词: {set(d['keyword'] for d in data)}")

        # 统计各关键词数据量
        for kw in keywords:
            count = len([d for d in data if d['keyword'] == kw])
            print(f"  {kw}: {count} 条")
    else:
        print("\n获取数据失败，请检查Cookie是否有效")


if __name__ == "__main__":
    main()

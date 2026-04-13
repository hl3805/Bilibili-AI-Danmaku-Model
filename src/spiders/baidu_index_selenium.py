"""
使用Selenium获取百度指数数据
需要：Chrome浏览器 + ChromeDriver
"""

import json
import csv
import time
from datetime import datetime
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("请先安装依赖: pip install selenium webdriver-manager")
    exit(1)


class BaiduIndexSelenium:
    """使用Selenium获取百度指数"""

    def __init__(self, cookies_dict=None):
        self.cookies_dict = cookies_dict or {}
        self.driver = None
        self.setup_driver()

    def setup_driver(self):
        """设置Chrome浏览器"""
        chrome_options = Options()
        # chrome_options.add_argument('--headless')  # 无头模式可能导致登录问题，先禁用
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--disable-features=IsolateOrigins,site-per-process')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        # 禁用webdriver特征
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
            })
            print("Chrome浏览器启动成功")
        except Exception as e:
            print(f"启动浏览器失败: {e}")
            print("请确保已安装Chrome浏览器")
            raise

    def add_cookies(self):
        """添加Cookie到浏览器"""
        # 先访问百度主页
        self.driver.get("https://www.baidu.com")
        time.sleep(2)

        # 添加Cookie
        for name, value in self.cookies_dict.items():
            try:
                self.driver.add_cookie({
                    'name': name,
                    'value': value,
                    'domain': '.baidu.com'
                })
                print(f"  已添加Cookie: {name}")
            except Exception as e:
                print(f"  添加Cookie失败 {name}: {e}")

    def fetch_keyword_data(self, keyword):
        """获取单个关键词的数据"""
        url = f"https://index.baidu.com/v2/main/index.html#/trend/{keyword}?words={keyword}"

        try:
            print(f"\n正在获取: {keyword}")
            self.driver.get(url)
            time.sleep(5)  # 等待页面加载

            # 尝试获取数据（百度指数数据是通过JS动态加载的）
            # 这里需要分析页面结构来获取数据

            # 方法1：尝试从页面源码中提取数据
            page_source = self.driver.page_source

            # 查找包含趋势数据的script标签
            scripts = self.driver.find_elements(By.TAG_NAME, "script")

            for script in scripts:
                text = script.get_attribute('innerHTML')
                if 'indexData' in text or 'userIndexes' in text:
                    print(f"  找到数据脚本")
                    # 这里需要解析具体的JSON数据
                    return self._extract_data_from_script(text)

            # 方法2：尝试直接访问API
            # 百度指数的API请求会携带加密参数，直接从页面获取较困难

            print(f"  未能从页面提取数据")
            return None

        except Exception as e:
            print(f"  获取失败: {e}")
            return None

    def _extract_data_from_script(self, script_text):
        """从脚本中提取数据"""
        # 这里需要根据实际的页面结构来实现
        # 百度指数的数据通常存储在window对象或特定的变量中
        try:
            # 尝试找到JSON数据
            import re
            json_match = re.search(r'indexData\s*=\s*({.*?});', script_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data
        except Exception as e:
            print(f"  解析数据失败: {e}")
        return None

    def export_data_via_ui(self, keyword):
        """
        通过UI操作导出数据
        百度指数页面有导出功能按钮
        """
        try:
            print(f"\n尝试导出: {keyword}")
            url = f"https://index.baidu.com/v2/main/index.html#/trend/{keyword}?words={keyword}"
            self.driver.get(url)
            time.sleep(5)

            # 尝试找到导出按钮
            # 注意：百度的UI经常变化，这里的选择器可能需要更新
            try:
                export_btn = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "index-export-btn"))
                )
                export_btn.click()
                print(f"  已点击导出按钮")
                time.sleep(3)
                return True
            except:
                print(f"  未找到导出按钮")
                return False

        except Exception as e:
            print(f"  导出失败: {e}")
            return False

    def close(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()


def main():
    """主函数"""
    # Cookie配置
    cookies = {
        'BDUSS': 'mdyakFKVGRUNU92RVBGUm5wOHN1SVhFOE0zWG9RVVlWVUQ1QU1NZW8tcFpJUEpwSUFBQUFBJCQAAAAAAAAAAAEAAABTfMgpMTUzMTI5ODAzegAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFmTymlZk8ppS',
        'BDUSS_BFESS': 'mdyakFKVGRUNU92RVBGUm5wOHN1SVhFOE0zWG9RVVlWVUQ1QU1NZW8tcFpJUEpwSUFBQUFBJCQAAAAAAAAAAAEAAABTfMgpMTUzMTI5ODAzegAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFmTymlZk8ppS',
        'BIDUPSID': 'E5D43470D09B19A245815838B60B751E',
        'PSTM': '1773553548',
        'BAIDUID': '50608C6C5D539A6A22E609078FB70B3A:FG=1',
    }

    keywords = ["人工智能", "AI", "ChatGPT", "Sora", "DeepSeek", "机器学习", "AI焦虑"]

    print("=" * 60)
    print("百度指数 Selenium 数据采集")
    print("=" * 60)
    print("\n注意：")
    print("1. 将自动打开Chrome浏览器")
    print("2. 请勿关闭浏览器窗口")
    print("3. 如果Cookie有效，数据将自动导出")
    print("4. 如果遇到验证码，需要手动处理\n")

    try:
        scraper = BaiduIndexSelenium(cookies)
        scraper.add_cookies()

        # 尝试获取每个关键词的数据
        for keyword in keywords[:2]:  # 先测试前两个关键词
            scraper.export_data_via_ui(keyword)
            time.sleep(5)

        print("\n采集完成")
        input("按Enter键关闭浏览器...")

    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        if 'scraper' in locals():
            scraper.close()


if __name__ == "__main__":
    main()

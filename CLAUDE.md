# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A data science research project analyzing how "AI" related tags impact video views and engagement on Bilibili (B站). The project involves web scraping, NLP analysis, statistical modeling, and visualization.

Key research questions:
- Quantify tag impact on video views and danmaku (bullet comments) engagement
- Validate audience generalization effects (does tag influence decrease as videos trend?)
- Explore relationships between UP follower count, title-tag consistency, Baidu Index trends

## Project Structure

```
bilibili-ai-tag-analysis/
├── docs/
│   ├── step0_related_work.md      # Literature review requirements
│   ├── step1_data_collect.md      # Baidu Index + API pre-research
│   ├── step2_video_data_collect.md # Bilibili scraping specifications
│   ├── step3_pre-cleaned-data_eda.md # Pre-cleaning EDA requirements
│   └── step4-clean-data-eda.md    # Clean data EDA requirements
├── src/
│   ├── spiders/                   # Bilibili API & Baidu Index crawlers
│   └── analysis/                  # EDA, plotting, and reporting scripts
├── data/
│   ├── raw/                       # Video CSV, danmaku/, Baidu Index CSVs, progress JSONs
│   └── cleaned-data/              # Deduplicated/filtered dataset + matching danmaku copies
├── results/                       # Markdown reports and figures/
└── reports/                       # Final analysis reports
```

**Important**: Wait for explicit user instruction before starting each phase. Read the corresponding `docs/stepX_*.md` file before implementing.

## Code Architecture

### Spider System
There are multiple crawler implementations in `src/spiders/` for different anti-blocking strategies:
- **`bilibili_crawler_v2.py`** — Main async crawler with proxy rotation, fuzzy keyword matching, and danmaku batch supplementing. Uses `MAX_CONCURRENT=2`, random delays (2–5s), and saves progress to `crawler_progress_v2.json`.
- **`bilibili_crawler_sequential.py`** — Lower-aggression variant (`MAX_CONCURRENT=1`, 5–10s delays) that processes tags sequentially. Good when v2 triggers frequent `-799`/`-412` blocks. Saves to `crawler_progress_sequential.json`.
- **`bilibili_crawler_quarterly.py`** / **`bilibili_crawler_quarterly_fast.py`** — Time-windowed strategies that split the date range to improve sample distribution.
- **`test_bili_api.py`** — Mandatory pre-flight test script for all Bilibili API endpoints (single async calls, no loops).


All crawlers share these conventions:
- Hardcoded absolute base path: `D:/Claude_Code/bilibili-ai-tag-analysis`
- Proxy hardcoded to `http://127.0.0.1:7890` in sequential/quarterly variants
- Output to `data/raw/bilibili_video_info.csv` and `data/raw/danmaku/{bvid}.csv`
- Use `bilibili-api-python` library (`from bilibili_api import ...`)

### Analysis Pipeline
- **`src/analysis/eda_pre_clean.py`** — Pre-cleaning EDA on `data/raw/bilibili_video_info.csv`. Generates bar charts for tag distribution and view bins.
- **`src/analysis/step4_clean_eda.py`** — **Core cleaning script**. Reads raw data, deduplicates by `bvid`, filters (`stat_view >= 50000`, `stat_danmaku >= 50`), performs bidirectional matching with `danmaku/` files, copies matching data to `data/cleaned-data/`, and computes tag co-occurrence + top-video stats. **Never modifies `data/raw/`.**
- **`src/analysis/plot_trends.py`** — Loads `data/raw/baidu_index_2401_2601.csv`, detects peaks with Z-Score, annotates known AI events, outputs to `results/figures/`.
-  generates `results/baidu_index_real_data_report.md`. for AI related news and its baidu_index

### Data Model
- **Video info CSV fields**: `bvid`, `aid`, `cid`, `title`, `pubdate`, `tname`, `owner_mid`, `owner_fans`, `duration`, `desc`, `stat_view`, `stat_danmaku`, `stat_reply`, `stat_like`, `stat_coin`, `stat_favorite`, `stat_share`, `tags` (pipe-delimited `|`), `search_tag`
- **Danmaku CSV fields**: `danmaku_id`, `content`, `progress`, `ctime`, `user_hash`
- **Join keys**: `bvid` (video level), `cid` (danmaku/page level)

## Common Commands

```bash
# Test Bilibili API before any large-scale run
python src/spiders/test_bili_api.py

# Run main crawler variants
python src/spiders/bilibili_crawler_v2.py --mode all
python src/spiders/bilibili_crawler_sequential.py
python src/spiders/bilibili_crawler_quarterly.py

# Supplement missing danmaku only
python src/spiders/bilibili_crawler_v2.py --mode danmaku_only

# EDA and cleaning
python src/analysis/eda_pre_clean.py
python src/analysis/step4_clean_eda.py

# Baidu Index visualization
python src/analysis/plot_trends.py
```

## Coding Standards

### Web Scraping Requirements
All crawlers in `src/spiders/` must include:
- Async concurrency control (`asyncio` + `asyncio.Semaphore`)
- Random sleep/delays between requests
- Retry mechanism with exponential backoff
- User-agent rotation or proxy configuration
- Progress saved to JSON for crash recovery
- Real-time CSV appending (not in-memory batching)

### Data Consistency
- Join data sources on `bvid` or `cid`
- Log all missing value handling decisions
- Save intermediate processed data to appropriate `data/` subdirectories

### Statistical Analysis Requirements
- Control for confounding variables: video duration, publish time, UP follower count
- Report p-values and confidence intervals for all regression analyses
- Document effect sizes, not just significance

## Key Technical Constraints

1. **Python Environment**: Use `C:\Users\z8360\miniconda3\envs\bili\python.exe` only.
2. **API Limits**: Bilibili API has rate limits. `-799` = frequency control, `-412` = IP blocked, `-1200` = request degraded. All crawlers must back off exponentially and rotate proxies.
3. **Hardcoded Paths**: Many scripts contain absolute Windows paths to `D:/Claude_Code/bilibili-ai-tag-analysis`. Preserve them when editing related logic.
4. **Data Join Keys**: Primary keys are `bvid` (video ID) and `cid` (danmaku ID).
5. **External Data**: Baidu Index requires cookie-based authentication (`BDUSS`). Current crawlers do **not** use Bilibili `SESSDATA` (run in guest mode).
6. **Visualization**: Matplotlib fonts are hardcoded to `SimHei` / `Microsoft YaHei` for Chinese character rendering.
7. **Clean Data Red Line**: Scripts in the cleaning/analysis phase must **never** modify or delete files under `data/raw/`. Write all cleaned outputs to `data/cleaned-data/`.

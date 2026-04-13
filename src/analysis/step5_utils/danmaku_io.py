import os
import sqlite3
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from tqdm import tqdm

BASE_DIR = "D:/Claude_Code/bilibili-ai-tag-analysis"
DANMAKU_DIR = os.path.join(BASE_DIR, "data/cleaned-data/danmaku")
PARQUET_PATH = os.path.join(BASE_DIR, "data/processed/step5/shared/danmaku_core.parquet")

def init_checkpoint_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed (
            bvid TEXT PRIMARY KEY,
            task TEXT,
            mtime REAL
        )
    """)
    conn.commit()
    conn.close()

def list_unprocessed_bvids(bvid_list, task, db_path):
    if not os.path.exists(db_path):
        init_checkpoint_db(db_path)
        return bvid_list
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(bvid_list))
    cur.execute(
        f"SELECT bvid FROM processed WHERE task=? AND bvid IN ({placeholders})",
        [task] + list(bvid_list)
    )
    done = {row[0] for row in cur.fetchall()}
    conn.close()
    return [b for b in bvid_list if b not in done]

def mark_processed(bvid_list, task, db_path):
    if not bvid_list:
        return
    conn = sqlite3.connect(db_path)
    import time
    t = time.time()
    conn.executemany(
        "INSERT OR REPLACE INTO processed (bvid, task, mtime) VALUES (?, ?, ?)",
        [(b, task, t) for b in bvid_list]
    )
    conn.commit()
    conn.close()

def read_danmaku_csv(bvid, cols=None):
    path = os.path.join(DANMAKU_DIR, f"{bvid}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=cols) if cols else pd.read_csv(path)
        df["bvid"] = bvid
        return df
    except Exception as e:
        return None

def _worker(args):
    bvid, cols = args
    return read_danmaku_csv(bvid, cols)

def process_in_pool(worker_func, bvid_list, task, db_path, n_workers=8, chunksize=10):
    init_checkpoint_db(db_path)
    remaining = list_unprocessed_bvids(bvid_list, task, db_path)
    if not remaining:
        return []
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(worker_func, bvid): bvid for bvid in remaining}
        batch_done = []
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Task {task}"):
            bvid = futures[fut]
            try:
                res = fut.result()
                if res is not None:
                    results.append(res)
                batch_done.append(bvid)
                if len(batch_done) >= chunksize:
                    mark_processed(batch_done, task, db_path)
                    batch_done = []
            except Exception:
                pass
        if batch_done:
            mark_processed(batch_done, task, db_path)
    return results

def preload_danmaku_to_parquet(output_path=None, cols=None, n_workers=8):
    if output_path is None:
        output_path = PARQUET_PATH
    if os.path.exists(output_path):
        return output_path
    if cols is None:
        cols = ["danmaku_id", "content", "progress", "ctime", "user_hash"]
    files = glob.glob(os.path.join(DANMAKU_DIR, "*.csv"))
    bvids = [os.path.splitext(os.path.basename(f))[0] for f in files]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(read_danmaku_csv, bvid, cols): bvid for bvid in bvids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preload danmaku"):
            try:
                res = fut.result()
                if res is not None:
                    results.append(res)
            except Exception:
                pass
    if results:
        df = pd.concat(results, ignore_index=True)
        # Clean string columns for parquet compatibility
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna("").astype(str)
        df.to_parquet(output_path, index=False)
        return output_path
    return None

def load_danmaku_prefer_parquet(bvid=None, cols=None):
    if os.path.exists(PARQUET_PATH):
        if bvid is None:
            return pd.read_parquet(PARQUET_PATH, columns=cols)
        df = pd.read_parquet(PARQUET_PATH, columns=cols)
        return df[df["bvid"] == bvid].copy()
    return read_danmaku_csv(bvid, cols)

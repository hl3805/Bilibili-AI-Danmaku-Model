import sys
sys.path.insert(0, "src")
from analysis.step5_utils.danmaku_io import preload_danmaku_to_parquet

if __name__ == "__main__":
    path = preload_danmaku_to_parquet()
    print("Parquet path:", path)

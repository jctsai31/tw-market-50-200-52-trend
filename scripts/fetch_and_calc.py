"""
Taiwan Stock Market Breadth Indicator
每日增量更新腳本 - 由 GitHub Actions 執行
計算：
  1. 收盤價 > 50MA / 200MA 的個股數
  2. 52週新高 / 新低的個股數
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# ── 設定 ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
FINMIND_BASE  = "https://api.finmindtrade.com/api/v4/data"
DATA_DIR      = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MA_FILE      = DATA_DIR / "breadth_ma.json"
WEEK52_FILE  = DATA_DIR / "breadth_52week.json"

# 計算 200MA 需要額外的歷史資料，所以抓 2 年
FETCH_DAYS   = 730   # 抓取原始資料的天數（約 2 年）
CHART_DAYS   = 365   # 圖表顯示最近 1 年
TRADING_DAYS_200MA = 200
TRADING_DAYS_52W   = 252


# ── FinMind API 工具函式 ───────────────────────────────────────────────

def finmind_get(dataset: str, params: dict, retry: int = 3) -> list:
    """呼叫 FinMind API，回傳 data list；失敗自動重試。"""
    p = {"token": FINMIND_TOKEN, "dataset": dataset, **params}
    for attempt in range(1, retry + 1):
        try:
            r = requests.get(FINMIND_BASE, params=p, timeout=30)
            r.raise_for_status()
            j = r.json()
            if j.get("status") == 200:
                return j.get("data", [])
            log.warning("API status %s: %s", j.get("status"), j.get("msg"))
            return []
        except Exception as e:
            log.warning("Attempt %d/%d failed: %s", attempt, retry, e)
            time.sleep(2 ** attempt)
    return []


# ── 個股清單（含過濾邏輯）────────────────────────────────────────────

def get_stock_list() -> pd.DataFrame:
    """
    取得上市+上櫃個股清單，過濾掉 ETF / 特別股 / 基金。
    過濾策略：
      [1] 代號規則：4碼純數字（普通股）
      [2] FinMind 類別欄位：排除含 ETF / 基金 / 受益憑證 的類別
    """
    data = finmind_get("TaiwanStockInfo", {})
    if not data:
        raise RuntimeError("無法取得個股清單")

    df = pd.DataFrame(data)
    log.info("原始個股清單：%d 筆", len(df))

    # ── 過濾 [1]：代號為 4 碼純數字（排除 ETF 6碼、特別股含英文）
    df = df[df["stock_id"].str.match(r"^\d{4}$")]
    log.info("代號規則過濾後：%d 筆", len(df))

    # ── 過濾 [2]：用 industry_category / type 欄位排除非普通股
    exclude_keywords = ["ETF", "基金", "受益憑證", "存託憑證", "特別股"]
    if "industry_category" in df.columns:
        pattern = "|".join(exclude_keywords)
        df = df[~df["industry_category"].str.contains(pattern, na=False)]
    if "type" in df.columns:
        df = df[~df["type"].str.contains("|".join(exclude_keywords), na=False)]

    log.info("類別欄位過濾後：%d 筆", len(df))
    return df[["stock_id", "stock_name"]].reset_index(drop=True)


# ── 取得歷史日K ───────────────────────────────────────────────────────

def fetch_stock_price(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """取得單支股票的日K資料，回傳含 date / close 欄位的 DataFrame。"""
    data = finmind_get("TaiwanStockPrice", {
        "stock_id":   stock_id,
        "start_date": start_date,
        "end_date":   end_date,
    })
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)[["date", "close"]].copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


# ── 取得台灣交易日曆 ─────────────────────────────────────────────────

def get_trading_dates(start_date: str, end_date: str) -> list[str]:
    """透過大盤指數取得交易日列表（TAIEX）。"""
    data = finmind_get("TaiwanStockPrice", {
        "stock_id":   "^TWII",   # TAIEX
        "start_date": start_date,
        "end_date":   end_date,
    })
    if not data:
        # fallback：用週一到週五排除假日（較不精確）
        dates = pd.bdate_range(start_date, end_date)
        return [d.strftime("%Y-%m-%d") for d in dates]
    df = pd.DataFrame(data)
    return sorted(df["date"].unique().tolist())


def get_last_trading_date() -> str:
    """找最近一個交易日（執行當天若是假日則往前找）。"""
    today = datetime.now()
    start = (today - timedelta(days=10)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
    dates = get_trading_dates(start, end)
    if not dates:
        # fallback：往前找最近週五
        d = today
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d.strftime("%Y-%m-%d")
    return dates[-1]


# ── 核心計算 ──────────────────────────────────────────────────────────

def calculate_breadth(stocks: pd.DataFrame, start_date: str, end_date: str) -> tuple[list, list]:
    """
    對所有個股計算每個交易日的廣度指標。
    回傳 (ma_records, week52_records)
    """
    log.info("開始抓取 %d 支個股資料，區間 %s ~ %s", len(stocks), start_date, end_date)

    # 收集所有個股的收盤價
    all_close = {}   # stock_id -> DataFrame(date, close)
    total = len(stocks)

    for i, row in stocks.iterrows():
        sid = row["stock_id"]
        df  = fetch_stock_price(sid, start_date, end_date)
        if not df.empty:
            all_close[sid] = df.set_index("date")["close"]
        if (i + 1) % 50 == 0:
            log.info("進度：%d / %d", i + 1, total)
        time.sleep(0.3)   # 避免 rate limit

    if not all_close:
        log.error("無法取得任何個股資料")
        return [], []

    # 合併成大寬表：index=date, columns=stock_id
    price_df = pd.DataFrame(all_close).sort_index()
    log.info("合併後資料維度：%s", price_df.shape)

    # 取最後 CHART_DAYS 個交易日做輸出（前面保留是為了算 MA）
    all_dates    = price_df.index
    chart_start  = pd.Timestamp(end_date) - pd.DateOffset(years=1)
    output_dates = all_dates[all_dates >= chart_start]

    ma_records    = []
    week52_records = []

    for dt in output_dates:
        loc = price_df.index.get_loc(dt)

        # ── MA 計算
        # 取到目前為止的資料
        window_200 = price_df.iloc[max(0, loc - TRADING_DAYS_200MA + 1): loc + 1]
        window_50  = price_df.iloc[max(0, loc - 50 + 1): loc + 1]

        close_today = price_df.loc[dt]
        ma50        = window_50.mean()
        ma200       = window_200.mean()

        valid_mask  = close_today.notna() & ma50.notna() & ma200.notna()
        total_count = valid_mask.sum()
        above50     = int((close_today[valid_mask] > ma50[valid_mask]).sum())
        above200    = int((close_today[valid_mask] > ma200[valid_mask]).sum())

        ma_records.append({
            "date":      dt.strftime("%Y-%m-%d"),
            "above50ma": above50,
            "above200ma": above200,
            "total":     int(total_count),
        })

        # ── 52週新高低
        window_52w = price_df.iloc[max(0, loc - TRADING_DAYS_52W + 1): loc + 1]
        high_52w   = window_52w.max()
        low_52w    = window_52w.min()

        valid52 = close_today.notna() & high_52w.notna() & low_52w.notna()
        new_high = int((close_today[valid52] >= high_52w[valid52]).sum())
        new_low  = int((close_today[valid52] <= low_52w[valid52]).sum())

        week52_records.append({
            "date":    dt.strftime("%Y-%m-%d"),
            "high52w": new_high,
            "low52w":  new_low,
            "total":   int(valid52.sum()),
        })

    log.info("計算完成，共 %d 個交易日", len(ma_records))
    return ma_records, week52_records


# ── JSON 讀寫（增量合併）─────────────────────────────────────────────

def load_json(path: Path) -> list:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    log.info("已儲存 %s（%d 筆）", path.name, len(records))


def merge_records(existing: list, new_records: list) -> list:
    """合併新舊資料，以 date 為 key，新資料覆蓋舊資料，保持日期排序。"""
    combined = {r["date"]: r for r in existing}
    for r in new_records:
        combined[r["date"]] = r
    return sorted(combined.values(), key=lambda x: x["date"])


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    log.info("===== Taiwan Stock Breadth Updater =====")

    # 決定日期區間
    end_date   = get_last_trading_date()
    # 往前抓 FETCH_DAYS 天（含假日，所以交易日會少一些），確保 200MA 資料足夠
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%d")
    log.info("資料區間：%s ~ %s（最後交易日）", start_date, end_date)

    # 讀取既有資料，判斷是否需要更新
    existing_ma    = load_json(MA_FILE)
    existing_52w   = load_json(WEEK52_FILE)
    latest_date    = existing_ma[-1]["date"] if existing_ma else ""

    if latest_date == end_date:
        log.info("資料已是最新（%s），無需更新。", end_date)
        return

    log.info("需要更新至 %s（現有最新：%s）", end_date, latest_date or "無")

    # 取得個股清單
    stocks = get_stock_list()

    # 計算廣度指標
    ma_records, week52_records = calculate_breadth(stocks, start_date, end_date)

    if not ma_records:
        log.error("計算結果為空，結束。")
        return

    # 合併並儲存
    merged_ma   = merge_records(existing_ma,  ma_records)
    merged_52w  = merge_records(existing_52w, week52_records)
    save_json(MA_FILE,     merged_ma)
    save_json(WEEK52_FILE, merged_52w)

    log.info("===== 更新完成 =====")


if __name__ == "__main__":
    main()

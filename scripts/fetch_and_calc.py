"""
Taiwan Stock Market Breadth Indicator
每日增量更新腳本 - 由 GitHub Actions 執行
使用 Yahoo Finance 取得資料，完全免費不需要 Token
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
import yfinance as yf

# ── 設定 ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MA_FILE     = DATA_DIR / "breadth_ma.json"
WEEK52_FILE = DATA_DIR / "breadth_52week.json"

FETCH_DAYS         = 730
TRADING_DAYS_200MA = 200
TRADING_DAYS_52W   = 252


# ── 個股清單（TWSE + TPEX 官方 API）─────────────────────────────────

def get_twse_stocks() -> pd.DataFrame:
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    try:
        r = requests.get(url, timeout=30)
        df = pd.DataFrame(r.json())
        df = df[df["Code"].str.match(r"^\d{4}$")][["Code", "Name"]].copy()
        df.columns = ["stock_id", "stock_name"]
        df["suffix"] = ".TW"
        return df
    except Exception as e:
        log.error("TWSE API 失敗: %s", e)
        return pd.DataFrame()


def get_tpex_stocks() -> pd.DataFrame:
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
    try:
        r = requests.get(url, timeout=30)
        df = pd.DataFrame(r.json())
        df = df[df["SecuritiesCompanyCode"].str.match(r"^\d{4}$")][
            ["SecuritiesCompanyCode", "CompanyName"]
        ].copy()
        df.columns = ["stock_id", "stock_name"]
        df["suffix"] = ".TWO"
        return df
    except Exception as e:
        log.error("TPEX API 失敗: %s", e)
        return pd.DataFrame()


def get_stock_list() -> pd.DataFrame:
    twse = get_twse_stocks()
    tpex = get_tpex_stocks()
    stocks = pd.concat([twse, tpex], ignore_index=True)
    stocks["yf_symbol"] = stocks["stock_id"] + stocks["suffix"]
    log.info("個股清單：上市 %d + 上櫃 %d = 合計 %d 筆", len(twse), len(tpex), len(stocks))
    return stocks


# ── yfinance 批次抓取 ─────────────────────────────────────────────────

def fetch_prices(symbols: list, start_date: str, end_date: str, batch_size: int = 100) -> pd.DataFrame:
    all_close = {}

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            raw = yf.download(
                " ".join(batch),
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                close_df = raw["Close"]
            else:
                close_df = raw[["Close"]]
                close_df.columns = batch

            for sym in batch:
                if sym in close_df.columns:
                    s = close_df[sym].dropna()
                    if len(s) > 50:
                        all_close[sym] = s
        except Exception as e:
            log.warning("批次 %d 失敗: %s", i // batch_size + 1, e)

        done = min(i + batch_size, len(symbols))
        if done % 500 == 0 or done == len(symbols):
            log.info("下載進度：%d / %d，成功 %d 支", done, len(symbols), len(all_close))
        time.sleep(0.5)

    return pd.DataFrame(all_close).sort_index()


# ── 核心計算 ──────────────────────────────────────────────────────────

def calculate_breadth(price_df: pd.DataFrame, end_date: str) -> tuple[list, list]:
    # 用 price_df 實際有的最後日期，不依賴傳入的 end_date
    actual_end = price_df.index[-1]
    log.info("yfinance 實際最新資料日期：%s", actual_end.date())
    chart_start  = actual_end - pd.DateOffset(years=1)
    output_dates = price_df.index[price_df.index >= chart_start]
    log.info("計算區間：%s ~ %s，共 %d 個交易日", output_dates[0].date(), output_dates[-1].date(), len(output_dates))

    ma_records     = []
    week52_records = []

    for n, dt in enumerate(output_dates):
        loc = price_df.index.get_loc(dt)

        # ── MA
        w50         = price_df.iloc[max(0, loc - 49):loc + 1]
        w200        = price_df.iloc[max(0, loc - 199):loc + 1]
        close_today = price_df.loc[dt]
        ma50        = w50.mean()
        ma200       = w200.mean()
        valid       = close_today.notna() & ma50.notna() & ma200.notna()
        ma_records.append({
            "date":       dt.strftime("%Y-%m-%d"),
            "above50ma":  int((close_today[valid] > ma50[valid]).sum()),
            "above200ma": int((close_today[valid] > ma200[valid]).sum()),
            "total":      int(valid.sum()),
        })

        # ── 52週
        w52w = price_df.iloc[max(0, loc - 251):loc + 1]
        h52w = w52w.max()
        l52w = w52w.min()
        v52  = close_today.notna() & h52w.notna() & l52w.notna()
        week52_records.append({
            "date":    dt.strftime("%Y-%m-%d"),
            "high52w": int((close_today[v52] >= h52w[v52]).sum()),
            "low52w":  int((close_today[v52] <= l52w[v52]).sum()),
            "total":   int(v52.sum()),
        })

    log.info("計算完成，共 %d 筆", len(ma_records))
    return ma_records, week52_records


# ── JSON 讀寫 ─────────────────────────────────────────────────────────

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
    combined = {r["date"]: r for r in existing}
    for r in new_records:
        combined[r["date"]] = r
    return sorted(combined.values(), key=lambda x: x["date"])


def get_latest_trading_date() -> str:
    """從 TWSE 官方 API 取得最新交易日（最準確，當天收盤後即更新）"""
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        if data:
            # TWSE API 回傳的資料含有 Date 欄位，格式為 YYYYMMDD
            date_str = data[0].get("Date", "")
            if len(date_str) == 8:
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    except Exception as e:
        log.warning("TWSE 取得交易日失敗: %s", e)
    # fallback：用昨天（避開 yfinance 延遲）
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    log.info("===== Taiwan Stock Breadth Updater (yfinance) =====")

    # 從 TWSE 取得最新交易日（比用 datetime.now() 更準確）
    end_date   = get_latest_trading_date()
    today      = datetime.now()
    start_date = (today - timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%d")
    log.info("資料區間：%s ~ %s（最新交易日）", start_date, end_date)

    # 檢查是否需要更新
    existing_ma  = load_json(MA_FILE)
    existing_52w = load_json(WEEK52_FILE)
    latest_date  = existing_ma[-1]["date"] if existing_ma else ""
    log.info("現有最新資料：%s，TWSE 最新交易日：%s", latest_date or "無", end_date)

    # 取得個股清單
    stocks  = get_stock_list()
    symbols = stocks["yf_symbol"].tolist()

    # 抓取收盤價
    price_df = fetch_prices(symbols, start_date, end_date)
    if price_df.empty:
        log.error("無法取得資料，結束。")
        return
    log.info("合併後資料：%d 個交易日 × %d 支個股", *price_df.shape)

    # 計算廣度指標
    ma_records, week52_records = calculate_breadth(price_df, end_date)

    # 合併並儲存
    save_json(MA_FILE,     merge_records(existing_ma,  ma_records))
    save_json(WEEK52_FILE, merge_records(existing_52w, week52_records))

    log.info("===== 更新完成 =====")


if __name__ == "__main__":
    main()

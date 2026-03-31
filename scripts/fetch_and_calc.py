"""
Taiwan Stock Market Breadth Indicator
每日增量更新腳本 - 由 GitHub Actions 執行

架構：
  - 歷史資料（2年）：yfinance 批次下載（初始化時使用）
  - 每日增量更新：TWSE + TPEX 官方 API（當天收盤後即更新，無延遲）
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
import yfinance as yf

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

FETCH_DAYS = 730   # yfinance 歷史資料抓取天數


# ══════════════════════════════════════════════════════════════
#  TWSE / TPEX 官方 API
# ══════════════════════════════════════════════════════════════

def fetch_twse_today() -> tuple[str, dict]:
    """
    從 TWSE 官方 API 取得今日所有上市股票收盤價。
    回傳 (date_str, {stock_id: close_price})
    """
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        if not data:
            return "", {}

        # 取得日期（格式 YYYYMMDD）
        date_raw = data[0].get("Date", "")
        if len(date_raw) == 8:
            date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
        else:
            return "", {}

        # 取得收盤價（只保留 4 碼純數字普通股）
        prices = {}
        for s in data:
            code = s.get("Code", "")
            if not code or not code.isdigit() or len(code) != 4:
                continue
            try:
                close = float(s.get("ClosingPrice", "").replace(",", ""))
                if close > 0:
                    prices[code] = close
            except (ValueError, AttributeError):
                continue

        log.info("TWSE 上市：日期 %s，取得 %d 筆收盤價", date_str, len(prices))
        return date_str, prices

    except Exception as e:
        log.error("TWSE API 失敗: %s", e)
        return "", {}


def fetch_tpex_today() -> tuple[str, dict]:
    """
    從 TPEX 官方 API 取得今日所有上櫃股票收盤價。
    回傳 (date_str, {stock_id: close_price})
    """
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        if not data:
            return "", {}

        # 取得日期（欄位名稱可能是 Date 或 date）
        date_raw = data[0].get("Date", data[0].get("date", ""))
        # TPEX 日期格式可能是 YYYY/MM/DD 或 YYYYMMDD
        date_str = ""
        if "/" in date_raw:
            parts = date_raw.split("/")
            if len(parts) == 3:
                date_str = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        elif len(date_raw) == 8 and date_raw.isdigit():
            date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"

        prices = {}
        for s in data:
            code = s.get("SecuritiesCompanyCode", "")
            if not code or not code.isdigit() or len(code) != 4:
                continue
            try:
                close = float(s.get("Close", "").replace(",", ""))
                if close > 0:
                    prices[code] = close
            except (ValueError, AttributeError):
                continue

        log.info("TPEX 上櫃：日期 %s，取得 %d 筆收盤價", date_str, len(prices))
        return date_str, prices

    except Exception as e:
        log.error("TPEX API 失敗: %s", e)
        return "", {}


# ══════════════════════════════════════════════════════════════
#  yfinance 歷史資料（初始化用）
# ══════════════════════════════════════════════════════════════

def get_stock_list_for_yf() -> list[str]:
    """取得所有上市+上櫃 4 碼股票的 yfinance 代號"""
    symbols = []
    try:
        r = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=30)
        for s in r.json():
            code = s.get("Code", "")
            if code.isdigit() and len(code) == 4:
                symbols.append(code + ".TW")
    except Exception as e:
        log.warning("TWSE 清單失敗: %s", e)

    try:
        r = requests.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes", timeout=30)
        for s in r.json():
            code = s.get("SecuritiesCompanyCode", "")
            if code.isdigit() and len(code) == 4:
                symbols.append(code + ".TWO")
    except Exception as e:
        log.warning("TPEX 清單失敗: %s", e)

    log.info("yfinance 代號清單：%d 支", len(symbols))
    return symbols


def fetch_history_yfinance(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """用 yfinance 批次下載歷史收盤價"""
    all_close = {}
    batch_size = 100

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
            log.info("yfinance 下載進度：%d / %d，成功 %d 支", done, len(symbols), len(all_close))
        time.sleep(0.5)

    return pd.DataFrame(all_close).sort_index()


# ══════════════════════════════════════════════════════════════
#  廣度指標計算
# ══════════════════════════════════════════════════════════════

def calc_one_day(price_df: pd.DataFrame, dt: pd.Timestamp) -> tuple[dict, dict]:
    """計算單一交易日的廣度指標"""
    loc = price_df.index.get_loc(dt)

    # MA
    w50  = price_df.iloc[max(0, loc - 49):loc + 1]
    w200 = price_df.iloc[max(0, loc - 199):loc + 1]
    close_today = price_df.loc[dt]
    ma50  = w50.mean()
    ma200 = w200.mean()
    valid = close_today.notna() & ma50.notna() & ma200.notna()
    ma_rec = {
        "date":       dt.strftime("%Y-%m-%d"),
        "above50ma":  int((close_today[valid] > ma50[valid]).sum()),
        "above200ma": int((close_today[valid] > ma200[valid]).sum()),
        "total":      int(valid.sum()),
    }

    # 52週
    w52w = price_df.iloc[max(0, loc - 251):loc + 1]
    h52w = w52w.max()
    l52w = w52w.min()
    v52  = close_today.notna() & h52w.notna() & l52w.notna()
    w52_rec = {
        "date":    dt.strftime("%Y-%m-%d"),
        "high52w": int((close_today[v52] >= h52w[v52]).sum()),
        "low52w":  int((close_today[v52] <= l52w[v52]).sum()),
        "total":   int(v52.sum()),
    }

    return ma_rec, w52_rec


def calculate_breadth_full(price_df: pd.DataFrame) -> tuple[list, list]:
    """計算 price_df 最後 1 年所有交易日的廣度指標"""
    actual_end   = price_df.index[-1]
    chart_start  = actual_end - pd.DateOffset(years=1)
    output_dates = price_df.index[price_df.index >= chart_start]
    log.info("計算區間：%s ~ %s，共 %d 個交易日",
             output_dates[0].date(), output_dates[-1].date(), len(output_dates))

    ma_records, week52_records = [], []
    for n, dt in enumerate(output_dates):
        ma_rec, w52_rec = calc_one_day(price_df, dt)
        ma_records.append(ma_rec)
        week52_records.append(w52_rec)

    log.info("計算完成，共 %d 筆", len(ma_records))
    return ma_records, week52_records


# ══════════════════════════════════════════════════════════════
#  JSON 讀寫
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════

def main():
    log.info("===== Taiwan Stock Breadth Updater =====")

    # ── Step 1：從 TWSE/TPEX 取得今日收盤價 ──
    twse_date, twse_prices = fetch_twse_today()
    tpex_date, tpex_prices = fetch_tpex_today()

    # 確認今日交易日期
    today_date = twse_date or tpex_date
    if not today_date:
        log.error("無法取得今日交易日期，結束。")
        return
    log.info("今日交易日：%s", today_date)

    # ── Step 2：讀取既有資料 ──
    existing_ma  = load_json(MA_FILE)
    existing_52w = load_json(WEEK52_FILE)
    latest_date  = existing_ma[-1]["date"] if existing_ma else ""
    log.info("現有最新資料：%s", latest_date or "無")

    if latest_date == today_date:
        log.info("資料已是最新（%s），無需更新。", today_date)
        return

    # ── Step 3：用 yfinance 抓歷史資料建立 price_df ──
    # 需要 2 年歷史才能算 200MA 和 52 週新高低
    today      = datetime.now()
    start_date = (today - timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%d")
    end_date   = today.strftime("%Y-%m-%d")

    symbols  = get_stock_list_for_yf()
    price_df = fetch_history_yfinance(symbols, start_date, end_date)

    if price_df.empty:
        log.error("yfinance 無法取得歷史資料，結束。")
        return
    log.info("yfinance 歷史資料：%d 個交易日 × %d 支個股", *price_df.shape)

    # ── Step 4：用今日 TWSE/TPEX 收盤價覆蓋/新增最新一天 ──
    # 合併上市+上櫃收盤價（轉成 yfinance 代號格式）
    today_prices = {}
    for code, price in twse_prices.items():
        today_prices[code + ".TW"] = price
    for code, price in tpex_prices.items():
        today_prices[code + ".TWO"] = price

    log.info("今日 TWSE+TPEX 收盤價：%d 支", len(today_prices))

    # 建立今日資料列，只填入 price_df 中有的欄位
    today_ts = pd.Timestamp(today_date)
    today_row = pd.Series(
        {sym: today_prices[sym] for sym in price_df.columns if sym in today_prices},
        name=today_ts
    )

    # 如果 price_df 已有今日資料就覆蓋，否則新增
    if today_ts in price_df.index:
        price_df.loc[today_ts] = today_row
        log.info("覆蓋 yfinance 今日資料（%s）", today_date)
    else:
        price_df = pd.concat([price_df, today_row.to_frame().T])
        price_df = price_df.sort_index()
        log.info("新增今日資料（%s）到 price_df", today_date)

    log.info("更新後 price_df 最新日期：%s", price_df.index[-1].date())

    # ── Step 5：計算廣度指標 ──
    ma_records, week52_records = calculate_breadth_full(price_df)

    # ── Step 6：合併並儲存 ──
    save_json(MA_FILE,     merge_records(existing_ma,  ma_records))
    save_json(WEEK52_FILE, merge_records(existing_52w, week52_records))

    log.info("===== 更新完成（%s）=====", today_date)


if __name__ == "__main__":
    main()

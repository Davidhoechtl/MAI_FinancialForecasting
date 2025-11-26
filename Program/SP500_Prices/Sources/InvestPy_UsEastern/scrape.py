import pandas as pd
import investpy
from datetime import datetime, timezone
import os
import time

CACHE_FILE_PATH = 'SP500_Prices/Sources/InvestPy_UsEastern/sp500_etf_daily.csv'
def get_sp500_data(start, end, max_retries=3, verbose=True):
    """
    Fetch daily SPDR S&P 500 ETF data between start and end
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if start_ts > end_ts:
        raise ValueError("start must be before or equal to end")

    if os.path.exists(CACHE_FILE_PATH):
        try:
            start_ts = pd.to_datetime(start).tz_localize("US/Eastern")
            end_ts = pd.to_datetime(end).tz_localize("US/Eastern")

            cached_df = pd.read_csv(CACHE_FILE_PATH, index_col=0)

            # Ensure the index is timezone-aware in US/Eastern
            cached_df.index = pd.to_datetime(cached_df.index, utc=True, errors='coerce')
            cached_df.index = cached_df.index.tz_convert('US/Eastern')

            # Slice to requested window
            df_slice = cached_df.loc[start_ts:end_ts]

            result = df_slice[['Close', 'Pct_Change', 'Volume']]
            return result
        except Exception as exc:
            raise Exception(f"[ERROR]: failed to load cache {CACHE_FILE_PATH}: {exc}")
    else:
        if verbose:
            print("Cache not used or not found, fetching data directly.")
        df = scrape_all(
            start,
            end,
            chunk_days=-1,
            verbose=verbose,
            max_retries=max_retries
        )
        return df[['Close', 'Pct_Change', 'Volume']]

def get_sp500_data_weekly(start, end):
    import investpy
    import pandas as pd

    df = investpy.etfs.get_etf_historical_data(
        etf='SPDR S&P 500',
        country='United States',
        from_date=start,
        to_date=end
    )

    # Ensure DateTimeIndex with timezone
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('US/Eastern')

    # Daily percent change
    df["Pct_Change"] = df["Close"].pct_change()

    # Weekly resample: take last available close per week (Friday or last trading day)
    df_weekly = df.resample("W").last()

    # Compute weekly percent change if desired
    df_weekly["Pct_Change"] = df_weekly["Close"].pct_change()

    return df_weekly[["Close", "Pct_Change"]]

def scrape_all(start, end, chunk_days=365, verbose=True, max_retries=3, stop_on_fail=True):
    """Scrape daily SPDR S&P 500 ETF data between start and end (inclusive).

    The function chunks the date range into pieces of `chunk_days` days to
    avoid API limits or large single requests. It concatenates responses,
    sorts by index, drops duplicates and writes to CSV if outpath is provided.

    Args:
      start, end: date-like (str or datetime). Strings can be ISO (YYYY-MM-DD)
        or dd/mm/YYYY. They will be converted for investpy.
      chunk_days: int, size of each chunk in days. Default 365.
      outpath: optional path to save CSV. If None, returns DataFrame only.
      verbose: bool, print progress.
      max_retries: int, number of retry attempts per chunk on failure.

    Returns:
      pandas.DataFrame with the concatenated daily data (index is tz-aware DateTimeIndex).
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if start_ts > end_ts:
        raise ValueError("start must be before or equal to end")

    # Build chunk boundaries
    ranges = []
    if chunk_days <= 0:
        ranges.append((start_ts, end_ts))
    else:
        cur_start = start_ts
        while cur_start <= end_ts:
            cur_end = min(cur_start + pd.Timedelta(days=chunk_days - 1), end_ts)
            ranges.append((cur_start, cur_end))
            cur_start = cur_end + pd.Timedelta(days=1)

    if verbose:
        print(f"Will fetch {len(ranges)} chunks from {start_ts.date()} to {end_ts.date()}")

    dfs = []
    for i, (s, e) in enumerate(ranges, 1):
        s_str = _to_ddmmyyyy(s)
        e_str = _to_ddmmyyyy(e)
        if verbose:
            print(f"Fetching chunk {i}/{len(ranges)}: {s_str} -> {e_str}")

        # retry loop for this chunk
        success = False
        stop = False
        for attempt in range(1, max_retries + 1):
            try:
                df_chunk = fetch_sp500_data(s_str, e_str)
                if df_chunk is None or len(df_chunk) == 0:
                    if verbose:
                        print(f"Chunk {i} returned no rows.")
                    # treat as success (no data) and break retries
                    success = True
                    break
                dfs.append(df_chunk)
                success = True
                if verbose:
                    print(f"Chunk {i} fetched {len(df_chunk)} rows.")
                break
            except Exception as exc:
                # log and backoff, then retry
                print(f"Warning: failed attempt {attempt}/{max_retries} for chunk {s_str} - {e_str}: {exc}")
                if attempt < max_retries:
                    backoff = 2 ** (attempt - 1)
                    if verbose:
                        print(f"Retrying after {backoff} seconds...")
                    time.sleep(backoff)
                else:
                    if stop_on_fail:
                        print("Max retries reached, stopping.")
                        stop = True
                    print(f"Giving up on chunk {s_str} - {e_str} after {max_retries} attempts")

        if stop:
            break

        # continue to next chunk even if this one failed
        if not success:
            if verbose:
                print(f"Skipping chunk {i} due to repeated failures.")
            continue

    if not dfs:
        raise RuntimeError("No data fetched for the given date range")

    # Concatenate, sort index, drop duplicate indices
    df_all = pd.concat(dfs)
    # If indexes are naive vs tz-aware mismatch, ensure they're all tz-aware
    if df_all.index.tz is None:
        df_all.index = df_all.index.tz_localize('US/Eastern')
    # Sort and dedupe
    df_all = df_all.sort_index()
    df_all = df_all[~df_all.index.duplicated(keep='last')]

    # Calculate daily percent change
    df_all['Pct_Change'] = df_all['Close'].pct_change()

    df_all.to_csv(CACHE_FILE_PATH, index=True)
    if verbose:
        print(f"Saved {len(df_all)} rows to {CACHE_FILE_PATH}")

    return df_all
def fetch_sp500_data(start_date, end_date):
    df = investpy.etfs.get_etf_historical_data(
        etf='SPDR S&P 500',
        country='United States',
        from_date=start_date,
        to_date=end_date
    )

    print(df.head(10))
    # Ensure the index is datetime and localize to US/Eastern
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('US/Eastern')  # directly localize since it's naive

    return df
def _to_ddmmyyyy(dt):
    """Convert a date-like (str/Datetime) to dd/mm/YYYY string for investpy."""
    if isinstance(dt, str):
        # accept ISO or dd/mm/YYYY, try parse
        parsed = pd.to_datetime(dt, dayfirst=False, errors='coerce')
        if pd.isna(parsed):
            parsed = pd.to_datetime(dt, dayfirst=True, errors='coerce')
        if pd.isna(parsed):
            raise ValueError(f"Unable to parse date string: {dt}")
        dt = parsed
    if isinstance(dt, (pd.Timestamp, datetime)):
        return pd.Timestamp(dt).strftime("%d/%m/%Y")
    raise TypeError("dt must be a str or Timestamp/datetime")

if __name__ == '__main__':
    os.chdir("D:/Studium/Master/Masterarbeit/MAI_FinancialForecasting/Program")

    start = "01/01/2010"
    end = "31/12/2022"
    out = "sp500_etf_daily.csv"
    chunk_days = 360

    try:
        df = scrape_all(start, end, chunk_days=chunk_days, verbose=True, max_retries=8)
    except Exception as e:
        print(f"Scrape failed: {e}")
        raise

#python Program/SP500_Prices/Sources/InvestPy_UsEastern/scrape.py
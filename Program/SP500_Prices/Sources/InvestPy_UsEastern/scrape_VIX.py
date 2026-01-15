import pandas as pd
import investpy
from datetime import datetime
import os
import time

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Update the cache path to reflect VIX data
CACHE_FILE_PATH = 'SP500_Prices/Sources/InvestPy_UsEastern/sp500_vix_daily.csv'

def get_vix_data(start, end, max_retries=3, verbose=True):
    """
    Fetch daily VIX index data between start and end
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    if start_ts > end_ts:
        raise ValueError("start must be before or equal to end")

    if os.path.exists(CACHE_FILE_PATH):
        try:
            # Ensure folder exists for reading if path is complex
            start_ts = pd.to_datetime(start).tz_localize("US/Eastern")
            end_ts = pd.to_datetime(end).tz_localize("US/Eastern")

            cached_df = pd.read_csv(CACHE_FILE_PATH, index_col=0)
            cached_df.index = pd.to_datetime(cached_df.index, utc=True, errors='coerce')
            cached_df.index = cached_df.index.tz_convert('US/Eastern')

            df_slice = cached_df.loc[start_ts:end_ts]

            # rename close colume to vix_close
            df_slice.rename(columns={'Close': 'VIX_Close'}, inplace=True)

            return df_slice[['VIX_Close']]
        except Exception as exc:
            raise Exception(f"[ERROR]: failed to load cache {CACHE_FILE_PATH}: {exc}")
    else:
        if verbose:
            print("Cache not used or not found, fetching data directly.")
        df = scrape_all_vix(
            start,
            end,
            chunk_days=-1,  # -1 fetches the whole range at once
            verbose=verbose,
            max_retries=max_retries
        )

        # rename close colume to vix_close
        df.rename(columns={'Close': 'VIX_Close'}, inplace=True)

        return df[['VIX_Close']]


def scrape_all_vix(start, end, chunk_days=365, verbose=True, max_retries=3, stop_on_fail=True):
    """Scrape daily VIX data between start and end."""
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # Ensure directory exists for the cache file
    os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)

    ranges = []
    if chunk_days <= 0:
        ranges.append((start_ts, end_ts))
    else:
        cur_start = start_ts
        while cur_start <= end_ts:
            cur_end = min(cur_start + pd.Timedelta(days=chunk_days - 1), end_ts)
            ranges.append((cur_start, cur_end))
            cur_start = cur_end + pd.Timedelta(days=1)

    dfs = []
    for i, (s, e) in enumerate(ranges, 1):
        s_str = _to_ddmmyyyy(s)
        e_str = _to_ddmmyyyy(e)

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                # Target 'VIX' as an index in the United States
                df_chunk = fetch_vix_data(s_str, e_str)
                if df_chunk is not None and len(df_chunk) > 0:
                    dfs.append(df_chunk)
                success = True
                break
            except Exception as exc:
                print(f"Warning: failed attempt {attempt}/{max_retries}: {exc}")
                time.sleep(2 ** (attempt - 1))

    if not dfs:
        raise RuntimeError("No data fetched for the given date range")

    df_all = pd.concat(dfs)
    if df_all.index.tz is None:
        df_all.index = df_all.index.tz_localize('US/Eastern')

    df_all = df_all.sort_index()
    df_all = df_all[~df_all.index.duplicated(keep='last')]
    df_all['Pct_Change'] = df_all['Close'].pct_change()

    df_all.to_csv(CACHE_FILE_PATH, index=True)
    return df_all


def fetch_vix_data(start_date, end_date):
    """Uses investpy.indices.get_index_historical_data for VIX."""
    df = investpy.indices.get_index_historical_data(
        index='S&P 500 VIX',  # Change: index name
        country='United States',
        from_date=start_date,
        to_date=end_date
    )
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('US/Eastern')
    return df


def _to_ddmmyyyy(dt):
    if isinstance(dt, str):
        parsed = pd.to_datetime(dt, dayfirst=False, errors='coerce')
        if pd.isna(parsed):
            parsed = pd.to_datetime(dt, dayfirst=True, errors='coerce')
        dt = parsed
    return pd.Timestamp(dt).strftime("%d/%m/%Y")


if __name__ == '__main__':
    # Adjust path if necessary
    # os.chdir("D:/Studium/Master/Masterarbeit/MAI_FinancialForecasting/Program")

    start = "01/01/2010"
    end = "31/12/2022"

    try:
        df_vix = scrape_all_vix(start, end, chunk_days=360, verbose=True, max_retries=5)
        print("VIX Data Sample:")
        print(df_vix.tail())
    except Exception as e:
        print(f"Scrape failed: {e}")
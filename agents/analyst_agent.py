from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta # Import pandas-ta
import time
import logging

from nifty_stocks import BENCHMARK_INDEX # Import benchmark index

log = logging.getLogger(__name__)

def _series_from_timeseries(rows: Optional[List[Dict]]) -> Optional[pd.DataFrame]:
    """Safely creates and prepares a DataFrame from timeseries data."""
    if not rows:
        return None
    try:
        df = pd.DataFrame(rows)
        if df.empty or 'date' not in df.columns or 'close' not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values("date").set_index('date') # Set date as index for ta
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close']) # Ensure close prices are valid numbers
        if df.empty:
            return None
        return df
    except Exception as e:
        log.warning(f"Error processing timeseries rows: {e}")
        return None

def analyze(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs quantitative analysis on the data bundle from the Data Agent.
    Calculates technical indicators (SMA, RSI) and fetches more fundamentals.
    
    """
    t0 = time.time()
    log.info("--- Analyst Agent START ---")
    market = bundle.get("market", {})
    ts = market.get("timeseries", {}) # Expects date-indexed DataFrame from DataAgent
    results = {"symbols": {}}

    primary_symbol = bundle.get("primary_symbol") # Get primary symbol identified by Data Agent
    symbols_to_analyze = list(ts.keys())
    log.info(f"Analyzing symbols: {symbols_to_analyze}, Primary: {primary_symbol}")

    # Process Benchmark Index First
    benchmark_df = None
    if BENCHMARK_INDEX in ts:
        benchmark_df = _series_from_timeseries(ts.get(BENCHMARK_INDEX))
        if benchmark_df is None:
            log.warning(f"Could not process benchmark index {BENCHMARK_INDEX} data.")

    # Analyze each symbol
    for sym in symbols_to_analyze:
        if sym == BENCHMARK_INDEX: # Skip benchmark for detailed analysis here
             continue

        rows = ts.get(sym, [])
        df = _series_from_timeseries(rows)

        if df is None or df.empty:
            log.warning(f"No valid timeseries data to analyze for {sym}")
            continue

        log.debug(f"Analyzing {sym} with {len(df)} data points.")
        metrics = {}

        # Basic Calcs
        metrics["latest_close"] = float(df["close"].iloc[-1])
        ret = df["close"].pct_change().dropna()

        # Returns & Volatility
        metrics["ret_10d"] = float((df["close"].iloc[-1] / df["close"].iloc[-11] - 1)) if len(df) >= 11 else None
        metrics["ret_20d"] = float((df["close"].iloc[-1] / df["close"].iloc[-21] - 1)) if len(df) >= 21 else None
        metrics["vol_10d"] = float(ret.tail(10).std() * np.sqrt(252)) if len(ret) >= 10 else None
        metrics["vol_20d"] = float(ret.tail(20).std() * np.sqrt(252)) if len(ret) >= 20 else None

        # Max Drawdown
        cummax = df["close"].cummax()
        drawdown = (df["close"] / cummax - 1.0)
        metrics["max_drawdown"] = float(drawdown.min()) if not drawdown.empty else None

        # --- Fetch More Fundamentals (with error handling) ---
        try:
            # Use yf.Ticker for info dictionary - potentially richer than download
            # Consider adding caching here if making many calls
            ticker_info = yf.Ticker(sym).info
            metrics["pe"] = ticker_info.get("trailingPE")
            metrics["pb"] = ticker_info.get("priceToBook")
            metrics["marketCap"] = ticker_info.get("marketCap")
            metrics["dividendYield"] = ticker_info.get("dividendYield")
            metrics["debtToEquity"] = ticker_info.get("debtToEquity")
            metrics["returnOnEquity"] = ticker_info.get("returnOnEquity")
            metrics["forwardPE"] = ticker_info.get("forwardPE")
            metrics["fiftyTwoWeekHigh"] = ticker_info.get("fiftyTwoWeekHigh")
            metrics["fiftyTwoWeekLow"] = ticker_info.get("fiftyTwoWeekLow")
            log.debug(f"Fetched fundamentals for {sym}")
            time.sleep(0.1) # Be respectful
        except Exception as e:
            log.warning(f"Could not fetch yfinance Ticker info for {sym}: {e}")
            # Initialize keys even if fetch fails
            metrics.update({
                "pe": None, "pb": None, "marketCap": None, "dividendYield": None,
                "debtToEquity": None, "returnOnEquity": None, "forwardPE": None,
                "fiftyTwoWeekHigh": None, "fiftyTwoWeekLow": None
            })
        # -----------------------------------------------

        # --- Calculate Technical Indicators using pandas-ta ---
        try:
            # Ensure enough data points for calculations
            if len(df) >= 50:
                 df.ta.sma(length=50, append=True)
                 metrics["sma_50d"] = df['SMA_50'].iloc[-1]
            else:
                 metrics["sma_50d"] = None

            if len(df) >= 200:
                 df.ta.sma(length=200, append=True)
                 metrics["sma_200d"] = df['SMA_200'].iloc[-1]
            else:
                 metrics["sma_200d"] = None

            if len(df) >= 15: # RSI needs n+1 periods typically
                 df.ta.rsi(length=14, append=True)
                 metrics["rsi_14d"] = df['RSI_14'].iloc[-1]
            else:
                 metrics["rsi_14d"] = None

            log.debug(f"Calculated technical indicators for {sym}")

        except Exception as e:
            log.warning(f"Could not calculate technical indicators for {sym}: {e}")
            metrics.update({"sma_50d": None, "sma_200d": None, "rsi_14d": None})
        # ----------------------------------------------------

        # --- Beta and Correlation vs Benchmark ---
        if benchmark_df is not None and not benchmark_df.empty:
            # Align data on common dates
            merged = pd.merge(
                df[["close"]].rename(columns={"close":"c_s"}),
                benchmark_df[["close"]].rename(columns={"close":"c_m"}),
                left_index=True, right_index=True, how="inner"
            )
            if len(merged) >= 21: # Need enough points for reliable beta/corr (e.g., 20 returns)
                sr = merged["c_s"].pct_change().dropna()
                mr = merged["c_m"].pct_change().dropna()
                # Ensure equal length after dropping NaNs
                common_index = sr.index.intersection(mr.index)
                sr, mr = sr.loc[common_index], mr.loc[common_index]

                if len(sr) >= 20:
                    try:
                        cov_matrix = np.cov(sr, mr)
                        cov = cov_matrix[0, 1]
                        varm = np.var(mr)
                        beta = float(cov / varm) if varm > 1e-10 else None # Avoid division by zero
                        corr = float(sr.corr(mr)) # Pandas corr handles NaNs potentially better
                        metrics["beta_vs_benchmark"] = beta
                        metrics["corr_vs_benchmark"] = corr
                        log.debug(f"Calculated Beta/Corr for {sym}")
                    except Exception as e:
                        log.warning(f"Could not calculate Beta/Corr for {sym}: {e}")
                        metrics["beta_vs_benchmark"] = None
                        metrics["corr_vs_benchmark"] = None
                else:
                    metrics["beta_vs_benchmark"] = None
                    metrics["corr_vs_benchmark"] = None
            else:
                 metrics["beta_vs_benchmark"] = None
                 metrics["corr_vs_benchmark"] = None
        else:
            metrics["beta_vs_benchmark"] = None
            metrics["corr_vs_benchmark"] = None
        # -----------------------------------------

        # Clean up NaN values for JSON serialization if needed, or keep None
        # metrics = {k: (None if pd.isna(v) else v) for k, v in metrics.items()}

        results["symbols"][sym] = metrics

    timing_ms = int((time.time() - t0) * 1000)
    log.info(f"--- Analyst Agent END --- (Total time: {timing_ms} ms)")
    return {
        "query": bundle.get("query", {}),
        "primary_symbol": primary_symbol, # Pass primary symbol along
        "analysis": results,
        "used_symbols": list(results["symbols"].keys()), # Symbols actually analyzed
        "benchmark_analyzed": BENCHMARK_INDEX if benchmark_df is not None else None,
        "diagnostics": {"timing_ms": timing_ms}
    }
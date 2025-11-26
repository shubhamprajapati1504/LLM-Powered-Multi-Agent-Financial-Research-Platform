from __future__ import annotations
import time, re, json, logging
from typing import List, Dict, Any
import pandas as pd
import yfinance as yf
import feedparser
from transformers import AutoTokenizer
from pathlib import Path
from fuzzywuzzy import process # For fuzzy matching

from config import VECTOR_FAISS_DIR, EMBEDDING_MODEL, TOP_K, TIMEZONE
from agents.index_loader import RetrieverIndex
from nifty_stocks import NAME_TO_TICKER, STOCK_NAMES, BENCHMARK_INDEX, TICKER_TO_NAME # Import stock list

log = logging.getLogger(__name__)

# --- Dynamic Ticker Resolution ---
def _resolve_query_symbols_fuzzy(q: str, entities: List[str] = [], threshold=80) -> List[str]:
    """
    Identifies potential NSE ticker symbols from a query using fuzzy matching
    against a predefined list (NIFTY 100/500).
    Uses entities extracted by the router agent if available.
    """
    hits = set()
    
    # Prioritize entities extracted by the router if available
    if entities:
        for entity in entities:
            entity_lower = entity.lower().replace(" ltd", "").replace(" limited", "").strip()
            # Direct ticker match first
            if entity.upper().endswith(".NS"):
                 hits.add(entity.upper())
                 continue
            # Check predefined mapping
            if entity_lower in NAME_TO_TICKER:
                hits.add(NAME_TO_TICKER[entity_lower])
                continue
            # Fuzzy match against known names
            match, score = process.extractOne(entity_lower, STOCK_NAMES)
            if score >= threshold and match in NAME_TO_TICKER:
                hits.add(NAME_TO_TICKER[match])
                log.info(f"Fuzzy matched entity '{entity}' to '{match}' (Score: {score}) -> {NAME_TO_TICKER[match]}")

    # Fallback: Search the entire query if no entities found or yielded results
    if not hits:
        query_lower = q.lower()
        # Look for explicit tickers first
        explicit_tickers = re.findall(r'([A-Z&-]+)\.NS', q)
        for tkr in explicit_tickers:
             hits.add(f"{tkr}.NS")
        
        # Then fuzzy match parts of the query against known names
        # (This is less precise, use cautiously or with higher threshold)
        # Potential optimization: only fuzzy match capitalized words or proper nouns
        words = re.findall(r'\b[A-Z][a-zA-Z-]+\b', q) # Simple heuristic for potential company names
        for word in words:
             word_lower = word.lower()
             if len(word_lower) > 3: # Avoid matching very short words
                 match, score = process.extractOne(word_lower, STOCK_NAMES)
                 if score >= (threshold + 5) and match in NAME_TO_TICKER: # Higher threshold for broad query matching
                     hits.add(NAME_TO_TICKER[match])
                     log.info(f"Fuzzy matched query word '{word}' to '{match}' (Score: {score}) -> {NAME_TO_TICKER[match]}")

    resolved = list(hits)
    log.info(f"Resolved tickers for query '{q}': {resolved}")
    return resolved
# -----------------------------


tok_cache = None
def _tok():
    global tok_cache
    if tok_cache is None:
        tok_cache = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return tok_cache

def _trim_text(txt: str, max_tokens: int) -> str:
    if not max_tokens or not txt: return txt or ""
    try:
        ids = _tok().encode(txt, add_special_tokens=False)
        return txt if len(ids) <= max_tokens else _tok().decode(ids[:max_tokens], skip_special_tokens=True)
    except Exception as e:
        log.warning(f"Token trimming failed: {e}. Falling back to character slice.")
        return txt[: max(1000, max_tokens * 4)] # Estimate character length

def _latest_summary(df: pd.DataFrame) -> dict:
    if df.empty: return {}
    s = df.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None)
    s = s.sort_values("date")
    last = s.iloc[-1]
    prev = s.iloc[-2] if len(s) > 1 else last
    closes = s["close"].astype(float).tail(10).pct_change().dropna()
    vol10 = float(closes.std() * (252 ** 0.5)) if len(closes) > 1 else None
    pct = float((last["close"] - prev["close"]) / prev["close"]) if prev["close"] and prev["close"] != 0 else 0.0
    return {"as_of": last["date"].isoformat(), "latest_close": float(last["close"]), "pct_change_1d": pct, "realized_vol_10d": vol10, "vendor": "yfinance"}

def _dedup_evidence(rows, per_domain_cap=2):
    seen_urls, per_domain = set(), {}
    deduped = []
    for e in rows:
        url = (e.get("url") or "").split("?")[0].rstrip("/").lower() # Normalize URL
        dom = (e.get("domain") or "").lower()
        if not url or url in seen_urls: continue
        # Simple domain capping
        current_domain_count = per_domain.get(dom, 0)
        if current_domain_count >= per_domain_cap: continue

        seen_urls.add(url)
        per_domain[dom] = current_domain_count + 1
        deduped.append(e)
    return deduped


class DataAgent:
    def __init__(self):
        try:
            self.index = RetrieverIndex(VECTOR_FAISS_DIR, EMBEDDING_MODEL)
            log.info(f"RetrieverIndex initialized successfully: {self.index.info()}")
        except Exception as e:
            log.error(f"Failed to initialize RetrieverIndex: {e}", exc_info=True)
            raise

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        try:
            return self.index.search(query, k=k)
        except Exception as e:
            log.error(f"Error during evidence retrieval: {e}", exc_info=True)
            return []

    def fetch_prices(self, symbols: List[str], period="3mo", interval="1d") -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        if not symbols: return out
        log.info(f"Fetching prices for symbols: {symbols}")
        # Add benchmark index if not present
        all_syms = list(set(symbols + [BENCHMARK_INDEX]))
        try:
            # Fetch all tickers in one go using yfinance download
            data = yf.download(all_syms, period=period, interval=interval, auto_adjust=False, progress=False)
            if data.empty:
                log.warning("yf.download returned empty DataFrame.")
                return out

            # Process each symbol
            for s in symbols: # Only process requested symbols, not necessarily the benchmark
                if s not in data['Close'] or data['Close'][s].isnull().all():
                     log.warning(f"No valid price data found for symbol: {s}")
                     continue

                # Select columns for the specific symbol, handling MultiIndex
                df_sym = data.loc[:, pd.IndexSlice[:, s]].copy()
                df_sym.columns = df_sym.columns.droplevel(1) # Drop the ticker level

                if df_sym.empty: continue
                df_sym = df_sym.rename_axis("date").reset_index()
                df_sym["date"] = pd.to_datetime(df_sym["date"]).dt.tz_localize(None) # Remove timezone
                df_sym = df_sym.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
                if "adj_close" not in df_sym.columns and "close" in df_sym.columns:
                    df_sym["adj_close"] = df_sym["close"] # Fallback if adj_close missing

                # Ensure required columns exist and handle potential NaNs
                required_cols = ["date","open","high","low","close","adj_close","volume"]
                df_sym = df_sym.dropna(subset=['close']) # Drop rows where close price is NaN
                if df_sym.empty: continue

                # Fill missing required columns with NaN or 0 if appropriate
                for col in required_cols:
                     if col not in df_sym:
                         df_sym[col] = 0 if col == 'volume' else np.nan

                out[s] = df_sym[required_cols].to_dict(orient="records")

            # Also store benchmark data if fetched successfully
            if BENCHMARK_INDEX in data['Close'] and not data['Close'][BENCHMARK_INDEX].isnull().all():
                 df_bench = data.loc[:, pd.IndexSlice[:, BENCHMARK_INDEX]].copy()
                 df_bench.columns = df_bench.columns.droplevel(1)
                 df_bench = df_bench.rename_axis("date").reset_index()
                 df_bench["date"] = pd.to_datetime(df_bench["date"]).dt.tz_localize(None)
                 df_bench = df_bench.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
                 if "adj_close" not in df_bench.columns and "close" in df_bench.columns:
                     df_bench["adj_close"] = df_bench["close"]
                 df_bench = df_bench.dropna(subset=['close'])
                 if not df_bench.empty:
                      required_cols_bench = ["date","open","high","low","close","adj_close","volume"]
                      for col in required_cols_bench:
                          if col not in df_bench:
                              df_bench[col] = 0 if col == 'volume' else np.nan
                      out[BENCHMARK_INDEX] = df_bench[required_cols_bench].to_dict(orient="records")

        except Exception as e:
            log.error(f"Error fetching prices with yf.download: {e}", exc_info=True)
            # Fallback to individual ticker fetching if download fails
            for s in symbols:
                try:
                    ticker = yf.Ticker(s)
                    df = ticker.history(period=period, interval=interval, auto_adjust=False)
                    if df.empty: continue
                    df = df.rename_axis("date").reset_index()
                    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
                    if "adj_close" not in df.columns and "close" in df.columns:
                        df["adj_close"] = df["close"]
                    df = df.dropna(subset=['close'])
                    if df.empty: continue
                    required_cols = ["date","open","high","low","close","adj_close","volume"]
                    for col in required_cols:
                         if col not in df:
                              df[col] = 0 if col == 'volume' else np.nan
                    out[s] = df[required_cols].to_dict(orient="records")
                    time.sleep(0.2) # Be respectful
                except Exception as ie:
                    log.warning(f"Could not fetch individual price history for {s}: {ie}")
        log.info(f"Finished fetching prices for {len(out)} symbols.")
        return out

    @staticmethod
    def _rss_url(q: str, hl="en-IN", gl="IN", ceid="IN:en") -> str:
        from urllib.parse import quote_plus
        return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl={hl}&gl={gl}&ceid={ceid}"

    def fetch_rss(self, queries: List[str], per_query_limit=15, sleep_s=0.25) -> List[Dict[str, Any]]:
        items, seen_links = [], set()
        log.info(f"Fetching RSS feeds for queries: {queries}")
        for q in queries:
            feed_url = self._rss_url(q)
            try:
                feed = feedparser.parse(feed_url)
                entries = feed.entries[:per_query_limit] if getattr(feed, "entries", None) else []
                count = 0
                for e in entries:
                    link = (e.get("link") or "").strip()
                    title = (e.get("title") or "").strip()
                    if not link or not title or link in seen_links: continue
                    seen_links.add(link)
                    items.append({
                        "title": title,
                        "link": link,
                        "published": (e.get("published") or e.get("updated") or "").strip(),
                        "source_feed": q # Store the query term instead of the long URL
                        #"query": q
                    })
                    count += 1
                log.info(f"Fetched {count} unique items for RSS query: '{q}'")
                time.sleep(sleep_s) # Be respectful to Google News
            except Exception as e:
                log.warning(f"Failed to fetch or parse RSS feed for query '{q}': {e}")
        log.info(f"Finished fetching RSS. Total unique items: {len(items)}")
        return items

    def run_pipeline(self, user_query: str, entities: List[str], default_tickers: List[str], rss_queries: List[str], k: int = TOP_K, limit_tokens_for_evidence: int = 256) -> Dict[str, Any]:
        """
        Main pipeline execution for the Data Agent.
        Uses entities from the router if available for ticker resolution.
        """
        t0 = time.time()
        log.info("--- Data Agent Pipeline START ---")

        # 1. Retrieve Evidence
        log.info(f"Retrieving top {k} evidence chunks for query: '{user_query}'")
        ev_raw = self.retrieve(user_query, k + 5) # Retrieve slightly more initially
        log.info(f"Retrieved {len(ev_raw)} raw evidence chunks.")
        ev_dedup = _dedup_evidence(ev_raw)[:k] # Deduplicate and cap to K
        log.info(f"Deduplicated evidence chunks: {len(ev_dedup)}")

        evidence = [{
            "id": int(e.get("id", i)), # Use index as fallback ID
            "external_id": f"{e.get('url','')}|{e.get('chunk',0)}",
            "url": e.get("url",""),
            "title": e.get("title",""),
            "published": e.get("published",""),
            "domain": e.get("domain",""),
            "score": float(e.get("score",0.0)),
            "chunk": int(e.get("chunk",0)),
            "text": _trim_text(e.get("text",""), limit_tokens_for_evidence)
        } for i, e in enumerate(ev_dedup)]

        # 2. Resolve Tickers
        resolved_tickers = _resolve_query_symbols_fuzzy(user_query, entities)
        # Combine resolved tickers with default tickers, ensuring uniqueness and adding benchmark
        tickers_to_fetch = list(dict.fromkeys(resolved_tickers + default_tickers + [BENCHMARK_INDEX]))

        # 3. Fetch Prices
        prices_raw = self.fetch_prices(tickers_to_fetch)
        market = {"symbols": {}, "timeseries": {}}
        successful_fetches = set()
        for sym, rows in prices_raw.items():
            df = pd.DataFrame(rows)
            if not df.empty:
                summary = _latest_summary(df)
                if summary: # Ensure summary calculation was successful
                    market["symbols"][sym] = summary
                    market["timeseries"][sym] = df.tail(60).to_dict(orient="records") # Keep timeseries for analysis
                    successful_fetches.add(sym)

        # Ensure benchmark timeseries is present if successfully fetched
        if BENCHMARK_INDEX in prices_raw and BENCHMARK_INDEX not in market["timeseries"]:
             df_bench = pd.DataFrame(prices_raw[BENCHMARK_INDEX])
             if not df_bench.empty:
                 market["timeseries"][BENCHMARK_INDEX] = df_bench.tail(60).to_dict(orient="records")


        # Determine primary symbol based ONLY on successfully fetched market data
        primary_symbol_candidates = [t for t in resolved_tickers if t in successful_fetches]
        primary_symbol = primary_symbol_candidates[0] if primary_symbol_candidates else None
        log.info(f"Primary symbol determined: {primary_symbol}")


        # 4. Fetch News Headlines (use resolved names/tickers for relevance)
        dynamic_rss_queries = list(dict.fromkeys(
             rss_queries + [TICKER_TO_NAME.get(t, t) for t in resolved_tickers] # Add names for resolved tickers
        ))
        headlines = self.fetch_rss(dynamic_rss_queries)

        # 5. Assemble Final Bundle
        timing_ms = int((time.time() - t0) * 1000)
        bundle = {
            "query": {"text": user_query, "timestamp": pd.Timestamp.now(tz=TIMEZONE).isoformat()},
            "primary_symbol": primary_symbol, # Add the identified primary symbol
            "evidence": evidence,
            "market": market,
            "news": {"rss": headlines, "source": "GoogleNewsRSS"},
            "diagnostics": {
                "resolved_tickers": resolved_tickers,
                "tickers_fetched": list(successful_fetches),
                "index_backend": self.index.backend,
                "vectors_in_index": len(getattr(self.index, "meta", [])),
                "timing_ms": timing_ms
            }
        }

        # 6. Persist Bundle (Optional but useful for debugging)
        try:
            runs = Path("runs"); runs.mkdir(exist_ok=True)
            slug = re.sub(r"[^a-z0-9]+", "-", user_query.lower()).strip("-")[:60] or "query"
            out_path = runs / f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{slug}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, ensure_ascii=False, indent=2, default=str)
            bundle["diagnostics"]["persisted_bundle"] = str(out_path)
            log.info(f"Data bundle persisted to {out_path}")
        except Exception as e:
            log.warning(f"Could not persist data bundle: {e}")

        log.info(f"--- Data Agent Pipeline END --- (Total time: {timing_ms} ms)")
        return bundle


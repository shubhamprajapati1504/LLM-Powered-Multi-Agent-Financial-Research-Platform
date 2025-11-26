from __future__ import annotations
from typing import Dict, Any, List, Optional, Literal
import logging
import time
import json

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

# Reuse Gemini LLM setup and config
from agents.thesis_agent import _make_llm
from config import GEMINI_MODEL_ID
# Need access to data/analysis functions potentially, or re-implement fetching
from agents.data_agent import DataAgent # To fetch data for multiple stocks
from agents.analyst_agent import analyze # To analyze multiple stocks
from nifty_stocks import NIFTY_100_STOCKS # Use a broader list of stocks

log = logging.getLogger(__name__)

# --- Pydantic Models for LLM Interaction ---

class PortfolioRequestDetails(BaseModel):
    """Details extracted from the user's portfolio request."""
    risk_profile: Literal["conservative", "moderate", "aggressive", "unknown"] = Field("unknown", description="Inferred risk profile (conservative, moderate, aggressive, or unknown).")
    investment_amount: Optional[float] = Field(None, description="Investment amount mentioned, if any.")
    time_horizon: Optional[str] = Field(None, description="Investment time horizon mentioned, if any (e.g., 'long-term', 'short-term').")
    specific_preferences: Optional[str] = Field(None, description="Any specific sectors or constraints mentioned (e.g., 'focus on IT', 'avoid Adani').")

class StockSuggestion(BaseModel):
    """Details of a suggested stock or sector."""
    name: str = Field(description="Stock ticker (e.g., RELIANCE.NS) or Sector name (e.g., 'Banking').")
    percentage: float = Field(description="Suggested allocation percentage (e.g., 20.0 for 20%).")
    rationale: str = Field(description="Brief reason for suggesting this stock/sector based on data and risk profile.")

class PortfolioSuggestionOutput(BaseModel):
    """Structured output for the portfolio suggestion."""
    headline: str = Field(description="A brief headline summarizing the suggestion type.")
    summary: str = Field(description="A short paragraph explaining the approach and mentioning the inferred user preferences.")
    sample_allocation: List[StockSuggestion] = Field(description="A list of suggested stocks/sectors and their allocation percentages.")
    key_considerations: List[str] = Field(description="Bullet points highlighting important factors or risks.")
    disclaimer: str = Field(description="Mandatory disclaimer stating this is illustrative, not financial advice.")


# --- Agent Logic ---

def _extract_request_details(query: str) -> PortfolioRequestDetails:
    """Uses LLM to extract key details from the user's portfolio request."""
    log.info("Extracting portfolio request details...")
    llm = _make_llm()
    structured_llm = llm.with_structured_output(PortfolioRequestDetails, method="json_mode")

    sys_prompt = "You are an expert at understanding user requests for investment portfolio suggestions. Extract the user's inferred risk profile, investment amount, time horizon, and any specific preferences mentioned. Default risk to 'unknown' if unclear. Respond strictly with the JSON schema."
    human_prompt = f"Analyze the following user request and extract the details:\n\nUser Request: \"{query}\""

    try:
        details = structured_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_prompt)
        ])
        log.info(f"Extracted details: {details}")
        return details
    except Exception as e:
        log.warning(f"Failed to extract portfolio details via LLM: {e}. Using defaults.")
        return PortfolioRequestDetails() # Return default values

def _filter_stocks(analyzed_data: Dict[str, Any], risk_profile: str) -> List[Dict[str, Any]]:
    """Simple filtering of stocks based on risk profile and basic metrics."""
    log.info(f"Filtering stocks based on risk profile: {risk_profile}")
    candidates = []
    symbols_data = analyzed_data.get("analysis", {}).get("symbols", {})

    for ticker, metrics in symbols_data.items():
        # Basic validation
        if not metrics or ticker == "^NSEI": # Exclude index
             continue

        # Example Filtering Logic (Needs refinement for real-world use)
        vol_20d = metrics.get("vol_20d")
        # --- FIX: Use correct key 'beta_vs_nifty' ---
        beta = metrics.get("beta_vs_nifty")
        # --- End Fix ---
        pe = metrics.get("pe")
        roe = metrics.get("returnOnEquity") # Key might still be missing or None

        passes = False
        if risk_profile == "conservative":
            # Prefer lower volatility, lower beta, positive PE (value indicator)
            if vol_20d is not None and vol_20d < 0.25 and \
               beta is not None and beta < 1.0 and \
               pe is not None and pe > 0:
                passes = True
        elif risk_profile == "aggressive":
             # Allow higher volatility/beta, consider higher ROE (growth indicator)
             if vol_20d is not None and vol_20d > 0.15 and \
                beta is not None and beta > 0.8 and \
                roe is not None and roe > 0.15: # ROE > 15%
                 passes = True
        else: # Moderate or Unknown - broader criteria
             if vol_20d is not None and vol_20d < 0.35 and \
                beta is not None and beta < 1.3 and \
                pe is not None: # Check if PE exists
                  passes = True

        if passes:
            # Add ticker and ensure returnOnEquity is included if available
            candidate_data = {"ticker": ticker, **metrics}
            # Explicitly add returnOnEquity if analyst agent provides it
            if 'returnOnEquity' not in candidate_data:
                 candidate_data['returnOnEquity'] = roe # Add it back if filtered based on it
            candidates.append(candidate_data)


    log.info(f"Found {len(candidates)} potential candidates after filtering.")
    # Limit candidates to avoid overwhelming the LLM prompt
    return candidates[:15] # Return top 15 candidates


def suggest_portfolio(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a *sample* portfolio suggestion based on inferred user needs and filtered stock data.
    
    """
    t0 = time.time()
    log.info("--- Portfolio Agent START ---")
    user_query = state.get("query", "N/A")
    # --- FIX: Get diagnostics correctly ---
    # Diagnostics might be nested from previous steps, ensure we get the latest
    diagnostics = state.get("data_bundle", {}).get("diagnostics", {})
    # --- End Fix ---


    # 1. Extract User Needs
    request_details = _extract_request_details(user_query)

    # 2. Gather & Analyze Data for a Broader Set of Stocks
    # Define a relevant subset of stocks (e.g., top 20-30 NIFTY stocks for demonstration)
    stock_subset_tickers = [ticker for name, ticker in NIFTY_100_STOCKS[:30]] # Example: Top 30
    log.info(f"Fetching and analyzing data for {len(stock_subset_tickers)} stocks...")
    try:
        # Re-use DataAgent's fetching capability (requires DataAgent instance or refactoring fetch_prices)
        # For simplicity here, let's assume we can call fetch and analyze
        # In LangGraph, this might be better done by calling the data/analyst nodes first
        # based on the intent, before reaching this node. This is a simplified approach.
        data_agent_instance = DataAgent() # Assuming it can be instantiated here
        temp_bundle = {"query": {"text": "internal analysis for portfolio"}} # Dummy query
        # --- FIX: Ensure benchmark index is included ---
        prices = data_agent_instance.fetch_prices(stock_subset_tickers + ["^NSEI"]) # Add NIFTY for Beta calc
        # --- End Fix ---
        temp_bundle["market"] = {"timeseries": prices}
        analyzed_data = analyze(temp_bundle) # Run analysis on the subset
    except Exception as e:
        log.error(f"Failed to fetch/analyze data for portfolio stocks: {e}", exc_info=True)
        timing_ms = int((time.time() - t0) * 1000)
        diagnostics["portfolio_agent_timing_ms"] = timing_ms
        return {
            "report": {
                 "query": {"text": user_query},
                 "analysis_type": "portfolio_suggestion",
                 "error": f"Failed to gather necessary stock data: {e}",
                 "disclaimer": "Could not generate suggestion due to data error. Not financial advice.",
                 "diagnostics": diagnostics
            }
        }


    # 3. Filter Stocks based on Risk
    filtered_candidates = _filter_stocks(analyzed_data, request_details.risk_profile)
    if not filtered_candidates:
        log.warning("No suitable stock candidates found after filtering.")
        timing_ms = int((time.time() - t0) * 1000)
        diagnostics["portfolio_agent_timing_ms"] = timing_ms
        return {
             "report": {
                 "query": {"text": user_query},
                 "analysis_type": "portfolio_suggestion",
                 "summary": "Could not identify suitable stocks based on the inferred risk profile and available data.",
                 "disclaimer": "Market conditions or filtering criteria may be too restrictive. This is not financial advice.",
                 "diagnostics": diagnostics
             }
        }

    # 4. Use LLM to Generate Sample Allocation
    log.info("Generating sample allocation using LLM...")
    llm = _make_llm()
    structured_llm = llm.with_structured_output(PortfolioSuggestionOutput, method="json_mode")

    # --- FIX: Prepare context for the LLM safely handling None ---
    candidates_summary = []
    for c in filtered_candidates:
        roe_val = c.get('returnOnEquity')
        roe_str = f"{roe_val*100:.1f}%" if roe_val is not None else "N/A"
        # --- FIX: Use correct key 'beta_vs_nifty' ---
        beta_val = c.get('beta_vs_nifty')
        # --- End Fix ---
        beta_str = f"{beta_val:.2f}" if beta_val is not None else "N/A"
        pe_val = c.get('pe')
        pe_str = f"{pe_val:.1f}" if pe_val is not None else "N/A"
        vol_val = c.get('vol_20d')
        vol_str = f"{vol_val:.2f}" if vol_val is not None else "N/A"

        candidates_summary.append(
            f"- {c['ticker']}: PE={pe_str}, Beta={beta_str}, Vol(20d)={vol_str}, ROE={roe_str}"
        )
    # --- End Fix ---


    sys_prompt = """
    You are an AI assistant helping to brainstorm *illustrative* portfolio allocations based on user preferences and filtered stock data.
    Your task is to create a *sample* allocation for educational purposes only.
    **CRITICAL: You MUST NOT provide financial advice.** Your language must be explicitly cautious and hypothetical.
    Use phrases like "A sample allocation might include...", "For illustration purposes...", "Based on the filtered data, one possible approach could be...".
    The total allocation must sum to 100%. You can suggest individual stocks or group them into sectors.
    Base your rationale *only* on the provided candidate stock summaries and the inferred user profile.
    Generate the response strictly following the JSON schema. Include a strong disclaimer.
    """.strip()

    human_prompt = f"""
    User Query: "{user_query}"
    Inferred User Profile: {request_details.json()}
    Filtered Stock Candidates (Ticker, PE, Beta, Volatility, ROE):
    {chr(10).join(candidates_summary)}

    Generate a *sample* portfolio allocation (summing to 100%) based on the inferred profile and the filtered candidates listed above.
    Focus on diversification. Provide a brief rationale for each suggestion based on the provided data.
    Respond strictly with the JSON schema, including the mandatory disclaimer.
    """.strip()

    try:
        suggestion = structured_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_prompt)
        ])
        log.info("LLM generated portfolio suggestion.")

        timing_ms = int((time.time() - t0) * 1000)
        diagnostics["portfolio_agent_timing_ms"] = timing_ms
        # Structure the final report
        return {
            # --- FIX: Pass state correctly, don't overwrite ---
            **state, # Include previous state keys
            # --- End Fix ---
            "report": {
                "query": {"text": user_query},
                "analysis_type": "portfolio_suggestion",
                "headline": suggestion.headline,
                "summary": suggestion.summary,
                "details": { # Nest the allocation details
                    "sample_allocation": [item.dict() for item in suggestion.sample_allocation],
                    "key_considerations": suggestion.key_considerations,
                },
                "disclaimer": suggestion.disclaimer, # Use the disclaimer from the LLM
                "diagnostics": diagnostics
            }
        }

    except Exception as e:
        log.error(f"Portfolio suggestion LLM call failed: {e}", exc_info=True)
        timing_ms = int((time.time() - t0) * 1000)
        diagnostics["portfolio_agent_timing_ms"] = timing_ms
        return {
            # --- FIX: Pass state correctly, don't overwrite ---
             **state, # Include previous state keys
            # --- End Fix ---
             "report": {
                 "query": {"text": user_query},
                 "analysis_type": "portfolio_suggestion",
                 "error": f"Failed generate suggestion via LLM: {e}",
                 "disclaimer": "Could not generate suggestion due to an internal error. Not financial advice.",
                 "diagnostics": diagnostics
             }
        }
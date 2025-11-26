from __future__ import annotations
import json, re, os, logging
from typing import Dict, Any, List, Optional

import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer
# --- NEW IMPORTS for Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
# -----------------------------
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from config import (
    GEMINI_MODEL_ID, # <-- Use Gemini model ID
    THESIS_MAX_NEW_TOKENS,
    THESIS_TEMPERATURE,
    THESIS_TOP_P,
    THESIS_K_EVIDENCE,
    THESIS_TOKENS_PER_PASSAGE,
    EMBEDDING_MODEL,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
load_dotenv()

# --- Pydantic model for structured output (unchanged) ---
class Citation(BaseModel):
    evidence_id: int = Field(description="The integer ID of the evidence passage being cited, e.g., 1.")
    reason: str = Field(description="A very short phrase explaining why this evidence is relevant.")
    url: Optional[str] = Field(None, description="The source URL of the evidence.")

class Verdict(BaseModel):
    leaning: str = Field(description="Must be one of 'bullish', 'bearish', or 'mixed'.")
    confidence: float = Field(description="Confidence from 0.0 (low) to 1.0 (high).")
    citations: List[Citation] = Field(description="A list of citations supporting the verdict.")

class ThesisOutput(BaseModel):
    bullish_thesis: str = Field(description="The bullish thesis, as numbered points, citing sources with (Domain, YY-MM-DD).")
    bearish_thesis: str = Field(description="The bearish thesis, as numbered points, citing sources with (Domain, YY-MM-DD).")
    verdict_scaffold: Verdict = Field(description="The final verdict scaffold.")

# --- Helper functions (unchanged) ---
def _primary_symbol(analysis: Dict[str, Any]) -> str | None:
    used = analysis.get("used_symbols") or []
    for s in used:
        if s and not s.startswith("^"): return s
    return used[0] if used else None

def _iso_date(s: str) -> str:
    try:
        return pd.to_datetime(s).date().isoformat()
    except Exception:
        return s or ""

def _trim_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text: return text or ""
    try:
        tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens: return text
        return tok.decode(ids[:max_tokens], skip_special_tokens=True)
    except Exception:
        return text[: max(1000, max_tokens * 4)]

def _select_evidence(bundle: Dict[str, Any], k: int, max_tokens_per_passage: int) -> List[Dict[str, Any]]:
    rows = bundle.get("evidence", []) or []
    rows = sorted(rows, key=lambda e: (float(e.get("score", 0.0)), _iso_date(e.get("published", ""))), reverse=True)
    seen_urls, per_dom, out = set(), {}, []
    for e in rows:
        url = (e.get("url") or "").split("?")[0].rstrip("/")
        dom = (e.get("domain") or "").lower()
        if not url or url in seen_urls or per_dom.get(dom, 0) >= 2: continue
        seen_urls.add(url); per_dom[dom] = per_dom.get(dom, 0) + 1
        ee = dict(e)
        ee["text"] = _trim_tokens(e.get("text","") or "", max_tokens_per_passage)
        out.append(ee)
        if len(out) >= k: break
    return out

def _metrics_table(analysis: Dict[str, Any], primary: str | None) -> Dict[str, Any]:
    symbols = (analysis.get("analysis") or {}).get("symbols") or {}
    if not primary or primary not in symbols:
        primary = list(symbols.keys())[0] if symbols else None
    return {"primary": primary, "metrics": symbols.get(primary, {})} if primary else {"primary": None, "metrics": {}}

def _build_messages(user_query: str, primary_symbol: str | None, metrics: Dict[str, Any], evid_list: List[Dict[str, Any]]) -> List:
    ev_lines = [f"[{i+1}] {e.get('title','').strip()} — {e.get('domain','')} — {_iso_date(e.get('published',''))}\n{e.get('text','') or ''}" for i, e in enumerate(evid_list)]
    sys_text = "You are an expert investment research analyst for the Indian stock market. Your task is to generate a balanced bull/bear thesis based *only* on the provided metrics and evidence. You must generate a JSON object that follows the specified schema. Do not output markdown code blocks."
    metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2, default=str)
    evid_block = "\n\n".join(ev_lines)
    user_text = f"USER QUESTION: {user_query}\nPRIMARY SYMBOL: {primary_symbol or 'unknown'}\n\nMETRICS (JSON for primary symbol):\n{metrics_json}\n\nEVIDENCE (top passages):\n{evid_block}\n\nBased on the data above, generate the investment thesis as a JSON object that strictly adheres to the schema."
    return [SystemMessage(content=sys_text), HumanMessage(content=user_text)]

# --- NEW: Function to create the Gemini LLM ---
def _make_llm(model_id: str = GEMINI_MODEL_ID):
    # The API key is automatically read from the GOOGLE_API_KEY environment variable
    return ChatGoogleGenerativeAI(
        model=model_id,
        temperature=THESIS_TEMPERATURE,
        top_p=THESIS_TOP_P,
        convert_system_message_to_human=True # Important for Gemini
    )

def generate_thesis(bundle: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    user_query = (bundle.get("query") or {}).get("text", "")
    primary = _primary_symbol(analysis)
    metrics = _metrics_table(analysis, primary)
    evid = _select_evidence(bundle, THESIS_K_EVIDENCE, THESIS_TOKENS_PER_PASSAGE)
    messages = _build_messages(user_query, primary, metrics, evid)

    try:
        log.info(f"Invoking Gemini ({GEMINI_MODEL_ID}) for structured thesis generation...")
        llm = _make_llm()
        # Use the native JSON mode for structured output
        structured_llm = llm.with_structured_output(ThesisOutput, method="json_mode")
        
        resp = structured_llm.invoke(messages)
        # print(resp)
        log.info("Structured Gemini invocation successful.")
        
        thesis_data = resp.dict()
        
        id_to_url = {i+1: e.get("url","") for i, e in enumerate(evid)}
        citations = thesis_data.get("verdict_scaffold", {}).get("citations", [])
        if citations:
            for c in citations:
                c["url"] = id_to_url.get(c.get("evidence_id"))
        
        return {
            "thesis_bull": thesis_data.get("bullish_thesis", ""),
            "thesis_bear": thesis_data.get("bearish_thesis", ""),
            "verdict_scaffold": thesis_data.get("verdict_scaffold", {}),
            "meta": { "repo_id": GEMINI_MODEL_ID, "used_primary_symbol": primary }
        }
        
    except Exception as e:
        log.error(f"CRITICAL: Thesis agent with Gemini failed. Error: {e}")
        return { "thesis_bull": "", "thesis_bear": "", "verdict_scaffold": {}, "error": f"LLM Structured Output Error: {str(e)}" }


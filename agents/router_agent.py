from __future__ import annotations
from typing import Dict, Any, Literal, List
import logging
import time

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from agents.thesis_agent import _make_llm # Reuse the same Gemini LLM

log = logging.getLogger(__name__)

# Define the possible intents
QueryIntent = Literal[
    "stock_analysis",        # Asking about a specific stock (buy/sell/outlook)
    "stock_comparison",      # Comparing two or more stocks
    "portfolio_request",     # Asking for portfolio suggestions or allocation
    "hypothetical_scenario", # Asking "what if" questions (e.g., rate hikes)
    "general_qa"             # General financial questions not covered above
]

class IntentClassification(BaseModel):
    """The classified intent of the user's query."""
    intent: QueryIntent = Field(description="The single most appropriate intent category for the user query.")
    entities: List[str] = Field(default_factory=list, description="List of company names or stock tickers explicitly mentioned (e.g., ['Infosys', 'TCS.NS', 'Reliance Industries']). Normalize names (e.g., 'Reliance' -> 'Reliance Industries').")
    confidence: float = Field(description="Confidence score (0.0 to 1.0) in the classification.")
    reasoning: str = Field(description="Brief explanation for the chosen intent.")

def classify_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classifies the user's query into a specific intent category using Gemini.
    
    """
    t0 = time.time()
    log.info("--- Router Agent START ---")
    user_query = state.get("query", "")
    if not user_query:
        log.warning("Router Agent: Empty query received. Defaulting to 'general_qa'.")
        timing_ms = int((time.time() - t0) * 1000)
        return {**state, "intent": "general_qa", "entities": [], "diagnostics": {"router_timing_ms": timing_ms}} # Default if query is empty

    sys_prompt = """
    You are an expert at understanding user queries related to finance and investments in the Indian market.
    Your task is to classify the user's query into ONE of the predefined intents and extract relevant entities (company names/tickers).
    Normalize extracted company names to their common full names (e.g., 'RIL' or 'Reliance' should become 'Reliance Industries').
    Provide your response strictly as a JSON object matching the requested schema.

    Intent Categories:
    - stock_analysis: Query focuses on a single stock (buy/sell/outlook/performance/news/data). Examples: "Should I buy INFY?", "What's the P/E of HDFC Bank?", "Latest news on Tata Motors."
    - stock_comparison: Query explicitly asks to compare two or more specific stocks. Examples: "Compare Infosys and TCS.", "Which is better, Reliance or Adani Enterprises?"
    - portfolio_request: Query asks for portfolio construction advice, asset allocation, or recommendations for multiple stocks to build a portfolio. Examples: "Suggest some stocks for a long-term portfolio.", "How should I allocate 50,000 rupees in stocks?"
    - hypothetical_scenario: Query asks 'what if' about market events or economic changes. Examples: "What happens to bank stocks if RBI raises rates?", "Impact of election results on NIFTY."
    - general_qa: General financial questions, market news summaries, definitions, or queries not fitting other categories. Examples: "What is RSI?", "Summarize today's market news.", "Who is the CEO of ICICI Bank?"
    """.strip()

    human_prompt = f"""
    User Query: "{user_query}"

    Classify the intent, extract and normalize entities based on the query. Respond strictly with the JSON schema provided.
    """.strip()

    try:
        log.info(f"Classifying intent for query: '{user_query}'")
        llm = _make_llm()
        structured_llm = llm.with_structured_output(IntentClassification, method="json_mode")
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_prompt)
        ]

        # Add timeout and retries maybe
        classification_result = structured_llm.invoke(messages, config={"max_retries": 1})

        log.info(f"Intent classified as: {classification_result.intent} (Confidence: {classification_result.confidence:.2f}) with entities: {classification_result.entities}. Reasoning: {classification_result.reasoning}")

        # Add classification to the state
        timing_ms = int((time.time() - t0) * 1000)
        return {
            **state,
            "intent": classification_result.intent,
            "entities": classification_result.entities, # Pass entities for Data Agent
            "diagnostics": {"router_timing_ms": timing_ms}
        }

    except Exception as e:
        log.error(f"Intent classification failed: {e}. Defaulting to 'general_qa'.", exc_info=True)
        # Fallback to a default intent in case of error
        timing_ms = int((time.time() - t0) * 1000)
        return {**state, "intent": "general_qa", "entities": [], "diagnostics": {"router_timing_ms": timing_ms}}
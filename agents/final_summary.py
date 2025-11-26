from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
import logging
import time

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
# Ensure _make_llm is correctly imported and configured for Gemini
try:
    from agents.thesis_agent import _make_llm
except ImportError:
    logging.error("Could not import _make_llm from agents.thesis_agent. Ensure the file and function exist.")
    raise

log = logging.getLogger(__name__)

# --- Pydantic Models for Different Response Types ---

class StandardSummaryResponse(BaseModel):
    """Conversational response for standard stock analysis."""
    headline: str = Field(description="A clear, one-sentence headline summarizing the stock's outlook based on the analysis.")
    summary: str = Field(description="A 3-4 sentence paragraph synthesizing the key findings (metrics, news, thesis, verification), written in a helpful, advisory tone.")
    key_points: List[str] = Field(description="A list of 3-5 bullet points covering the most important bullish, bearish factors, and verification notes.")
    next_steps: str = Field(description="A concluding sentence suggesting what the user might consider next (e.g., monitor earnings, check specific indicators), framed as educational advice.")
    disclaimer: str = Field(default="Remember, this AI analysis is for informational purposes only and not financial advice.", description="Standard disclaimer.")

class PortfolioSummaryResponse(BaseModel):
    """Conversational response for portfolio suggestions."""
    headline: str = Field(description="Headline summarizing the nature of the portfolio suggestion (e.g., 'Illustrative Portfolio Allocation').")
    summary: str = Field(description="Paragraph explaining the generated sample allocation, mentioning the inferred user profile and approach.")
    key_points: List[str] = Field(description="Bullet points describing the main components of the sample allocation or key considerations mentioned.")
    next_steps: str = Field(description="Suggestion for the user, likely emphasizing further research or consultation.")
    disclaimer: str = Field(description="Mandatory disclaimer emphasizing this is illustrative, educational, and not financial advice.")

class SimulationSummaryResponse(BaseModel):
     """Conversational response for hypothetical scenario analysis."""
     headline: str = Field(description="Headline summarizing the potential outcome or focus of the scenario analysis.")
     summary: str = Field(description="Paragraph describing the analyzed scenario and the main potential impacts discussed.")
     key_points: List[str] = Field(description="Bullet points highlighting the key potential impacts and major uncertainties/assumptions.")
     next_steps: str = Field(description="Suggestion for the user regarding the analysis (e.g., consider these factors, monitor related news).")
     disclaimer: str = Field(description="Mandatory disclaimer emphasizing the speculative nature, reliance on assumptions, and that it's not financial advice.")

class ErrorSummaryResponse(BaseModel):
     """Conversational response for errors during analysis."""
     headline: str = Field(description="Headline indicating an error occurred.")
     summary: str = Field(description="Explanation of the error encountered during the process.")
     key_points: List[str] = Field(default_factory=list, description="Usually empty, maybe suggest checking logs.")
     next_steps: str = Field(description="Suggestion to the user (e.g., rephrase query, check detailed output).")
     disclaimer: str = Field(description="Disclaimer stating this is an error message.")


# --- Main Function ---
def generate_final_summary(report: Dict[str, Any]) -> BaseModel: # Return type is now BaseModel
    """
    Takes the entire analysis report and generates a final, human-readable summary,
    adapting the output based on the analysis type reported by the pipeline.
    
    """
    t0 = time.time()
    log.info("--- Final Summary Agent START ---")

    # Determine analysis type and check for errors first
    analysis_type = report.get("analysis_type", "stock_analysis") # Default if missing
    user_query = report.get("query", {}).get("text", "N/A")
    error = report.get("error")
    diagnostics = report.get("diagnostics", {})

    # Handle explicit errors reported by the pipeline
    if error:
         log.warning(f"Pipeline reported an error: {error}. Generating error summary.")
         timing_ms = int((time.time() - t0) * 1000)
         diagnostics["final_summary_timing_ms"] = timing_ms
         report["diagnostics"] = diagnostics # Update report diagnostics
         return ErrorSummaryResponse(
             headline="Analysis Incomplete",
             summary=f"The analysis could not be completed due to an error: {error}",
             next_steps="Please try rephrasing your query or check the detailed agent output below.",
             disclaimer="This is an error message."
         )

    # Prepare context for the LLM
    # Reduce noise for the summary LLM - maybe exclude full evidence text?
    summary_context_report = report.copy()
    if 'evidence_topk' in summary_context_report:
        # Keep only essential evidence info for summary context
        summary_context_report['evidence_summary'] = [
             f"[{i+1}] {e.get('title','')} ({e.get('domain','')}, {e.get('published','')[:10]}) - Score: {e.get('score'):.2f}"
             for i, e in enumerate(summary_context_report.pop('evidence_topk', []))
        ]
    report_str = json.dumps(summary_context_report, indent=2, default=str)

    # Select the appropriate Pydantic model and prompt based on analysis type
    if analysis_type == "portfolio_suggestion":
        TargetModel = PortfolioSummaryResponse
        sys_prompt_core = """
        Your role is to synthesize a JSON report containing a *sample* portfolio suggestion into a conversational summary.
        Explain the sample allocation provided, referencing the inferred user profile mentioned in the report.
        Emphasize the illustrative nature and the critical disclaimer.
        Respond strictly with the 'PortfolioSummaryResponse' JSON schema.
        **CRITICAL: Reiterate this is NOT financial advice.**
        """
    elif analysis_type == "hypothetical_scenario":
        TargetModel = SimulationSummaryResponse
        sys_prompt_core = """
        Your role is to synthesize a JSON report containing a *qualitative* scenario analysis into a conversational summary.
        Explain the scenario analyzed and the potential impacts discussed in the report.
        Highlight the key assumptions and uncertainties mentioned.
        Emphasize the speculative nature and the critical disclaimer.
        Respond strictly with the 'SimulationSummaryResponse' JSON schema.
        **CRITICAL: Reiterate this is NOT a prediction or financial advice.**
        """
    else: # Default to standard stock analysis (stock_analysis, stock_comparison, general_qa fallback)
        TargetModel = StandardSummaryResponse
        sys_prompt_core = """
        Your role is to synthesize a JSON report from multiple AI agents (data, analysis, thesis, verification) about a stock into a conversational summary.
        Provide a balanced overview based on metrics, news (evidence), bull/bear arguments, and verification findings.
        If verification found issues, mention them subtly.
        Respond strictly with the 'StandardSummaryResponse' JSON schema.
        **CRITICAL: Do NOT give buy/sell advice.** Use cautious, educational language.
        """

    sys_prompt = f"""
    You are an AI financial assistant specializing in the Indian stock market.
    {sys_prompt_core}
    Base your summary *only* on the provided JSON report content. Do not add external information.
    Ensure the final output is a valid JSON object matching the requested schema.
    """.strip()

    human_prompt = f"""
    Here is the full analysis report generated by the AI agent team for my query "{user_query}":

    ```json
    {report_str}
    ```

    Please synthesize this report into a final, conversational response for me, adhering strictly to the '{TargetModel.__name__}' JSON schema and all instructions in the system prompt.
    """.strip()

    try:
        log.info(f"Generating final summary using {TargetModel.__name__} for query: '{user_query}'")
        llm = _make_llm()
        structured_llm = llm.with_structured_output(TargetModel, method="json_mode")
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_prompt)
        ]

        final_summary_obj = structured_llm.invoke(messages, config={"max_retries": 1})
        log.info("Final summary LLM call successful.")
        timing_ms = int((time.time() - t0) * 1000)
        diagnostics["final_summary_timing_ms"] = timing_ms
        report["diagnostics"] = diagnostics # Update report diagnostics
        return final_summary_obj

    except Exception as e:
        log.error(f"Final summary generation failed: {e}", exc_info=True)
        timing_ms = int((time.time() - t0) * 1000)
        diagnostics["final_summary_timing_ms"] = timing_ms
        report["diagnostics"] = diagnostics # Update report diagnostics
        # Fallback to a generic error response structure
        return ErrorSummaryResponse(
            headline="Summary Generation Failed",
            summary=f"An error occurred during the final synthesis step: {e}. Please refer to the detailed JSON output below.",
            next_steps="Try re-running the analysis or check application logs.",
            disclaimer="This is an error message."
        )
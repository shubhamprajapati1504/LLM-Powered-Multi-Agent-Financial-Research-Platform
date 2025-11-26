from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging
import time
import json

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

# Reuse Gemini LLM setup and config
from agents.thesis_agent import _make_llm
from config import GEMINI_MODEL_ID
# Need DataAgent to retrieve relevant context
from agents.data_agent import DataAgent, _trim_text
from config import THESIS_TOKENS_PER_PASSAGE # Reuse token limit

log = logging.getLogger(__name__)

# --- Pydantic Models for LLM Interaction ---

class ScenarioDetails(BaseModel):
    """Details extracted from the user's hypothetical scenario query."""
    scenario_description: str = Field(description="A clear, concise summary of the hypothetical event (e.g., 'RBI increases repo rate by 0.5%', 'Poor monsoon season impacting agriculture').")
    key_factors: List[str] = Field(description="Specific elements or variables involved in the scenario.")
    target_impact_area: Optional[str] = Field(None, description="The specific sector, index, or asset the user is asking about (e.g., 'banking stocks', 'NIFTY 50', 'inflation').")

class SimulationAnalysisOutput(BaseModel):
    """Structured output for the qualitative scenario analysis."""
    headline: str = Field(description="A brief headline summarizing the potential outcome or key theme.")
    scenario_summary: str = Field(description="Restatement of the scenario being analyzed.")
    potential_impacts: List[str] = Field(description="Bullet points describing potential qualitative impacts on the target area, based on evidence and reasoning.")
    key_assumptions_uncertainties: List[str] = Field(description="Bullet points highlighting assumptions made or key uncertainties affecting the outcome.")
    disclaimer: str = Field(description="Mandatory disclaimer stating this is a qualitative analysis based on available information and assumptions, not a prediction or financial advice.")

# --- Agent Logic ---

def _extract_scenario_details(query: str) -> ScenarioDetails:
    """Uses LLM to extract key details from the user's scenario query."""
    log.info("Extracting scenario details...")
    llm = _make_llm()
    structured_llm = llm.with_structured_output(ScenarioDetails, method="json_mode")

    sys_prompt = "You are an expert at understanding user queries about hypothetical financial or economic scenarios in the Indian context. Extract a clear description of the scenario, the key factors involved, and the specific area the user is asking about the impact on (e.g., a sector, index). Respond strictly with the JSON schema."
    human_prompt = f"Analyze the following user query and extract the scenario details:\n\nUser Query: \"{query}\""

    try:
        details = structured_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_prompt)
        ])
        log.info(f"Extracted scenario: {details}")
        return details
    except Exception as e:
        log.warning(f"Failed to extract scenario details via LLM: {e}. Cannot proceed.")
        raise ValueError(f"Could not understand the scenario from the query: {e}")


def run_simulation(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provides a *qualitative* analysis of a hypothetical scenario's potential impact,
    based on retrieved context and LLM reasoning.
    
    """
    t0 = time.time()
    log.info("--- Simulation Agent START ---")
    user_query = state.get("query", "N/A")
    diagnostics = state.get("diagnostics", {})

    # 1. Extract Scenario Details
    try:
        scenario_details = _extract_scenario_details(user_query)
    except ValueError as e:
        timing_ms = int((time.time() - t0) * 1000)
        diagnostics["simulation_agent_timing_ms"] = timing_ms
        return {
            "report": {
                 "query": {"text": user_query},
                 "analysis_type": "hypothetical_scenario",
                 "error": str(e),
                 "disclaimer": "Could not proceed due to ambiguity in the scenario description.",
                 "diagnostics": diagnostics
            }
        }

    # 2. Retrieve Relevant Context (News, Analysis about similar past events or opinions)
    log.info(f"Retrieving context for scenario: {scenario_details.scenario_description}")
    context_query = f"Impact of {scenario_details.scenario_description} on {scenario_details.target_impact_area or 'Indian economy'} analysis opinions"
    try:
        # Use the DataAgent instance potentially passed in state or instantiate
        data_agent_instance = DataAgent() # Assuming instantiation is feasible
        # Retrieve more evidence for context
        evidence_raw = data_agent_instance.retrieve(context_query, k=10) # Get top 10 relevant articles/chunks
        # Trim evidence for the prompt
        evidence_context = [
             f"[{i+1}] {e.get('title','')} ({e.get('domain','')}, {e.get('published','')[:10]}):\n{_trim_text(e.get('text',''), THESIS_TOKENS_PER_PASSAGE)}"
             for i, e in enumerate(evidence_raw)
        ]
        context_block = "\n\n".join(evidence_context)
        log.info(f"Retrieved {len(evidence_raw)} context passages.")
    except Exception as e:
        log.warning(f"Failed to retrieve context for simulation: {e}. Proceeding without it.")
        context_block = "No specific context retrieved."


    # 3. Use LLM to Analyze Scenario Qualitatively
    log.info("Generating qualitative analysis using LLM...")
    llm = _make_llm()
    structured_llm = llm.with_structured_output(SimulationAnalysisOutput, method="json_mode")

    sys_prompt = """
    You are an AI financial analyst providing *qualitative* insights into hypothetical scenarios for the Indian market.
    Your task is to analyze the potential impacts of the described scenario based *only* on the provided context (if any) and your general knowledge of financial principles.
    **CRITICAL: Do NOT make specific predictions or give financial advice.** Your language must be cautious and focus on potential effects, influencing factors, and uncertainties.
    Use phrases like "could potentially lead to," "might affect," "depends heavily on," "factors to consider include," "historically, similar events have shown...".
    Generate the response strictly following the JSON schema. Include a strong disclaimer about the speculative nature.
    """.strip()

    human_prompt = f"""
    User Query: "{user_query}"
    Extracted Scenario: {scenario_details.json()}

    Retrieved Context (Articles/Analysis related to the scenario or similar past events):
    ---CONTEXT START---
    {context_block}
    ---CONTEXT END---

    Based on the scenario described and the provided context (if available), provide a *qualitative* analysis of the potential impacts.
    Focus on the target area ({scenario_details.target_impact_area or 'the market/economy in general'}).
    Discuss potential effects, influencing factors, and key uncertainties/assumptions.
    Respond strictly with the JSON schema, including the mandatory disclaimer.
    """.strip()

    try:
        analysis_output = structured_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_prompt)
        ])
        log.info("LLM generated simulation analysis.")

        timing_ms = int((time.time() - t0) * 1000)
        diagnostics["simulation_agent_timing_ms"] = timing_ms
        # Structure the final report
        return {
            "report": {
                "query": {"text": user_query},
                "analysis_type": "hypothetical_scenario",
                "headline": analysis_output.headline,
                "summary": analysis_output.scenario_summary,
                "details": { # Nest the details
                    "potential_impacts": analysis_output.potential_impacts,
                    "key_assumptions_uncertainties": analysis_output.key_assumptions_uncertainties,
                },
                "disclaimer": analysis_output.disclaimer, # Use the disclaimer from the LLM
                "diagnostics": diagnostics
            }
        }

    except Exception as e:
        log.error(f"Simulation analysis LLM call failed: {e}", exc_info=True)
        timing_ms = int((time.time() - t0) * 1000)
        diagnostics["simulation_agent_timing_ms"] = timing_ms
        return {
            "report": {
                 "query": {"text": user_query},
                 "analysis_type": "hypothetical_scenario",
                 "error": f"Failed generate qualitative analysis via LLM: {e}",
                 "disclaimer": "Could not generate analysis due to an internal error. Not financial advice.",
                 "diagnostics": diagnostics
            }
        }
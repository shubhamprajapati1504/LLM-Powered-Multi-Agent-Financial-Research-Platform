# agents/verification_agent.py
from __future__ import annotations
from typing import Dict, Any, List
import json, re

from langchain_core.messages import SystemMessage, HumanMessage
from agents.thesis_agent import _make_llm # Reuse the same LLM helper

def _extract_numbers(text: str) -> List[float]:
    """Finds numbers, percentages, and currencies in text."""
    # Matches integers, floats, and those with % sign
    return [float(n.replace('%', '')) for n in re.findall(r'-?\d+\.?\d*%?', text)]

def _check_numerical_claims(thesis_text: str, metrics: Dict[str, Any]) -> List[str]:
    """Checks if numbers in the thesis are supported by the metrics."""
    issues = []
    claims = _extract_numbers(thesis_text)
    
    # Flatten the metrics for easier checking
    metric_values = []
    for symbol_data in metrics.get("symbols", {}).values():
        for value in symbol_data.values():
            if isinstance(value, (int, float)):
                metric_values.append(value)
                # Also check percentage values
                metric_values.append(value * 100) 

    for claim in claims:
        # Check if the claimed number is reasonably close to any metric
        is_supported = any(abs(claim - m) < 1.0 for m in metric_values if m is not None) # Use a tolerance
        if not is_supported:
            issues.append(f"Unsupported numerical claim: The number '{claim}' was not found in the provided financial metrics.")
            
    return issues


def verify(analysis: Dict[str, Any], thesis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifies the generated thesis against the analysis data.
    Acts as a self-check and consistency layer.
    """
    
    bull_thesis = thesis.get("thesis_bull", "")
    bear_thesis = thesis.get("thesis_bear", "")
    metrics = analysis.get("analysis", {})
    
    # --- Step 1: Rule-based numerical check ---
    numerical_issues = []
    numerical_issues.extend(_check_numerical_claims(bull_thesis, metrics))
    numerical_issues.extend(_check_numerical_claims(bear_thesis, metrics))

    # --- Step 2: LLM-based consistency check ---
    sys_prompt = """
    You are a meticulous financial compliance officer. Your task is to verify if an investment thesis is consistent
    with the provided numerical data. Do not check for factual correctness with the real world, only check for
    consistency between the 'METRICS' and the 'THESIS'.
    
    If the thesis makes a claim that is NOT supported by the metrics, you must flag it.
    If the thesis makes a claim that directly CONTRADICTS the metrics, you must flag it.
    
    Respond with a JSON object with two keys: "is_consistent" (boolean) and "reason" (a brief, one-sentence explanation).
    """.strip()

    human_prompt_template = """
    METRICS:
    {metrics_json}

    THESIS to verify:
    {thesis_text}
    
    Now, provide your verification assessment as a JSON object.
    """.strip()

    llm = _make_llm()
    llm_findings = []
    
    for name, thesis_text in [("Bull", bull_thesis), ("Bear", bear_thesis)]:
        if not thesis_text:
            continue
        
        metrics_json = json.dumps(metrics, indent=2)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_prompt_template.format(metrics_json=metrics_json, thesis_text=thesis_text))
        ]
        
        try:
            resp = llm.invoke(messages)
            content = getattr(resp, "content", str(resp))
            # Extract JSON from the response
            json_str = re.search(r'\{.*\}', content, re.DOTALL)
            if json_str:
                assessment = json.loads(json_str.group(0))
                if not assessment.get("is_consistent", True):
                    llm_findings.append(f"{name} Thesis Inconsistency: {assessment.get('reason', 'No reason provided.')}")
        except Exception as e:
            llm_findings.append(f"Error during LLM verification for {name} Thesis: {str(e)}")
            
    # --- Step 3: Compile the final report ---
    final_issues = numerical_issues + llm_findings
    
    return {
        "passed": len(final_issues) == 0,
        "issues_found": len(final_issues),
        "details": final_issues
    }
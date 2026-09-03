"""LLM-based natural language explanation module using Ollama (local LLM)."""

import json
from typing import Optional, Dict, Any
import warnings



def safe_import_requests():
    """Safely import requests library for Ollama API calls."""
    try:
        import requests
        return requests
    except ImportError:
        warnings.warn("requests library not found. LLM explanations will use template fallback.")
        return None


def generate_explanation_with_llm(
    order_features: Dict[str, Any],
    top_shap_contributors: list,
    risk_score: float,
    confidence: float,
    risk_band: str,
    threshold: float,
    use_llm: bool = True,
    ollama_url: str = "http://localhost:11434/api/generate"
) -> str:
    """
    Generate a natural language explanation using local Ollama LLM.
    
    Uses local Ollama LLM (qwen2.5:7b) if available and use_llm=True, 
    otherwise fallback to template.
    
    Args:
        order_features: dict of order feature values
        top_shap_contributors: list of dicts with 'Feature', 'SHAP_Value', 'Direction'
        risk_score: hybrid risk score (0-1)
        confidence: confidence level (0-1)
        risk_band: 'Low', 'Watch', or 'High'
        threshold: model's risk threshold
        use_llm: whether to attempt LLM call
        ollama_url: Ollama API endpoint
    
    Returns:
        Natural language explanation string
    """
    
    if use_llm:
        explanation = _try_ollama_explanation(
            order_features, top_shap_contributors, risk_score, confidence, risk_band, ollama_url
        )
        if explanation is not None:
            return explanation
    
    # Fallback to template
    return _template_explanation(
        order_features, top_shap_contributors, risk_score, confidence, risk_band
    )


def _try_ollama_explanation(
    order_features: Dict[str, Any],
    top_shap_contributors: list,
    risk_score: float,
    confidence: float,
    risk_band: str,
    ollama_url: str
) -> Optional[str]:
    """
    Attempt to generate explanation using local Ollama LLM.
    
    Returns None if Ollama is not running or call fails.
    """
    requests = safe_import_requests()
    if requests is None:
        return None
    
    try:
        # Build the prompt
        shap_summary = "\n".join([
            f"  - {c.get('Feature', 'Unknown')}: {c.get('SHAP_Value', 0):+.2f} ({c.get('Direction', '?')})"
            for c in top_shap_contributors[:5]
        ])
        
        features_summary = ", ".join([
            f"{k}={v}" for k, v in list(order_features.items())[:5]
        ])
        
        prompt = (
            f"Based on this order's risk assessment, generate a brief (1-2 sentence) "
            f"plain-English explanation of why it was flagged.\n\n"
            f"Order summary: {features_summary}\n"
            f"Risk score: {risk_score:.1%}\n"
            f"Risk band: {risk_band}\n"
            f"Confidence: {confidence:.0%}\n\n"
            f"Top factors contributing to this risk:\n{shap_summary}\n\n"
            f"Generate a concise, customer-friendly explanation (e.g., 'Flagged due to large discount combined "
            f"with a new customer account'). Do NOT mention probabilities or technical details. "
            f"Keep it under 50 words."
        )
        
        # Call Ollama API
        response = requests.post(
            ollama_url,
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            warnings.warn(f"Ollama API returned status {response.status_code}")
            return None
    
    except requests.exceptions.ConnectionError:
        warnings.warn(
            "Could not connect to Ollama at http://localhost:11434. "
            "Make sure Ollama is running: 'ollama serve' and model is loaded: 'ollama pull qwen2.5:7b'"
        )
        return None
    except Exception as e:
        warnings.warn(f"LLM explanation generation failed: {str(e)}")
        return None


def _template_explanation(
    order_features: Dict[str, Any],
    top_shap_contributors: list,
    risk_score: float,
    confidence: float,
    risk_band: str
) -> str:
    """
    Generate explanation using a deterministic template (fallback).
    """
    if not top_shap_contributors:
        return f"Flagged as {risk_band.lower()} risk (score: {risk_score:.1%}, confidence: {confidence:.0%})."
    
    # Extract top positive and negative contributors
    positive = [c for c in top_shap_contributors if c.get('SHAP_Value', 0) > 0]
    negative = [c for c in top_shap_contributors if c.get('SHAP_Value', 0) < 0]
    
    parts = []
    
    if positive:
        risk_factors = ", ".join([c.get('Feature', '?') for c in positive[:2]])
        parts.append(f"Flagged primarily due to {risk_factors}")
    
    if negative:
        mitigating = ", ".join([c.get('Feature', '?') for c in negative[:1]])
        parts.append(f"mitigated by {mitigating}")
    
    explanation = " — ".join(parts) + "."
    
    if confidence < 0.6:
        explanation += f" (Borderline case with {confidence:.0%} confidence.)"
    
    return explanation


def get_llm_explanation_disclaimer() -> str:
    """Return a disclaimer for LLM-generated explanations."""
    return (
        "**AI-generated summary:** This explanation is generated by a local LLM (Ollama qwen2.5:7b) "
        "and is for reference only. It is not a substitute for the underlying model output, SHAP analysis, "
        "or human judgment. Always verify order details independently."
    )


def get_ollama_status(ollama_url: str = "http://localhost:11434/api/tags") -> Dict[str, Any]:
    """
    Check if Ollama is running and what models are available.
    
    Returns dict with 'running' (bool) and 'models' (list) keys.
    """
    requests = safe_import_requests()
    if requests is None:
        return {"running": False, "models": [], "error": "requests library not available"}
    
    try:
        response = requests.get(ollama_url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [m.get('name', '?') for m in data.get('models', [])]
            return {"running": True, "models": models}
        else:
            return {"running": False, "models": [], "error": f"Status {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {
            "running": False, 
            "models": [], 
            "error": "Could not connect to Ollama (http://localhost:11434). Start with: ollama serve"
        }
    except Exception as e:
        return {"running": False, "models": [], "error": str(e)}

"""
Local LangChain-compatible LLM using transformers.pipeline.
Runs 100% offline after the first model download — no API calls needed.
Uses google/flan-t5-small (~300MB, CPU-friendly, Streamlit-free-tier-safe).
"""
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# Module-level pipeline cache — avoids Pydantic v2 PrivateAttr conflicts
_PIPELINE_CACHE: dict = {}


def _get_pipeline(model_id: str, max_new_tokens: int):
    """Load and cache the pipeline once; reuse on subsequent calls."""
    if model_id not in _PIPELINE_CACHE:
        from transformers import pipeline
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device = -1

        _PIPELINE_CACHE[model_id] = pipeline(
            "text2text-generation",
            model=model_id,
            device=device,
            max_new_tokens=max_new_tokens,
        )
    return _PIPELINE_CACHE[model_id]


class HFInferenceLLM(LLM):
    """Local HuggingFace pipeline LLM — runs offline after first download."""

    model_id: str = "google/flan-t5-small"
    max_new_tokens: int = 256
    hf_token: str = ""  # kept for interface compatibility but unused locally

    @property
    def _llm_type(self) -> str:
        return "hf_local_pipeline"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            pipe = _get_pipeline(self.model_id, self.max_new_tokens)
            # Truncate to avoid token overflow on small model
            truncated = prompt[:1500]
            result = pipe(truncated)
            if result and isinstance(result, list):
                return result[0].get("generated_text", "").strip()
            return str(result)
        except Exception as e:
            return f"[Local LLM Error]: {str(e)}"

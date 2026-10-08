"""
Local LangChain-compatible LLM using transformers.pipeline.
Runs 100% offline after the first model download — no API calls, no network needed.
Uses google/flan-t5-small (~300MB, CPU-friendly, Streamlit-free-tier-safe).
"""
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class HFInferenceLLM(LLM):
    """Local HuggingFace pipeline LLM — runs offline after first download."""

    model_id: str = "google/flan-t5-small"
    max_new_tokens: int = 256
    hf_token: str = ""  # kept for API compat but not used locally

    # Store pipeline as class variable to avoid re-loading on every call
    _pipeline: Any = None

    @property
    def _llm_type(self) -> str:
        return "hf_local_pipeline"

    def _get_pipeline(self):
        if HFInferenceLLM._pipeline is None:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            HFInferenceLLM._pipeline = pipeline(
                "text2text-generation",
                model=self.model_id,
                device=device,
                max_new_tokens=self.max_new_tokens,
            )
        return HFInferenceLLM._pipeline

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            pipe = self._get_pipeline()
            # Truncate prompt to avoid token overflow on tiny model
            truncated = prompt[:1500]
            result = pipe(truncated)
            if result and isinstance(result, list):
                return result[0].get("generated_text", "").strip()
            return str(result)
        except Exception as e:
            return f"[Local LLM Error]: {str(e)}"

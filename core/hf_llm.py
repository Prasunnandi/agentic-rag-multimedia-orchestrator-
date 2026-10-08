"""
Custom LangChain-compatible LLM that calls HuggingFace Inference API directly
via requests — bypassing the broken provider routing in huggingface_hub >= 0.27.
Works 100% on the free HF tier.
"""
import requests
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class HFInferenceLLM(LLM):
    """Direct HuggingFace Inference API LLM — no provider routing."""

    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    hf_token: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "hf_inference_direct"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "return_full_text": False,
            },
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            # HF returns a list of dicts with "generated_text"
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            elif isinstance(result, dict):
                # Some models return {"error": "..."} if loading
                if "error" in result:
                    return f"[Model loading, please try again in 20s]: {result['error']}"
                return str(result)
            return str(result)

        except requests.exceptions.RequestException as e:
            return f"[API Error]: {str(e)}"

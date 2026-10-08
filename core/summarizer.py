from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class Summarizer:
    def __init__(self, llm):
        self.llm = llm

    def map_reduce_summarize(self, text_chunks):
        """Summarize each chunk then combine — works with small local models."""
        summaries = []
        prompt = PromptTemplate.from_template(
            "Summarize the following text in 2-3 sentences:\n\n{text}"
        )
        chain = prompt | self.llm | StrOutputParser()

        for chunk in text_chunks:
            try:
                s = chain.invoke({"text": chunk[:1200]})
                summaries.append(s.strip())
            except Exception:
                summaries.append(chunk[:200] + "...")

        # Combine all summaries
        combined = " ".join(summaries)
        if len(summaries) > 1:
            try:
                final = chain.invoke({"text": combined[:1200]})
                return final.strip()
            except Exception:
                return combined
        return combined

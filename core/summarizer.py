from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document

class Summarizer:
    def __init__(self, llm):
        self.llm = llm

    def map_reduce_summarize(self, text_chunks):
        docs = [Document(page_content=t) for t in text_chunks]
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        result = chain.invoke(docs)
        return result["output_text"]

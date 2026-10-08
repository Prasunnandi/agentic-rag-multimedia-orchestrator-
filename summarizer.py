from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

def generate_report(transcript, research_findings):
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    combined_text = f"--- TRANSCRIPT ---\n{transcript}\n\n--- RESEARCH FINDINGS ---\n{research_findings}"
    texts = text_splitter.split_text(combined_text)
    docs = [Document(page_content=t) for t in texts]
    
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke(docs)
    return summary["output_text"]

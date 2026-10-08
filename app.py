from core.agents import build_search_agent, build_reader_agent, build_writer_chain
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2

from utils.audio_processor import download_and_process_audio
from core.transcriber import Transcriber
from core.vector_store import VectorStore
from core.summarizer import Summarizer
from core.extractor import Extractor

load_dotenv()

st.set_page_config(page_title="AI Video Assistant & RAG", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; background-color: #0a0a0f !important; color: #e8e8f0 !important; }
.stApp { background: #0a0a0f !important; }
.stApp::before {
    content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-image: linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px; pointer-events: none; z-index: 0;
}
[data-testid="stSidebar"] { background: #111118 !important; border-right: 1px solid #2a2a3a !important; }
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
h1, h2, h3, h4, h5, h6 { font-family: 'Syne', sans-serif !important; color: #e8e8f0 !important; }
.stButton>button { background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🎥 AI Video Assistant & Document RAG")
st.markdown("Powered by Hugging Face API (Mistral), Whisper (Tiny), and LangChain")

# Initialize LLM
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("Please set HF_TOKEN in your .env file")
    st.stop()

@st.cache_resource
def get_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512,
        temperature=0.3
    )

llm = get_llm()

# UI Sidebar
st.sidebar.header("Data Ingestion")
data_source = st.sidebar.radio("Source Type", ["YouTube URL", "Document Upload (PDF/TXT)"])

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if data_source == "YouTube URL":
    yt_url = st.sidebar.text_input("YouTube Link")
    if st.sidebar.button("Process Video"):
        with st.spinner("Downloading and processing audio (16kHz Mono)..."):
            audio_path = download_and_process_audio(yt_url)
            if audio_path:
                with st.spinner("Transcribing with Whisper..."):
                    transcriber = Transcriber(model_size="tiny")
                    st.session_state.transcript = transcriber.transcribe_whisper(audio_path)
                st.success("Transcription Complete!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload File", type=['pdf', 'txt'])
    if st.sidebar.button("Process Document") and uploaded_file:
        with st.spinner("Extracting text..."):
            if uploaded_file.name.endswith('.pdf'):
                reader = PyPDF2.PdfReader(uploaded_file)
                text = "".join(page.extract_text() for page in reader.pages)
            else:
                text = uploaded_file.read().decode("utf-8")
            st.session_state.transcript = text
        st.success("Extraction Complete!")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Transcript & Summary", "🎯 Insights (Actions/Decisions)", "💬 Chat with Data", "🌐 Deep Web Research"])

if st.session_state.transcript:
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(st.session_state.transcript)
    
    # Initialize Vector Store if not done
    if st.session_state.retriever is None:
        with st.spinner("Building Vector Database..."):
            vs = VectorStore()
            metadatas = [{"chunk": i} for i in range(len(chunks))]
            st.session_state.retriever = vs.create_collection(chunks, metadatas)
    
    with tab4:
        st.subheader("Multi-Agent Web Research")
        st.markdown("Uses the **ReAct Agent Architecture** (Search Agent -> Reader Agent -> Writer Chain) based on the textbook.")
        research_query = st.text_input("Enter a research topic related to the document:")
        if st.button("Start Agents") and research_query:
            with st.spinner("Search Agent is looking for URLs..."):
                search_agent = build_search_agent(llm)
                try:
                    search_results = search_agent.invoke({"input": f"Find URLs relevant to: {research_query}"})
                    urls_found = search_results.get("output", search_results)
                except Exception as e:
                    urls_found = str(e)
                st.info(f"**Search Agent Output:**\n{urls_found}")
                
            with st.spinner("Reader Agent is scraping URLs..."):
                reader_agent = build_reader_agent(llm)
                try:
                    reader_results = reader_agent.invoke({"input": f"Scrape the following URLs and extract key facts about {research_query}:\n{urls_found}"})
                    facts = reader_results.get("output", reader_results)
                except Exception as e:
                    facts = str(e)
                st.info(f"**Reader Agent Output:**\n{facts}")
                
            with st.spinner("Writer Chain is writing the final report..."):
                writer_chain = build_writer_chain(llm)
                try:
                    final_report = writer_chain.invoke({"facts": facts})
                except Exception as e:
                    final_report = str(e)
                st.success("**Final Research Report (Writer Agent):**")
                st.markdown(final_report)

    with tab1:
        st.subheader("Raw Text")
        with st.expander("View Full Transcript/Text"):
            st.write(st.session_state.transcript)
            
        if st.button("Generate Map-Reduce Summary"):
            with st.spinner("Summarizing..."):
                summarizer = Summarizer(llm)
                st.write(summarizer.map_reduce_summarize(chunks[:5])) # Limit to 5 chunks for API speed on HF free tier
                
    with tab2:
        st.subheader("Automated Extraction")
        if st.button("Extract Action Items & Decisions"):
            with st.spinner("Analyzing..."):
                extractor = Extractor(llm)
                colA, colB = st.columns(2)
                # Limit text size to prevent exceeding API context limits
                safe_text = st.session_state.transcript[:3000] 
                with colA:
                    st.markdown("### Action Items")
                    st.write(extractor.extract_action_items(safe_text))
                with colB:
                    st.markdown("### Key Decisions")
                    st.write(extractor.extract_decisions(safe_text))
                    
    with tab3:
        st.subheader("RAG Q&A")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input("Ask a question about the video/document..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.spinner("Thinking..."):
                system_prompt = (
                    "You are a helpful assistant. Answer the user's question based ONLY on the provided context.\n"
                    "Context: {context}\n"
                )
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])
                
                qa_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(st.session_state.retriever, qa_chain)
                
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)

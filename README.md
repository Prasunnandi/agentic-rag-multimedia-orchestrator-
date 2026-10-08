# 🎥 Agentic RAG Multimedia Orchestrator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-ready AI application combining Retrieval-Augmented Generation, local LLMs, YouTube transcription, and a multi-agent research pipeline — all 100% free to deploy.**

[🚀 Live Demo](#) · [📖 Documentation](#architecture) · [🐛 Report Bug](https://github.com/Prasunnandi/agentic-rag-multimedia-orchestrator-/issues)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎬 **YouTube Video Analysis** | Paste any YouTube URL — audio is extracted, downsampled to 16kHz mono, and transcribed using OpenAI Whisper Tiny |
| 📄 **Document Intelligence** | Upload PDF or TXT files and instantly build a searchable knowledge base |
| 🧠 **RAG Chat** | Ask questions about your video/document using a vector-powered retrieval chain backed by `all-MiniLM-L6-v2` embeddings |
| 📝 **Map-Reduce Summarization** | LangChain's Map-Reduce chain condenses long documents into concise summaries |
| 🔍 **Action & Decision Extraction** | Automatically extract action items and key decisions from meeting transcripts |
| 🤖 **Multi-Agent Web Research** | A ReAct-architecture pipeline (Search Agent → Reader Agent → Writer Chain) using Tavily Search and BeautifulSoup |
| 🌐 **100% Free** | Runs on a local `distilgpt2` LLM — no OpenAI key, no paid API, no credit card |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                     │
│         Dark Theme · Tabs · Sidebar · Chat Interface        │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────────┐  ┌───────────────────┐
│ Audio/Video  │   │  Document (PDF/  │  │  Multi-Agent Web  │
│  Pipeline    │   │     TXT) Path    │  │  Research Pipeline│
│              │   │                  │  │                   │
│ yt-dlp       │   │ PyPDF2 / plaintext│  │ Search Agent      │
│ pydub (16kHz)│   │ Text Extraction  │  │ (Tavily Search)   │
│ Whisper Tiny │   │                  │  │ Reader Agent      │
└──────┬───────┘   └────────┬─────────┘  │ (BeautifulSoup)   │
       │                    │            │ Writer Chain      │
       └────────────────────┘            └───────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │    Vector Store       │
        │  ChromaDB + FAISS     │
        │ all-MiniLM-L6-v2      │
        │    Embeddings         │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Local LLM Engine   │
        │  distilgpt2 (82MB)   │
        │  Runs 100% offline   │
        └───────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- `ffmpeg` installed on your system (for audio extraction)
- A free [Hugging Face](https://huggingface.co) account
- A free [Tavily](https://tavily.com) API key (for web research)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Prasunnandi/agentic-rag-multimedia-orchestrator-.git
cd agentic-rag-multimedia-orchestrator-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your keys:
# HF_TOKEN=hf_your_token_here
# TAVILY_API_KEY=tvly_your_key_here

# 4. Run the app
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> ⚡ **First run:** The app will automatically download `distilgpt2` (~82MB) and `all-MiniLM-L6-v2` (~90MB) locally. Subsequent runs start instantly.

---

## 📦 Project Structure

```
agentic-rag-multimedia-orchestrator/
│
├── app.py                  # Main Streamlit application & UI
│
├── core/
│   ├── hf_llm.py           # Custom local LLM (distilgpt2, no API needed)
│   ├── agents.py           # ReAct multi-agent pipeline (Search/Reader/Writer)
│   ├── extractor.py        # Action item & decision extraction chains
│   ├── summarizer.py       # Map-Reduce summarization
│   ├── transcriber.py      # Whisper Tiny speech-to-text
│   ├── tools.py            # Tavily search & BeautifulSoup scraper tools
│   └── vector_store.py     # ChromaDB vector store with MiniLM embeddings
│
├── utils/
│   └── audio_processor.py  # yt-dlp download + pydub 16kHz conversion
│
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages for Streamlit Cloud (ffmpeg)
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit 1.35+ with custom CSS (Syne + JetBrains Mono) |
| **LLM** | `distilgpt2` via Hugging Face Transformers (local) |
| **Embeddings** | `all-MiniLM-L6-v2` via Sentence Transformers |
| **Vector Store** | ChromaDB / FAISS |
| **Orchestration** | LangChain 0.2+ (LCEL chains, ReAct agents) |
| **Transcription** | OpenAI Whisper Tiny (local) |
| **Video/Audio** | yt-dlp + pydub + ffmpeg |
| **Web Research** | Tavily Search API + BeautifulSoup4 |

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your fork → set main file to `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   HF_TOKEN = "hf_your_token_here"
   TAVILY_API_KEY = "tvly_your_key_here"
   ```
5. Click **Deploy** — Streamlit Cloud will auto-install `ffmpeg` via `packages.txt`

---

## 📸 Screenshots

> Upload a YouTube link or PDF, get instant AI-powered insights.

| Tab | Feature |
|---|---|
| 📄 Transcript & Summary | View raw text + generate a Map-Reduce summary |
| 🔍 Insights | Extract action items & key decisions side by side |
| 💬 RAG Chat | Conversational Q&A grounded in your document |
| 🌐 Web Research | Multi-agent pipeline generates a research report |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](https://github.com/Prasunnandi/agentic-rag-multimedia-orchestrator-/issues).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
Made with ❤️ using Streamlit, LangChain & Hugging Face
</div>

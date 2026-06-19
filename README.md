# 🧠 NeuroMind AI - Agent for Neuroscience

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58+-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.3.13-green?logo=chainlink)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3-orange?logo=thunderbird)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple)
![Pyodide](https://img.shields.io/badge/Pyodide-Browser%20Python-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

**An autonomous AI agent system for neuroscience research — extracting insights from papers, simulating neurons in the browser, and generating analysis code.**

[Live Demo 🚀](https://huggingface.co/spaces/Huzaif-Enan/neuromind-ai) • [Report Bug 🐛](https://github.com/MdHuzaif/neuromind-ai/issues) • [Request Feature ✨](https://github.com/MdHuzaif/neuromind-ai/issues)

</div>

---

## 🌟 Overview

**NeuroMind AI** is a specialized AI agent system designed for neuroscience researchers. Unlike generic chatbots, it understands **brain regions, p-values, methodologies, and experimental conditions** — extracting them with page-level citations from research papers.

### ✨ What Makes It Unique?

| Feature | Generic RAG Chatbot | NeuroMind AI |
|---------|---------------------|--------------|
| **Domain Knowledge** | General purpose | Neuroscience-specialized |
| **Citations** | Document-level | Page + Section-level |
| **Data Extraction** | Text only | p-values, n, brain regions, methods |
| **Code Execution** | None | Browser-based (Pyodide) |
| **Neuron Simulation** | None | LIF, Hodgkin-Huxley models |
| **Export** | Chat history | CSV, PDF reports, Python code |

---

## 🎯 Key Features

### 📚 Research Paper Analysis
- **Neuro-Specific Extraction**: Automatically identifies brain regions (hippocampus, cortex, etc.), statistical values (p-values, effect sizes), sample sizes, and methodologies
- **Page-Level Citations**: Every claim is backed by exact page and section references
- **Multi-PDF Support**: Process multiple papers simultaneously
- **Smart Caching**: FAISS vector index for instant subsequent queries

### 🧠 Neuron Simulation (Post-MVP)
- **Browser-Based Execution**: Run Python simulations directly in the browser using Pyodide — no installation needed
- **Pre-Built Models**: Leaky Integrate-and-Fire (LIF) and Hodgkin-Huxley neuron models
- **Custom Code Generation**: AI generates neuroscience-specific Python code based on your prompts
- **Publication-Quality Plots**: Matplotlib visualizations rendered in-browser

### 🔬 Code Generation
- **AI-Powered**: Groq LLM (Llama 3.3 70B) generates executable Python code
- **Context-Aware**: Uses uploaded papers to inform code generation
- **Quick Templates**: One-click access to common analyses (t-tests, ANOVA, correlation)

### 📊 Data Export & Privacy
- **CSV Export**: Download chat history as structured data
- **PDF Reports**: Generate professional research reports
- **Privacy Controls**: One-click session wipe for HIPAA/GDPR compliance
- **Auto-Delete**: All data cleared on session end

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.58+ | Interactive web interface |
| **LLM** | Groq (Llama 3.3 70B) | Lightning-fast inference |
| **Orchestration** | LangChain 0.3.13 | RAG pipeline management |
| **Vector Store** | FAISS | High-performance similarity search |
| **Embeddings** | Sentence Transformers (all-mpnet-base-v2) | Semantic embeddings |
| **PDF Parsing** | PyMuPDF + Unstructured | Text extraction with OCR fallback |
| **Code Execution** | Pyodide (WebAssembly) | Browser-based Python runtime |
| **Language** | Python 3.9+ | Core application logic |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Groq API Key (free at [console.groq.com](https://console.groq.com/))

### Installation

```bash
# Clone the repository
git clone https://github.com/MdHuzaif/neuromind-ai.git
cd neuromind-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

Configuration
Create a .streamlit/secrets.toml file:
GROQ_API_KEY = "gsk_your_groq_api_key_here"
Run the Application
streamlit run app.py
The app will open at http://localhost:8501

📖 Usage Guide
1. Upload Research Papers

    Click the file uploader
    Select one or more neuroscience PDF files
    Wait 20-30 seconds for processing

2. Ask Questions
Try these example queries:
✅ "What brain regions were studied and what were the p-values?"
✅ "Extract the sample sizes and statistical methods used"
✅ "What experimental conditions were tested?"
✅ "Summarize the methodology section"

3. Simulate Neurons

    Navigate to the "🧠 Neuron Simulator" tab
    Select a pre-built model (LIF or Hodgkin-Huxley)
    Click "Run Simulation in Browser" to see results instantly

4. Generate Custom Code

    Describe your analysis needs in the Code Generator tab
    AI generates executable Python code
    Download or run directly in the browser

5. Export Results

    Download chat history as CSV or PDF
    Export generated code as .py files
🏗️ Architecture
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (Streamlit Web App)                         │
└────────────────┬────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬──────────────┐
    │            │            │              │
    ▼            ▼            ▼              ▼
┌────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│  Chat  │  │ Neuron  │  │  Code    │  │  Export  │
│  RAG   │  │Simulator│  │Generator │  │  System  │
└───┬────┘  └────┬────┘  └────┬─────┘  └────┬─────┘
    │            │            │              │
    └────────────┼────────────┼──────────────┘
                 │            │
                 ▼            ▼
         ┌──────────────┐  ┌──────────┐
         │   LangChain  │  │  Pyodide │
         │   Pipeline   │  │  (WASM)  │
         └──────┬───────┘  └──────────┘
                │
        ┌───────┼───────┐
        │       │       │
        ▼       ▼       ▼
    ┌──────┐ ┌─────┐ ┌──────┐
    │Groq  │ │FAISS│ │PyMuPDF│
    │ LLM  │ │ DB  │ │Parser │
    └──────┘ └─────┘ └──────┘

🗺️ Roadmap
Phase 1: Core Features ✅ (Complete)

    Neuro-specific RAG chatbot
    Page-level citations
    CSV/PDF export
    Privacy controls

Phase 2: Neuron Simulation ✅ (Complete)

    Pyodide browser-based execution
    LIF and Hodgkin-Huxley models
    Custom code generation

Phase 3: Multi-Agent Literature Mining (In Progress)

    Agent 1: Paper Discovery (PubMed/arXiv API)
    Agent 2: Data Extraction (structured JSON)
    Agent 3: Meta-Analysis (Forest plots, effect sizes)
    Agent 4: Quality Assessment (study rigor evaluation)
    Agent 5: Report Generation (academic paper drafts)

Phase 4: Advanced Features (Planned)

    Multi-modal analysis (fMRI/EEG image interpretation)
    Automated statistical pipelines
    Citation validation and fact-checking
    LaTeX/PDF paper generation

🔐 Security & Privacy

    ✅ API keys stored securely in .streamlit/secrets.toml (gitignored)
    ✅ No data persistence beyond session
    ✅ FAISS index stored locally
    ✅ One-click session wipe for HIPAA/GDPR compliance
    ✅ All uploaded PDFs processed in-memory only

    ⚠️ Note: For sensitive research data, deploy on private infrastructure.

🤝 Contributing
Contributions are welcome! Here's how:

    Fork the repository
    Create your feature branch (git checkout -b feature/AmazingFeature)
    Commit your changes (git commit -m 'Add AmazingFeature')
    Push to the branch (git push origin feature/AmazingFeature)
    Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE
 file for details.
👨‍💻 Author
Md. Huzaif Ibnul Amin
Software Engineer | Competitive Programmer | Neuroscience Enthusiast

    🎓 B.Sc. in Computer Science & Engineering, Khulna University
    💼 Junior Software Developer (Laravel ERP)
    🏆 HULT Prize Winner 2019, KU CSE FEST Hackathon Winner
    💻 500+ competitive programming problems solved (Codeforces, LeetCode, UVA)

Connect with Me

    📧 Email: enanamin002429@gmail.com
    💼 LinkedIn: linkedin.com/in/huzaif-ibnul-amin
    🐙 GitHub: github.com/MdHuzaif

🙏 Acknowledgments

    LangChain
     - RAG orchestration framework
    Groq
     - Ultra-fast LLM inference
    FAISS
     - Vector search by Meta AI
    Streamlit
     - Rapid web app development
    Pyodide
     - Python in the browser
    HuggingFace
     - Model hosting and embeddings

<div align="center">

Built with ❤️ for the Neuroscience Research Community
Empowering researchers with AI-driven insights
⭐ Star this repo if you find it useful!
</div>
```

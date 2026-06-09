import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="NeuroMind AI",
    page_icon="🧠",
    layout="wide"
)

# Header Section
st.title("🧠 NeuroMind AI | Research Data Extractor")
st.markdown("---")

# Status Check
st.success("✅ System Online! Docker + Streamlit Running Successfully.")
st.info("🚧 Phase 1 Complete: Infrastructure Ready. Hour 4: PDF Extraction coming next!")

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Built for Neuroscience Researchers")
    st.caption("Zero-cost RAG Pipeline on HF Spaces")
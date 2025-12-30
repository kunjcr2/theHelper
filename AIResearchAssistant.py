import streamlit as st
from dotenv import load_dotenv
from assistant import check_api_key

load_dotenv()

st.set_page_config(
    page_title="theHelper - AI Research Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("theHelper")
st.subheader("AI-Powered Research Assistant")

# API Key Status
if check_api_key():
    st.success("OpenAI API configured", icon="✓")
else:
    st.error("Please add your OpenAI API key to `.env`", icon="⚠")

st.markdown("""
---

### How It Works

**theHelper** uses a hybrid approach for fast, cost-effective PDF analysis:

| Feature | Technology | Speed |
|---------|------------|-------|
| Embeddings | SentenceTransformers | Instant |
| Summarization | BART (local) | ~3-5s |
| Q&A | OpenAI GPT-4o-mini | ~2-3s |

---

### Quick Start

1. Add your OpenAI API key to `.env`
2. Go to **"Try it out"** in the sidebar
3. Upload a PDF → get summary → ask questions

---

### Cost

- **Summarization**: Free (runs locally)
- **Q&A**: ~$0.001 per question (OpenAI API)

---

*Contact: kunjcr2@gmail.com*
""")
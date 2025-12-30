# theHelper – AI Research Assistant

> Upload a PDF, get a summary, and ask questions — fast and accurately.

---

## Overview

**theHelper** is a hybrid RAG system for PDF analysis:
- **Local transformers** for fast summarization (no API calls)
- **OpenAI** for intelligent question answering

---

## Architecture

| Component | Technology | API Cost |
|-----------|------------|----------|
| Embeddings | `all-MiniLM-L6-v2` | Free |
| Summarization | `facebook/bart-large-cnn` | Free |
| Q&A | OpenAI `gpt-4o-mini` | Per query |

---

## Setup

```bash
# Clone and install
git clone https://github.com/yourusername/theHelper.git
cd theHelper
pip install -r requirements.txt

# Add API key to .env
echo "OPENAI_API_KEY=sk-your-key" > .env

# Run
streamlit run AIResearchAssistant.py
```

---

## Usage

### Web Interface
1. Run `streamlit run AIResearchAssistant.py`
2. Go to "Try it out" → upload PDF
3. View summary (instant, local) → ask questions (uses API)

### Python
```python
from assistant import Assistant

helper = Assistant()
summary = helper.get_summary('doc.pdf')  # Local BART
answer = helper.ask("What is the main finding?")  # OpenAI
```

---

## Project Structure

```
theHelper/
├── assistant.py          # Core RAG (hybrid)
├── AIResearchAssistant.py
├── pages/Try it out.py
├── requirements.txt
├── .env                  # API key
└── .gitignore
```

---

## License

MIT

---

## Contact

kunjcr2@gmail.com

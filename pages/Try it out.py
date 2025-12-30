import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from assistant import Assistant, check_api_key

load_dotenv()

st.set_page_config(page_title="Try It Out - theHelper", page_icon="📄")

st.title("Try It Out")

# Check API key first
if not check_api_key():
    st.error("Please add your OpenAI API key to the `.env` file first.")
    st.stop()

# Initialize assistant in session state
if "assistant" not in st.session_state:
    st.session_state.assistant = Assistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

assistant = st.session_state.assistant

# File upload
st.markdown("Upload a PDF document to analyze.")
file = st.file_uploader("Choose PDF file", type=["pdf"])

if file is not None:
    stream = BytesIO(file.read())
    
    # Process document and show summary
    if not st.session_state.document_processed:
        with st.spinner("Processing document..."):
            try:
                summary = assistant.get_summary(stream)
                st.session_state.document_processed = True
                st.session_state.summary = summary
            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.stop()
    
    # Display summary
    st.subheader("Summary")
    st.write(st.session_state.summary)
    
    st.divider()
    
    # Chat interface
    st.subheader("Ask Questions")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about the document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = assistant.ask(question)
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.session_state.document_processed = False
    st.session_state.messages = []
    st.info("Upload a PDF to get started.")
import requests
import streamlit as st 

# -----------------------------
# Config
# -----------------------------
API_URL_UPLOAD = "http://127.0.0.1:8000/upload_pdf"
API_URL_ASK = "http://127.0.0.1:8000/ask"
SESSION_ID = "ui"

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="JVAI Policy Chatbot", layout="centered")

# Header
st.title("📄 JVAI Policy Chatbot")
st.markdown("Upload a policy document and ask questions about its content")

# File upload section
st.subheader("Upload Document")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        resp = requests.post(API_URL_UPLOAD, files=files)
        
        if resp.status_code == 200:
            st.success(f"✅ Document processed successfully!")
        else:
            st.error(f"❌ Upload failed: {resp.text}")

# Question section
st.subheader("Ask Questions")
question = st.text_input("Enter your question")
use_llm = st.checkbox("Use LLM for summarized answers", value=True)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question")
    elif uploaded_file is None:
        st.warning("Please upload a document first")
    else:
        with st.spinner("Finding answer..."):
            payload = {
                "session_id": SESSION_ID,
                "question": question,
                "use_llm": use_llm
            }
            resp = requests.post(API_URL_ASK, json=payload)

            if resp.status_code == 200:
                data = resp.json()
                
                # Answer section
                st.markdown("### Answer")
                st.write(data["answer"])
                
                # Sources section
                with st.expander("View Sources"):
                    st.markdown("### Sources")
                    for r in data["results"]:
                        snippet = r["text"].strip().replace("\n", " ")
                        st.markdown(f"**Page {r['page']}** (Relevance: {r['score']:.3f})")
                        st.caption(snippet[:200] + "...")
                        st.divider()
            else:
                st.error(f"❌ Error: {resp.text}")

# Footer
st.markdown("---")
st.caption("JVAI Policy Chatbot | Upload PDF documents and ask questions")
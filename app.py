import streamlit as st
import requests

st.set_page_config(page_title="Agri-Scheme RAG Assistant", layout="centered")

st.title("🌾 Agricultural Scheme RAG Assistant")

# Navigation Tabs
tab1, tab2 = st.tabs(["📤 Upload Pipeline", "💬 Chatbox"])

BACKEND_URL = "http://127.0.0.1:8000"

with tab1:
    st.header("Upload Agriculture Scheme PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    # Automatically process when a file is uploaded
    if uploaded_file is not None:
        with st.spinner("Automatically processing, embedding into FAISS, and saving..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post(f"{BACKEND_URL}/upload", files=files)
                if response.status_code == 200:
                    st.success(response.json().get("message"))
                else:
                    st.error(f"Error: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Could not connect to backend server: {e}")

with tab2:
    st.header("Chat with Scheme Database")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about any agricultural scheme..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching vector database..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/chat", json={"question": prompt})
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data["sources"]
                        
                        full_response = f"{answer}\n\n---\n**Sources:** {', '.join(set(sources))}"
                        st.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        st.error(response.json().get("detail"))
                except Exception as e:
                    st.error(f"Connection error: {e}")
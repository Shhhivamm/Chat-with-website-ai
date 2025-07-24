import os
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Theme toggle
theme = st.radio("🌓 Choose Theme", ["Light", "Dark"], horizontal=True)
if theme == "Dark":
    st.markdown("""
    <style>
        body, .stApp { background-color: #0e1117; color: white; }
        .stTextInput input, .stTextArea textarea, .stButton>button {
            background-color: #222; color: white; border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Save chat history locally
def save_chat_log():
    os.makedirs("chat_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"chat_logs/chat_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(st.session_state.chat_history, f, indent=4)

# Load and embed website content
@st.cache_resource
def get_vectorstore_from_url(urls):
    loader = WebBaseLoader(urls)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

# Page config
st.set_page_config(page_title="Chat with Website 🤖", layout="centered")

# App UI
st.title("💬 Chat with a Website using Gemini AI")
st.markdown("Paste one or more website URLs (not JS-heavy) and ask questions about them!")

# Input URLs
url_input = st.text_area("📄 Enter one or more URLs (one per line)", height=150, placeholder="https://realpython.com/\nhttps://geeksforgeeks.org/...")
urls = [u.strip() for u in url_input.split("\n") if u.strip()]

# Process URLs
if urls:
    with st.spinner("⏳ Loading and processing websites..."):
        try:
            vectorstore = get_vectorstore_from_url(urls)
        except Exception as e:
            import traceback
            st.error(f"❌ Failed to load website(s): {str(e)}")
            st.code(traceback.format_exc(), language='python')
            st.stop()

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.2
        )

        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ Websites loaded and ready!")

        # Chat state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Clear history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()



        # Summarize
        if st.button("📝 Summarize Website(s)"):
            with st.spinner("Summarizing..."):
                summary = qa_chain.run("Summarize these websites in simple terms.")
                st.session_state.chat_history.append(("Summarize these websites in simple terms.", summary))
                save_chat_log()
                st.markdown("### 📋 Summary")
                st.markdown(summary)

        # Ask questions
        query = st.text_input("❓ Ask a question", key="user_question")
        if query:
            with st.spinner("🤖 Thinking..."):
                response = qa_chain.run(query)
                st.session_state.chat_history.append((query, response))
                save_chat_log()

        # Show chat
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("🧠 Chat History")
            for user, bot in st.session_state.chat_history[::-1]:
                st.markdown(f"**🧑‍💻 You:** {user}")
                st.markdown(f"**🤖 Gemini:** {bot}")
                st.markdown("---")

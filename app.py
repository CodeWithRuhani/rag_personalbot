import streamlit as st
import os
import time
import pickle
import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# Set API Key for Gemini.
# It is best practice to use Streamlit Secrets for this.
# You can also use an environment variable.
# Remove the hardcoded key before deploying!
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize model
model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------- DATA INGESTION & VECTORSTORE SETUP --------------------

@st.cache_resource
def get_vectorstore():
    """
    This function handles the data ingestion and vector store creation.
    It's cached with st.cache_resource to run only once per session.
    """
    DB_FAISS_PATH = "index_faiss"

    # Check if FAISS index already exists
    if os.path.exists(DB_FAISS_PATH):
        st.info("Loading existing FAISS vector store...")
    else:
        st.info("Creating new FAISS vector store from mydata.txt...")
        try:
            # Load the text document
            loader = TextLoader("mydata.txt", encoding="utf-8")
            documents = loader.load()

            if not documents:
                st.error("❌ mydata.txt is empty or not loaded correctly.")
                return None

            # Split into chunks
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)

            # Use HuggingFace embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Build FAISS vector store
            db = FAISS.from_documents(docs, embeddings)

            # Save the FAISS index locally
            db.save_local(DB_FAISS_PATH)
            st.success("✅ Data ingestion complete. FAISS index saved.")

        except Exception as e:
            st.error(f"Error during data ingestion: {e}")
            return None

    try:
        # Load the embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load the FAISS vector store
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

# Get the vector store. This will either load or create it.
vectorstore = get_vectorstore()

# -------------------- BACKEND FUNCTIONS --------------------

def generate_answer(query):
    # --- Part 1: Pre-emptive Keyword Check (The Direct Fix) ---
    # This ensures a perfect answer for key questions about you.
    # It bypasses the RAG search to guarantee a correct response.
    keywords = ["ruhani", "gera", "ruhani gera", "developer", "creator", "who made you"]
    if any(word in query.lower().split() for word in keywords):

        # A well-structured, comprehensive response.
        predefined_response = f"""
        I was created by **Ruhani Gera**, an aspiring data scientist and AI enthusiast. My purpose is to provide information about her professional profile.

        **About Ruhani:**
        - **Education**: B.Sc. in Artificial Intelligence and Data Science (expected 2026) at Sri Guru Teg Bahadur Khalsa College.
        - **Skills**: Python, C++, R, Machine Learning, Deep Learning, TensorFlow, and Flask.
        - **Projects**: Hand gestures-based calculator and this personal AI assistant.

        **Contact & Links:**
        - **Email**: ruhanigera7@gmail.com
        - **LinkedIn**: [Ruhani Gera on LinkedIn](https://www.linkedin.com/in/ruhani-gera-851454300)
        - **GitHub**: [Ruhani Gera on GitHub](https://www.github.com/CodeWithRuhani)
        """
        return predefined_response

    # --- Part 2: Standard RAG Logic (For all other questions) ---
    try:
        # Use a lower relevance threshold for other questions
        RELEVANCE_THRESHOLD = 0.40

        if vectorstore:
            docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=3)
            relevant_docs = [doc for doc, score in docs_with_scores if score > RELEVANCE_THRESHOLD]

            if relevant_docs:
                context = "\n".join([doc.page_content for doc in relevant_docs])
                prompt = f"""
                You are a helpful assistant. Use the following context to answer the question.

                Context:
                {context}

                Question:
                {query}
                """
            else:
                prompt = f"""
                You are a helpful and knowledgeable AI assistant. Answer the following question.

                Question:
                {query}
                """
        else:
            prompt = f"""
            You are a helpful and knowledgeable AI assistant. Answer the following question.

            Question:
            {query}
            """

        response = model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text.strip()

    except Exception as e:
        return f"❌ Error: {e}"

# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Dark Theme CSS
st.markdown("""
    <style>
    .stApp {
        background: #0f1117;
        color: #e4e6eb;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #1c1f2e;
        color: white;
        padding: 20px;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: #bb86fc;
    }
    .sidebar-sub {
        font-size: 14px;
        color: #bbb;
    }
    .dev-info {
        margin-top: 20px;
        font-size: 15px;
        color: #ffcc00;
    }
    .history-item {
        background: #2b2f44;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 6px 0;
        font-size: 14px;
        color: white;
        cursor: pointer;
        border: 1px solid #444;
    }
    .history-item:hover {
        background: #3c4260;
    }
    .user-msg {
        background-color: #6a1b9a; /* purple bubble */
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0px;
        font-size: 16px;
        max-width: 75%;
        float: right;
        clear: both;
        color: #fff;
    }
    .bot-msg {
        background-color: #1f2233;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0px;
        font-size: 16px;
        max-width: 75%;
        float: left;
        clear: both;
        color: #e4e6eb;
        border: 1px solid #333;
    }
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 300px;
        right: 0;
        background: #0f1117;
        padding: 12px;
        border-top: 1px solid #333;
    }
    .stChatInput input {
        background: #2b2f44 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 14px !important;
        border: 1px solid #555 !important;
    }
    .typing-indicator {
        font-size: 14px;
        color: #bbb;
        font-style: italic;
        margin: 6px 0;
    }
    .stButton > button {
        background: #9c27b0; /* Darker purple */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 18px;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: #7b1fa2;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🤖 RAG Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Built to assist with general questions.</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='dev-info'>👩‍💻 Developer: Ruhani Gera</div>", unsafe_allow_html=True)
    st.markdown("<div class='dev-info'>💼 Role: Chatbot Developer</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Contact Button
    linkedin_url = "https://www.linkedin.com/in/ruhani-gera-851454300/"
    st.markdown(f"<a href='{linkedin_url}' target='_blank'><button style='background:#bb86fc;color:white;padding:10px 15px;border:none;border-radius:8px;cursor:pointer;'>📩 Contact with me</button></a>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<b style='color:#bb86fc;'>✨ Features:</b><br>- RAG-powered responses<br>- Context-aware conversations<br>- Simple, modern UI", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<b style='color:#bb86fc;'>🕘 History:</b>", unsafe_allow_html=True)

    if st.button("📜 Show History", use_container_width=True):
        st.session_state.show_history = not st.session_state.get("show_history", False)

    if st.session_state.get("show_history", False):
        if "history" in st.session_state and st.session_state.history:
            for speaker, msg in st.session_state.history:
                if speaker == "You":
                    st.markdown(f"<div class='history-item'>🙋 {msg}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='history-item'>🤖 {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:gray;'>No history yet...</p>", unsafe_allow_html=True)


    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.chat_start_time = time.strftime("%H:%M:%S")
        st.rerun()


# -------------------- MAIN --------------------
st.markdown("<h1 style='text-align:center; color:#bb86fc; animation: fadeIn 2s;'>💬 Welcome to RAG Chatbot</h1>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.chat_start_time = time.strftime("%H:%M:%S")

st.markdown(f"<p style='text-align:center; color:gray;'>🕒 Session started at {st.session_state.chat_start_time}</p>", unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    for speaker, text in st.session_state.history:
        if speaker == "You":
            st.markdown(f"<div class='user-msg'>🙋 {text}</div>", unsafe_allow_html=True)
        else:
            # THIS IS THE KEY FIX
            # Instead of wrapping text in a div, pass the raw markdown and class
            st.markdown(f"<div class='bot-msg'>🤖 {text}</div>", unsafe_allow_html=True)

# -------------------- CHAT INPUT --------------------
st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)

placeholder_text = "💡 Ask me anything..."
user_input = st.chat_input(placeholder_text)

if "pending_input" in st.session_state and st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = ""

if user_input:
    st.session_state.history.append(("You", user_input))

    with chat_container:
        st.markdown("<div class='typing-indicator'>🤖 Bot is typing...</div>", unsafe_allow_html=True)
        st.empty()

    answer = generate_answer(user_input)
    st.session_state.history.append(("Bot", answer))
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import streamlit as st
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# ===== Configuration =====
load_dotenv()

# ===== Initialize Database =====


def init_database():
    conn = sqlite3.connect("chat_assistant.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT,
                    answer TEXT,
                    timestamp TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    full_text TEXT,
                    upload_time TEXT
                )''')
    conn.commit()
    conn.close()

# ===== Database Helpers =====


def save_chat_to_db(question, answer):
    conn = sqlite3.connect("chat_assistant.db")
    c = conn.cursor()
    c.execute("INSERT INTO chat_logs (question, answer, timestamp) VALUES (?, ?, ?)",
              (question, answer, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def display_chat_history():
    conn = sqlite3.connect("chat_assistant.db")
    c = conn.cursor()
    c.execute(
        "SELECT question, answer FROM chat_logs ORDER BY timestamp DESC LIMIT 5")
    rows = c.fetchall()
    conn.close()

    if rows:
        st.subheader("🕘 Chat History (Latest 5)")
        for idx, (q, a) in enumerate(rows):
            with st.expander(f"{idx+1}. Q: {q}"):
                st.markdown(f"**A:** {a}")


# ===== API Key Handling =====
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = st.text_input(
        "Enter your Google Generative AI API Key:", type="password")
    if not api_key:
        st.info("Please enter your API key to continue.", icon="🗝️")
        st.stop()
    else:
        st.success("API key received. Verifying...")

try:
    genai.configure(api_key=api_key)
    test_model = genai.GenerativeModel("gemini-1.5-flash-001")
    _ = test_model.generate_content("Hello!").text
except Exception as e:
    st.error("Invalid or unauthorized API key. Please double-check and try again.")
    st.stop()

# ===== Streamlit Page Setup =====
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="centered"
)

# ===== Core Functions =====


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.warning(f"⚠️ Error reading {pdf.name}: {str(e)}")
    return text


def get_text_chunks(text):
    if not text.strip():
        st.error("No text extracted from PDFs!")
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text chunks to process!")
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer strictly from the context. If unsure, say "I couldn't find this in the documents."
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def handle_user_query(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local("faiss_index", embeddings,
                              allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_question, k=3)

        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        final_answer = response["output_text"]
        save_chat_to_db(user_question, final_answer)

        st.success("💡 " + final_answer)

        # Show reference snippet
        if docs:
            st.markdown("🧾 **Referenced Snippet**:")
            st.info(docs[0].page_content[:500])

    except Exception as e:
        st.error(f"🔴 Error processing query: {str(e)}")

# ===== Main Application =====


def main():
    init_database()

    st.title("📄 Chat with Your Documents")
    st.caption("Upload PDFs and ask questions about their content")

    user_question = st.chat_input("Ask anything about your documents...")
    if user_question:
        with st.spinner("Analyzing documents..."):
            handle_user_query(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs here",
            type="pdf",
            accept_multiple_files=True,
            help="Max 10MB per file"
        )

        if st.button("Process Documents", type="primary"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF")
            else:
                with st.status("Processing...", expanded=True) as status:
                    st.write("Extracting text from PDFs...")
                    raw_text = get_pdf_text(pdf_docs)

                    # Save document to DB
                    conn = sqlite3.connect("chat_assistant.db")
                    c = conn.cursor()
                    for pdf in pdf_docs:
                        text = get_pdf_text([pdf])
                        c.execute("INSERT INTO documents (filename, full_text, upload_time) VALUES (?, ?, ?)",
                                  (pdf.name, text, datetime.now().isoformat()))
                    conn.commit()
                    conn.close()

                    st.write("Preparing document chunks...")
                    text_chunks = get_text_chunks(raw_text)

                    st.write("Generating search index...")
                    get_vector_store(text_chunks)
                    st.write("Text chunks created:", len(text_chunks))

                    status.update(label="Processing complete!",
                                  state="complete")
                st.toast("Documents ready for questioning!", icon="✅")

    # Show last 5 chat messages
    display_chat_history()


if __name__ == "__main__":
    main()

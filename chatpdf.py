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
from dotenv import load_dotenv
# ===== Configuration =====
load_dotenv()
# from PyPDF2 import PdfReader
# Added for safety settings


# Try to get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

# If not found in .env, ask the user to input it
if not api_key:
    api_key = st.text_input(
        "Enter your Google Generative AI API Key:", type="password")
    if not api_key:
        st.info("Please enter your API key to continue.", icon="🗝️")
        st.stop()

# Try configuring Gemini with the provided API key
try:
    genai.configure(api_key=api_key)

    # Validate API key with a test prompt
    test_model = genai.GenerativeModel("gemini-1.5-flash-001")
    _ = test_model.generate_content("Hello!").text

except Exception as e:
    st.error("Invalid or unauthorized API key. Please double-check and try again.")
    st.stop()

# ===== Core Functions =====


def get_pdf_text(pdf_docs):
    """Extracts text from multiple PDFs"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle None returns
        except Exception as e:
            st.warning(f"⚠️ Error reading {pdf.name}: {str(e)}")
    return text


def get_text_chunks(text):
    """Splits text into manageable chunks"""
    if not text.strip():
        st.error("No text extracted from PDFs!")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    """Creates and saves embeddings database"""
    if not text_chunks:
        st.error("No text chunks to process!")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # Fixed typo in "embedding"
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():
    """Sets up the QA chain with safety controls"""
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

# ===== UI & Interaction =====


def handle_user_query(user_question):
    """Processes questions against stored documents"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001")  # Fixed typo
        db = FAISS.load_local("faiss_index", embeddings,
                              allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_question, k=3)

        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.success("💡 " + response["output_text"])
    except Exception as e:
        st.error(f"🔴 Error processing query: {str(e)}")


def main():
    st.set_page_config(
        page_title="PDF Chat Assistant",
        page_icon="📄",
        layout="centered"
    )

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

                    st.write("Preparing document chunks...")
                    text_chunks = get_text_chunks(raw_text)

                    st.write("Generating search index...")
                    get_vector_store(text_chunks)

                    status.update(label="Processing complete!",
                                  state="complete")
                st.toast("Documents ready for questioning!", icon="✅")


if __name__ == "__main__":
    main()

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Page layout & styling
st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄", layout="wide")
st.title("📄 PDF Knowledge Assistant")

# Cache the embedding model to avoid reloading on every rerun
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Initialize Session State variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Sidebar: PDF Upload & Processing
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Parsing and indexing documents..."):
                docs = []
                for uploaded_file in uploaded_files:
                    # Write temporary file to disk for PyPDFLoader
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    loader = PyPDFLoader(tmp_path)
                    docs.extend(loader.load())
                    os.remove(tmp_path)

                # Split text into manageable chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(docs)

                # Initialize in-memory Chroma vector store
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=embedding_model
                )

                # Configure retriever
                st.session_state.retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
                )
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.success("Indexing complete! You can now start asking questions.")

    if st.session_state.processed_files:
        st.divider()
        st.markdown("**Active Documents:**")
        for name in st.session_state.processed_files:
            st.markdown(f"- `{name}`")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Set up LLM and Prompt Template with Chat History support
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful AI assistant. Use only the provided context to answer the question. "
     "If the answer is not present in the context, say: 'I could not find the answer in the document.'\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Display existing chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat Input & Response Generation
if user_query := st.chat_input("Ask a question about your uploaded document..."):
    if not st.session_state.retriever:
        st.info("Please upload and process a PDF in the sidebar before asking questions.")
    else:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve documents
        retrieved_docs = st.session_state.retriever.invoke(user_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate response using history and context
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                final_prompt = prompt.invoke({
                    "context": context_text,
                    "chat_history": st.session_state.messages,
                    "question": user_query
                })
                response = llm.invoke(final_prompt)
                st.markdown(response.content)

                # Show retrieved chunks in an expander for transparency
                with st.expander("🔍 View Retrieved Context"):
                    for idx, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**Chunk {idx}:**")
                        st.caption(doc.page_content)
                        st.divider()

        # Update session state history
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.session_state.messages.append(AIMessage(content=response.content))
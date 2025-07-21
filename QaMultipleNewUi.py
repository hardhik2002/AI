import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import fitz  # PyMuPDF

# --- Helpers ---
def read_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract full text from PDF bytes."""
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in pdf)

# Create or retrieve a cached vector store (ignore unhashable docs arg)
@st.cache_resource(show_spinner=False)
def get_vector_store(_documents, api_key: str) -> Chroma:
    """Builds a Chroma vector store from document chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    return Chroma.from_documents(_documents, embeddings)

# Prompt template
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert assistant. "
        "Use the following context to answer comprehensively (>=1000 words). "
        "Only use provided context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
)

# Generate response
def generate_response(uploaded_files, api_key, query_text):
    docs = []
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        uploaded_file.seek(0)
        if uploaded_file.type == "application/pdf":
            text = read_pdf_bytes(bytes_data)
        else:
            text = bytes_data.decode("utf-8")
        docs.extend(
            splitter.create_documents([text], metadatas=[{"source": uploaded_file.name}])
        )

    vectorstore = get_vector_store(docs, api_key)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.5}
    )
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=api_key, temperature=0.0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )
    result = chain({"query": query_text})
    return result.get("result", "")

# --- Streamlit UI ---
st.set_page_config(page_title="📖 Interactive PDF RAG Reader", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📚 Your Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or Text",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    st.markdown("---")
    st.text_input("OpenAI API Key", key="api_key", type="password")

# Main interface
st.markdown("# Interactive Reader 🚀")
query_text = st.text_input(
    "Ask a question from the uploaded docs:",
    disabled=not uploaded_files
)

# Chat interface
if query_text and st.session_state.get("api_key", "").startswith("sk-"):
    if st.button("Send"):
        with st.spinner("Processing... 📡"):
            answer = generate_response(
                uploaded_files,
                st.session_state.api_key,
                query_text
            )
        st.markdown(f"**You:** {query_text}")
        st.markdown(f"**Assistant:** {answer}")
else:
    if uploaded_files and not st.session_state.get("api_key", "").startswith("sk-"):
        st.warning("Enter your OpenAI API key in the sidebar to enable chat.")

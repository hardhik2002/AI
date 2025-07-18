import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import fitz  # PyMuPDF

# --- Helpers ---
def read_pdf(file) -> str:
    """Extract full text from a PDF file."""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = []
    for page in pdf:
        text.append(page.get_text())
    return "\n".join(text)

# Cache the vector store between runs (ignore unhashable documents param)
@st.cache_resource(show_spinner=False)
def get_vector_store(_documents, api_key: str) -> Chroma:
    """Create or retrieve a cached Chroma vector store from document chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    return Chroma.from_documents(_documents, embeddings)

# Build a prompt template enforcing minimum length
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert assistant. "
        "Use the following retrieved context to answer the question comprehensively. "
        "Your answer must be at least 1000 words long. "
        "Ensure absolute accuracy by only using the provided context; do not hallucinate."
        "\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
)

# Generate response via a retrieval QA chain
def generate_response(uploaded_files, api_key, query_text):
    # 1. Read and chunk all documents
    docs = []
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for uploaded_file in uploaded_files:
        full_text = read_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else uploaded_file.read().decode("utf-8")
        docs.extend(splitter.create_documents([full_text], metadatas=[{"source": uploaded_file.name}]))

    # 2. Create or retrieve cached vector store
    vectorstore = get_vector_store(docs, api_key)

    # 3. Configure retriever with MMR for diverse sources
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5}
    )

    # 4. Build the RetrievalQA chain Build the RetrievalQA chain
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=api_key, temperature=0.0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    # 5. Invoke chain
    result = chain({"query": query_text})
    answer = result.get("result", "")
    sources = result.get("source_documents", [])

    # 6. Format with citations
    citation_text = "\n\nSources:\n"
    for doc in sources:
        citation_text += f"- {doc.metadata.get('source', 'unknown')}\n"

    return answer + citation_text

# --- Streamlit App ---
st.title(" RAG-Powered PDF Reader")

uploaded_files = st.file_uploader('Upload PDFs or text files', type=['pdf', 'txt'], accept_multiple_files=True)
query_text = st.text_input('Ask your question:', placeholder='Enter a detailed query...', disabled=not uploaded_files)

with st.form('query_form'):
    api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_files and query_text))
    submitted = st.form_submit_button('Submit')

    if submitted:
        if not api_key.startswith('sk-'):
            st.error("Please enter a valid OpenAI API key starting with sk-")
        else:
            with st.spinner("Generating..."):
                response = generate_response(uploaded_files, api_key, query_text)
            st.success("Answer generated!")
            st.markdown(response)

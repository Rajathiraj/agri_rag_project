import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Load OPENAI_API_KEY from .env
load_dotenv()

app = FastAPI(title="Agri-Scheme RAG Backend")

DB_PATH = "vector_store"

# Initialize OpenAI Embeddings and Chat LLM
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Agri-Scheme RAG API is running!"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # 1. Data Load
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        # 2. Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        # 3 & 4. Embed with OpenAI & Persist locally in FAISS
        if os.path.exists(DB_PATH):
            db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(chunks)
        else:
            db = FAISS.from_documents(chunks, embeddings)
        
        db.save_local(DB_PATH)
        
        return {"message": f"Successfully processed {file.filename} and saved to FAISS."}
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/chat")
async def chat_with_docs(request: QueryRequest):
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=400, detail="Vector database is empty. Please upload documents first.")
    
    # Load persisted FAISS index
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve top 3 relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(request.question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Generate grounded response using OpenAI
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant for agricultural schemes. "
        "Answer the user's question using ONLY the context provided below. "
        "If the answer is not present in the context, state that you do not have that information.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": request.question})
    
    return {
        "question": request.question,
        "answer": response.content,
        "sources": [doc.metadata.get("source", "unknown") for doc in relevant_docs]
    }
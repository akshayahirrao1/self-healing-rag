import os
import sys
import json
import asyncio
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Ensure the src module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import build_graph
from src.ingestion import split_documents, store_in_chroma

app = FastAPI(title="Self-Healing RAG API")

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the LangGraph app once at startup
rag_app = build_graph()

class ChatRequest(BaseModel):
    question: str

def serialize_state(state_update):
    """Helper to serialize non-JSON serializable objects like LangChain Documents."""
    serialized = {}
    for k, v in state_update.items():
        if k == "documents" and isinstance(v, list):
            serialized[k] = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in v if hasattr(doc, "page_content")
            ]
        else:
            serialized[k] = v
    return serialized

async def stream_rag_events(question: str):
    initial_state = {
        "question": question,
        "current_query": question,
        "documents": [],
        "answer": "",
        "critique": "",
        "decision": "",
        "retry_count": 0
    }
    
    # LangGraph's .stream() yields the output of each node as it finishes.
    # We use stream_mode="updates" to get incremental state changes.
    for output in rag_app.stream(initial_state, stream_mode="updates"):
        # output is a dict where key is the node name and value is the state update
        for node_name, state_update in output.items():
            
            event_data = {
                "node": node_name,
                "state": serialize_state(state_update)
            }
            
            # Server-Sent Events (SSE) format requires "data: <json>\n\n"
            yield f"data: {json.dumps(event_data)}\n\n"
            
            # Small sleep to ensure the frontend can render animations
            await asyncio.sleep(0.1)

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Accepts a question and streams back the LangGraph execution steps via SSE.
    """
    return StreamingResponse(
        stream_rag_events(request.question), 
        media_type="text/event-stream"
    )

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF or TXT file, saves it, chunks it, and ingests it into ChromaDB.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        if file.filename.endswith('.pdf'):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
        else:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path)
            
        docs = loader.load()
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
        store_in_chroma(chunks)
        
        return {"status": "success", "message": f"Successfully ingested {file.filename}!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

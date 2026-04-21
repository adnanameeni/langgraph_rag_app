from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import requests

# ======================
# ENV
# ======================
load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ======================
# LLM (GROQ)
# ======================
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# ======================
# EMBEDDINGS (FREE)
# ======================
embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# ======================
# THREAD STORAGE
# ======================
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    return _THREAD_RETRIEVERS.get(str(thread_id)) if thread_id else None


# ======================
# PDF INGESTION
# ======================
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        temp_path = f.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


# ======================
# TOOLS (FIXED)
# ======================
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform arithmetic operations: add, sub, mul, div.
    """
    if operation == "add":
        return {"result": first_num + second_num}
    if operation == "sub":
        return {"result": first_num - second_num}
    if operation == "mul":
        return {"result": first_num * second_num}
    if operation == "div":
        if second_num == 0:
            return {"error": "Division by zero"}
        return {"result": first_num / second_num}
    return {"error": "Invalid operation"}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Get stock price for a symbol using Alpha Vantage API.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=demo"
    )
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Search uploaded PDF document.
    """
    retriever = _get_retriever(thread_id)

    if not retriever:
        return {"error": "No PDF uploaded"}

    docs = retriever.invoke(query)

    return {
        "query": query,
        "context": [d.page_content for d in docs],
    }


tools = [calculator, get_stock_price, rag_tool]


# ======================
# LLM WITH TOOLS
# ======================
llm_with_tools = llm.bind_tools(tools)


# ======================
# STATE
# ======================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ======================
# CHAT NODE (FIXED RAG FORCE LOGIC)
# ======================
def chat_node(state: ChatState, config=None):

    thread_id = config.get("configurable", {}).get("thread_id") if config else None

    user_message = state["messages"][-1].content.lower()

    # 🔥 FORCE RAG DETECTION (IMPORTANT FIX)
    is_pdf_query = any(
        keyword in user_message
        for keyword in [
            "pdf", "document", "file", "write", "explain", "summarize",
            "what does", "according to", "in the text"
        ]
    )

    system_message = SystemMessage(
        content=(
            "You are a strict AI assistant.\n"
            "\n"
            "RULES:\n"
            "- If question is about ANY document/PDF → MUST use rag_tool\n"
            "- Never answer document questions from memory\n"
            "- For math → calculator\n"
            "- For stocks → get_stock_price\n"
        )
    )

    messages = [system_message, *state["messages"]]

    # 🔥 EXTRA FORCE: help model choose tool correctly
    if is_pdf_query:
        messages.append(
            SystemMessage(
                content=f"IMPORTANT: Use rag_tool with thread_id={thread_id} for this query."
            )
        )

    response = llm_with_tools.invoke(messages, config=config)

    return {"messages": [response]}


# ======================
# TOOL NODE
# ======================
tool_node = ToolNode(tools)


# ======================
# SQLITE MEMORY
# ======================
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# ======================
# GRAPH
# ======================
graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

chatbot = graph.compile(checkpointer=checkpointer)


# ======================
# HELPERS
# ======================
def retrieve_all_threads() -> list[str]:
    try:
        return list({
            c.config["configurable"]["thread_id"]
            for c in checkpointer.list(None)
        })
    except Exception:
        return []


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> Optional[dict]:
    return _THREAD_METADATA.get(str(thread_id))
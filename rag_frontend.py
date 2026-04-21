import uuid
import streamlit as st
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# ========================= UI CONFIG =========================
st.set_page_config(
    page_title="Smart AI Assistant",
    page_icon="🧠",
    layout="wide"
)

# ========================= UTILITIES =========================
def generate_thread_id():
    return str(uuid.uuid4())


def generate_chat_name():
    """Human-friendly chat names (HCI improvement)"""
    return f"Chat {len(st.session_state['chat_threads']) + 1}"


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["chat_names"][thread_id] = generate_chat_name()

    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
        st.session_state["chat_names"][thread_id] = generate_chat_name()


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ========================= SESSION STATE =========================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "chat_names" not in st.session_state:
    st.session_state["chat_names"] = {}

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]

selected_thread = None

# ========================= CUSTOM CSS (HCI UI) =========================
st.markdown("""
<style>
.chat-bubble-user {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0px;
    max-width: 75%;
}

.chat-bubble-ai {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0px;
    max-width: 75%;
}

.sidebar-title {
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ========================= SIDEBAR =========================
st.sidebar.title("🧠 Smart AI Assistant")

chat_name = st.session_state["chat_names"].get(thread_key, "New Chat")
st.sidebar.markdown(f"### 💬 {chat_name}")

st.sidebar.caption(f"Session ID (hidden): {thread_key[:8]}...")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# PDF status
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"📄 {latest_doc.get('filename')} \n"
        f"{latest_doc.get('chunks')} chunks | {latest_doc.get('documents')} pages"
    )
else:
    st.sidebar.info("No PDF uploaded")

uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:
    if uploaded_pdf.name not in thread_docs:
        with st.sidebar.status("Indexing PDF..."):
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            st.sidebar.success("PDF ready ✅")

st.sidebar.divider()
st.sidebar.subheader("🕘 History")

if not threads:
    st.sidebar.write("No chats yet")
else:
    for t in threads:
        name = st.session_state["chat_names"].get(str(t), "Chat")
        if st.sidebar.button(f"💬 {name}", key=f"t-{t}"):
            selected_thread = t

# ========================= MAIN UI =========================
st.title("💬 Multi-Utility AI Chatbot")

st.caption("Powered by LangGraph + Groq + RAG")

# ========================= CHAT DISPLAY =========================
for msg in st.session_state["message_history"]:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-ai'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

# ========================= INPUT =========================
user_input = st.chat_input("Ask anything... PDF, web, calculator, stock...")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_box = st.empty()

        def stream_response():
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    status_box.info(f"🔧 Using tool: {getattr(chunk, 'name', 'tool')}")

                if isinstance(chunk, AIMessage):
                    yield chunk.content

        response = st.write_stream(stream_response())

        status_box.success("Done ✅")

    st.session_state["message_history"].append(
        {"role": "assistant", "content": response}
    )

    meta = thread_document_metadata(thread_key)
    if meta:
        st.caption(f"📄 {meta.get('filename')} | {meta.get('chunks')} chunks")

# ========================= THREAD SWITCH =========================
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    formatted = []
    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        formatted.append({"role": role, "content": m.content})

    st.session_state["message_history"] = formatted
    st.rerun()
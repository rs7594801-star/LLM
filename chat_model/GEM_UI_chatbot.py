import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json

load_dotenv()

# Page setup
st.set_page_config(
    page_title="KNOGHT Workspace",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, modern dark theme styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Top workspace header */
    .header-box {
        padding-bottom: 1.2rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .header-box h2 {
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 0.2rem;
        color: #f0f6fc;
    }
    .header-box p {
        color: #8b949e;
        font-size: 0.9rem;
        margin: 0;
    }

    /* Chat bubble polish */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #161b22;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #1c2128;
        border: 1px solid rgba(56, 139, 253, 0.2);
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Minimal quick chip buttons */
    .chip-container button {
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background-color: transparent !important;
        color: #8b949e !important;
        font-size: 0.82rem !important;
        padding: 0.35rem 0.85rem !important;
        transition: all 0.2s ease;
    }
    .chip-container button:hover {
        border-color: #58a6ff !important;
        color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# Avatar icons (native Streamlit Material icons)
USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/auto_awesome:"

# Initialize session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.subheader("Model Configuration")
    
    model_choice = st.selectbox(
        "Engine",
        ["mistral-small-latest", "mistral-large-latest", "codestral-latest", "mistral-small-2506"],
        index=0
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.05,
        help="Higher values make output more random; lower values make it deterministic."
    )
    
    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=256,
        max_value=4096,
        value=2048,
        step=256
    )

    st.markdown("---")
    
    system_prompt = st.text_area(
        "System Instructions",
        value="You are an expert AI collaborator. Provide concise, highly accurate, and well-structured answers.",
        height=100
    )

    st.markdown("---")
    
    # Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
    with col2:
        export_data = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in st.session_state.messages
        ]
        st.download_button(
            label="Export JSON",
            data=json.dumps(export_data, indent=2),
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True
        )

# Main Area Header
st.markdown("""
<div class="header-box">
    <h2>Mistral Workspace</h2>
    <p>Streamlined interface for real-time conversational inference</p>
</div>
""", unsafe_allow_html=True)

# Quick Starter Chips (Only visible on fresh chat)
selected_prompt = None
if len(st.session_state.messages) == 0:
    st.markdown("<p style='color: #8b949e; font-size: 0.85rem; margin-bottom: 0.6rem;'>Suggested prompts</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown('<div class="chip-container">', unsafe_allow_html=True)
        if st.button("Python Script", key="p1", use_container_width=True):
            selected_prompt = "Write a clean, modular Python script for asynchronous file processing."
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="chip-container">', unsafe_allow_html=True)
        if st.button("Code Refactor", key="p2", use_container_width=True):
            selected_prompt = "Review this code structure and suggest performance improvements."
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c3:
        st.markdown('<div class="chip-container">', unsafe_allow_html=True)
        if st.button("Explain Architecture", key="p3", use_container_width=True):
            selected_prompt = "Explain the difference between event-driven architecture and microservices."
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chip-container">', unsafe_allow_html=True)
        if st.button("Draft Documentation", key="p4", use_container_width=True):
            selected_prompt = "Draft a clean README template for a production REST API project."
        st.markdown('</div>', unsafe_allow_html=True)

# Render Chat History
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User Input Resolution
user_input = st.chat_input("Message KNIGHT...")
prompt = selected_prompt or user_input

if prompt:
    # Append & render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Build LangChain Core message list
    langchain_messages = []
    if system_prompt.strip():
        langchain_messages.append(SystemMessage(content=system_prompt))

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))

    # Initialize model
    model = ChatMistralAI(
        model=model_choice,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Stream response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk in model.stream(langchain_messages):
                full_response += chunk.content
                response_placeholder.markdown(full_response + " ▌")
            response_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            if selected_prompt:
                st.rerun()
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
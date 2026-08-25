# CLAUDE 

import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

st.set_page_config(page_title="Mistral Chat", page_icon="🤖", layout="centered")

DEFAULT_SYSTEM_PROMPT = "you are a good and an intellectual assistant"

# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Model",
        ["mistral-small-2506", "mistral-large-latest", "open-mistral-nemo"],
        index=0,
    )

    temperature = st.slider("Temperature", 0.0, 1.5, 0.9, 0.1)

    system_prompt = st.text_area(
        "System prompt", value=DEFAULT_SYSTEM_PROMPT, height=100
    )

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = [SystemMessage(content=system_prompt)]
        st.rerun()

    st.divider()
    st.caption("Make sure `MISTRAL_API_KEY` is set in your `.env` file.")

# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=system_prompt)]

# If the system prompt changed in the sidebar, update the stored SystemMessage
if isinstance(st.session_state.messages[0], SystemMessage):
    st.session_state.messages[0] = SystemMessage(content=system_prompt)

# ---------------- Cache the model so it isn't rebuilt every rerun ----------------
@st.cache_resource(show_spinner=False)
def get_model(model_name: str, temperature: float):
    return ChatMistralAI(model=model_name, temperature=temperature)

model = get_model(model_name, temperature)

# ---------------- Header ----------------
st.title("🤖 Mistral Chatbot")
st.caption("A simple Streamlit UI built on top of your LangChain + ChatMistralAI script.")

# ---------------- Render chat history ----------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    # SystemMessage is intentionally not rendered in the chat feed

# ---------------- Chat input ----------------
prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)
        st.markdown(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))
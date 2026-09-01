import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# Load environment variables (API Key)
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Persona & Tone Behavior Analyzer",
    page_icon="🎭",
    layout="centered"
)

# 1. Base Role Profiles
ROLE_PROMPTS = {
    "HACKATHON": "You are a hackathon mentor focusing on rapid prototyping, innovative architectures, fast MVP delivery, and quick debugging.",
    "ACADEMIC": "You are an academic mentor providing conceptual explanations, theoretical rigor, and structured pedagogical breakdowns.",
    "PLACEMENT INTERVIEW": "You are a technical placement interviewer focusing on DSA, system design concepts, and structured interview answers.",
    "PROFESSIONAL INTERACTION": "You are a corporate communication advisor assisting with formal, workplace-appropriate, and executive interactions."
}

# 2. Emotional Tone Overlays
TONE_PROMPTS = {
    "HAPPY": "Adopt an overwhelmingly cheerful, energetic, and optimistic tone. Use celebratory language, enthusiastic encouragement, and high positivity.",
    "SAD": "Adopt a deeply melancholic, gloomy, and pessimistic tone. Focus on the burden of effort, existential fatigue, and subdued sorrow."
}

# --- Sidebar Controls ---
st.sidebar.title("🎭 Behavior Controls")

selected_role = st.sidebar.selectbox(
    "1. Select Domain Mode:",
    options=list(ROLE_PROMPTS.keys()),
    index=0
)

selected_tone = st.sidebar.radio(
    "2. Select Emotional Tone:",
    options=["HAPPY", "SAD"],
    index=0,
    horizontal=True
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.5,
    value=0.9,
    step=0.05,
    help="Higher values emphasize stylistic traits and variability."
)

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# Dynamic Composite System Prompt
composite_system_prompt = f"{ROLE_PROMPTS[selected_role]}\n\nTone Directive: {TONE_PROMPTS[selected_tone]}"

# Track mode changes to manage state
current_config = f"{selected_role}_{selected_tone}"
if "active_config" not in st.session_state:
    st.session_state.active_config = current_config
    st.session_state.messages = []
elif st.session_state.active_config != current_config:
    st.session_state.active_config = current_config
    st.session_state.messages = []

# --- Initialize Model ---
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=temperature
)

# --- Main UI ---
st.title("🎭 Behavior & Tone Analyzer")
st.markdown(f"**Current Role:** `{selected_role}` | **Active Tone:** `{selected_tone}`")

# Expander to inspect the underlying prompt injected into Mistral
with st.expander("🔍 Inspect Active System Prompt"):
    st.code(composite_system_prompt, language="markdown")

# --- Display Chat History ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# --- Chat Input & Streaming ---
if user_prompt := st.chat_input("Enter a prompt to analyze behavior..."):
    # Display user query
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append(HumanMessage(content=user_prompt))

    # Form payload with the composite system prompt
    langchain_messages = [
        SystemMessage(content=composite_system_prompt)
    ] + st.session_state.messages

    # Stream response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk in model.stream(langchain_messages):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            
            st.session_state.messages.append(AIMessage(content=full_response))
        except Exception as e:
            st.error(f"Error communicating with Mistral AI: {str(e)}")
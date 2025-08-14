import streamlit as st
from inference import Chatbot
import json

DEFAULT_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"   # finetuned model
DEFAULT_ADAPTER_PATH = "/path/to/your/fine-tuned/adapter"

st.set_page_config(page_title="Medical Chatbot", layout="centered")
st.title("Medical Chatbot")
st.caption("Educational demo. Not a substitute for professional medical advice.")

# Sidebar controls
st.sidebar.header("Model Settings")
model_id = st.sidebar.text_input("Model ID or path", DEFAULT_MODEL_ID)
adapter_path = st.sidebar.text_input("LoRA Adapter Path (optional)", DEFAULT_ADAPTER_PATH)
max_new_tokens = st.sidebar.slider("Max new tokens", 64, 512, 200)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95)

# Initialize chatbot in session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = Chatbot(
        model_id=model_id,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )

if st.sidebar.button("Reload Model"):
    st.session_state.chatbot = Chatbot(
        model_id=model_id,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    st.success("Model reloaded.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Disclaimer
with st.expander("Disclaimer"):
    st.write("""
    This chatbot is for **educational purposes only**.
    It does **not** provide medical diagnoses.
    Always consult a qualified healthcare provider for concerns.
    """)

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
user_input = st.chat_input("Describe your symptoms...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    reply, meta = st.session_state.chatbot.respond(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)

    if meta.get("should_stop"):
        st.info("The assistant believes it has enough information for a preliminary triage.")

# Utilities
col1, col2 = st.columns(2)
with col1:
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.experimental_rerun()
with col2:
    st.download_button(
        "Download Chat (JSON)",
        data=json.dumps(st.session_state.messages, indent=2, ensure_ascii=False),
        file_name="chat_history.json",
        mime="application/json"
    )

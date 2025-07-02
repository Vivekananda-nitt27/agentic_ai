import streamlit as st
import requests

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("🤖 AI Chatbot Agents")
st.write("Create and interact with dynamic AI agents powered by LangGraph + Groq/OpenAI!")

# System prompt input
system_prompt = st.text_area(
    "🧠 Define your AI Agent's behavior:",
    height=70,
    placeholder="e.g. 'Act as a friendly assistant who explains concepts simply...'"
)

# Model selection
MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.radio("🛰️ Select Model Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select a Groq Model:", MODEL_NAMES_GROQ)
else:
    selected_model = st.selectbox("Select an OpenAI Model:", MODEL_NAMES_OPENAI)

# Optional tool usage
allow_web_search = st.checkbox("🌐 Enable Web Search (Tavily)")

# User input
user_query = st.text_area("💬 Ask your question:", height=150, placeholder="e.g. What are the trends in AI research?")

# Backend API URL
API_URL = "http://127.0.0.1:9999/chat"

# Submit
if st.button("🚀 Ask Agent!"):
    if not user_query.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Thinking..."):
            payload = {
                "model_name": selected_model,
                "model_provider": provider,
                "system_prompt": system_prompt,
                "messages": [user_query],
                "allow_search": allow_web_search
            }

            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    if "error" in data:
                        st.error(f"❌ Error: {data['error']}")
                    else:
                        st.subheader("🧠 Agent Response")
                        st.markdown(f"**{data['response']}**" if "response" in data else f"{data}")
                else:
                    st.error(f"❌ Server returned status {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"🚫 Connection error: {e}")

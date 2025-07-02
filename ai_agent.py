import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

# Load keys (if not using pipenv or .env file)
# from dotenv import load_dotenv
# load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

system_prompt = "Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    # Select LLM
    if provider.lower() == "groq":
        llm = ChatGroq(model=llm_id)
    elif provider.lower() == "openai":
        llm = ChatOpenAI(model=llm_id)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Select tools
    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    # Create agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
       
    )

    # Prepare input state
    state = {"messages": query}

    # Invoke agent
    response = agent.invoke(state)

    # Handle response safely
    if isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
    else:
        raise ValueError("Unexpected response format from agent")

    # Extract AI messages
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
    return ai_messages[-1] if ai_messages else "No AI response found."

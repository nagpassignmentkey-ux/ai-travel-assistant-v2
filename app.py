import os
import streamlit as st
import requests
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="AI Travel Planning Assistant (Singapore)", layout="wide")

st.title("🇸🇬 AI Travel Planning Assistant")
st.markdown("Your intelligent companion combining **Destination Knowledge (RAG)** with **Live Weather & Currency Tools (MCP)**.")

# Get Google Gemini API Key safely from user input or environment variable
api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# ==========================================
# 1. RAG: KNOWLEDGE BASE & VECTOR STORE
# ==========================================
@st.cache_resource
def initialize_vector_store():
    kb_path = "data/singapore_guide.md"
    if not os.path.exists(kb_path):
        os.makedirs("data", exist_ok=True)
        # Fallback dummy writer if file missing
        with open(kb_path, "w") as f:
            f.write("# Singapore Travel Guide\nDefault content: Marina Bay, Sentosa, Orchard Road.")

    loader = TextLoader(kb_path, encoding="utf-8")
    docs = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # Create Local Free Embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Store into Chroma vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

try:
    retriever = initialize_vector_store()
except Exception as e:
    st.error(f"Error initializing knowledge base: {e}")
    st.stop()

# ==========================================
# 2. MCP TOOLS (Weather & Currency - 100% Free)
# ==========================================
@tool
def get_weather_forecast(query: str = "Singapore") -> str:
    """Get the current weather forecast for Singapore to plan indoor or outdoor activities."""
    try:
        # Open-Meteo free API coordinates for Singapore (1.3521° N, 103.8198° E)
        url = "https://api.open-meteo.com/v1/forecast?latitude=1.3521&longitude=103.8198&daily=temperature_2m_max,precipitation_probability_max,weathercode&timezone=Asia/Singapore"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        rain_probs = daily.get("precipitation_probability_max", [])
        
        forecast_summary = "Live Weather Forecast for Singapore (Source: Open-Meteo MCP Tool):\n"
        for i in range(min(3, len(dates))):
            forecast_summary += f"- Date: {dates[i]} | Max Temp: {max_temps[i]}°C | Rain Probability: {rain_probs[i]}%\n"
        return forecast_summary
    except Exception as e:
        return f"Weather MCP Tool failed to fetch data: {str(e)}"

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str = "SGD") -> str:
    """Convert an amount from one currency to another using the free Frankfurter API."""
    try:
        url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency.upper()}&to={to_currency.upper()}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if "rates" in data:
            converted = data["rates"][to_currency.upper()]
            return f"Currency Conversion Result (Source: Frankfurter API Tool): {amount} {from_currency.upper()} = {converted} {to_currency.upper()}"
        else:
            return f"Conversion failed. Response: data"
    except Exception as e:
        return f"Currency MCP Tool failed: {str(e)}"

@tool
def search_knowledge_base(query: str) -> str:
    """Search the official Singapore travel guide knowledge base for destination facts, attractions, transport, culture, and sample itineraries."""
    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        return "I do not have enough information in the knowledge base to answer this question."
    
    result = "Knowledge Base Results (Source: Wikivoyage & Visit Singapore Official Guide):\n"
    for doc in relevant_docs:
        result += f"\n--- Content Snippet ---\n{doc.page_content}\n"
    return result

tools = [get_weather_forecast, convert_currency, search_knowledge_base]

# ==========================================
# 3. AGENT & PROMPT ORCHESTRATION
# ==========================================
if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("⚠️ Please input your Google Gemini API Key in the sidebar to begin.")
    st.stop()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)

# System Prompt Enforcement
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert AI Travel Planning Assistant for Singapore. "
     "You have access to a permanent destination knowledge base tool (`search_knowledge_base`) and live tools (`get_weather_forecast`, `convert_currency`).\n\n"
     "Rules:\n"
     "1. Use `search_knowledge_base` for static destination info (attractions, culture, transport, itineraries).\n"
     "2. Use `get_weather_forecast` ONLY when real-time or upcoming weather forecasts are requested.\n"
     "3. Use `convert_currency` ONLY when currency conversion is requested.\n"
     "4. If a request combines both (e.g. weather-aware itineraries or budget planning), query both tools and clearly distinguish destination facts, MCP live data, and your final recommendations.\n"
     "5. If information is unavailable in any source, state clearly: 'I do not have enough information to answer this.' Never invent facts."
),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==========================================
# 4. STREAMLIT USER INTERFACE CHAT
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display prior chat messages
for msg in st.session_state.chat_history:
    role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# User Input Box
user_input = st.chat_input("Ask about Singapore attractions, weather-adjusted itineraries, or budget conversions...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking and gathering insights from Knowledge Base & Live Tools..."):
            try:
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                # output_text = response["output"]
                # st.markdown(output_text)

                # --- FIX: Clean string extraction logic ---
                raw_output = response.get("output", "")
                
                if isinstance(raw_output, list):
                    # Handle list of text blocks
                    output_text = "".join([
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in raw_output
                    ])
                elif isinstance(raw_output, dict):
                    # Handle single dict response
                    output_text = raw_output.get("text", str(raw_output))
                else:
                    output_text = str(raw_output)

                st.markdown(output_text)
                
                # Update history
                from langchain_core.messages import HumanMessage, AIMessage
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=output_text))
                
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
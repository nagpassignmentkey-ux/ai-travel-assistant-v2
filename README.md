# 🇸🇬 AI Travel Planning Assistant (Singapore)

An intelligent, context-aware travel planning assistant that combines static destination knowledge (**Retrieval-Augmented Generation / RAG**) with live data from MCP.

---

## Architecture & System Overview

The system orchestrates two distinct information tiers: **stable knowledge-base facts** (RAG) and **dynamic live services** (MCP), coordinated via a stateful tool-calling agent execution loop.

                     ┌─────────────────────────────┐
                     │   Streamlit Web Interface   │
                     └──────────────┬──────────────┘
                                    │ User Inputs & Memory State
                                    ▼
                     ┌─────────────────────────────┐
                     │   LangChain Agent Executor  │
                     │    (Gemini 3.6 Flash )      │
                     └──────┬──────────────┬───────┘
                            │              │
        ┌───────────────────┘              └───────────────────┐
        │ Query Static Facts                                   │ Query Dynamic APIs
        ▼                                                      ▼
┌──────────────────────┐                              ┌──────────────────────┐
│  RAG Tool (ChromaDB) │                              │   MCP Tools Suite    │
└──────────┬───────────┘                              └──────────┬───────────┘
           │                                                     │
  ┌────────┴────────┐                                   ┌────────┴────────┐
  │ Vector Database │                                   │ Open-Meteo REST │ (Weather)
  │ (MiniLM-L6-v2)  │                                   │ Frankfurter FX  │ (Currency)
  └────────┬────────┘                                   └─────────────────┘
         │
┌──────────┴───────────┐
│ Knowledge Base Docs  │
│ (singapore_guide.md) │
└──────────────────────┘

---

## Knowledge-Base Sources

The destination knowledge base focuses on **Singapore** and is constructed from public travel guides:

1. **Wikivoyage Singapore Travel Guide**: Geography, transit networks, cultural districts, safety, and local rules.
2. **Visit Singapore Official Tourism Portal**: Highlights, seasonal events, and sample itineraries.
3. **Visit Singapore Essential Visitor Information**: Practical visitor guidelines, dining options, and indoor/outdoor attraction classifications.

All source content is compiled into structured markdown (`data/singapore_guide.md`).

---

## RAG Workflow

The RAG pipeline processes and serves static destination facts through the following steps:

1. **Document Loading**: `TextLoader` ingests structured Markdown travel documentation.
2. **Chunking Strategy**: `RecursiveCharacterTextSplitter` divides documents into chunks of **500 characters** with an **overlap of 50 characters** to preserve semantic continuity across headings.
3. **Embeddings**: Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace (384-dimensional vector representation).
4. **Vector Store**: Embeddings are indexed in an in-memory/persistent **ChromaDB** instance under the `singapore_travel_kb` collection.
5. **Retrieval**: Queries retrieve top-$k$ ($k=3$) matching chunks using cosine similarity search.
6. **Grounding & Attribution**: Retrieved chunks are injected into the agent scratchpad alongside explicit source labels (`[Source: Wikivoyage & Visit Singapore Official Guide]`).

---

## MCP Tools Architecture

Real-time information is accessed via Model Context Protocol (MCP) compatible tools designed using REST endpoint services:

### MCP Tool 1: Live Weather Forecast Service
* **Identifier**: `get_weather_forecast`
* **Underlying Provider**: Open-Meteo API (Latitude: `1.3521`, Longitude: `103.8198`).
* **Functionality**: Returns a 3-day daily forecast including maximum temperatures and maximum precipitation probabilities (`precipitation_probability_max`).
* **Use Case**: Informs weather-aware itinerary planning by detecting rain probabilities ($>50\%$).

### MCP Tool 2: Foreign Exchange Currency Converter
* **Identifier**: `convert_currency`
* **Underlying Provider**: Frankfurter Foreign Exchange API.
* **Functionality**: Performs dynamic conversions between major world currencies (e.g., `INR`, `USD`, `EUR`) and Singapore Dollars (`SGD`).
* **Use Case**: Handles budget estimation queries and financial context setup.

---

## 🚀 Setup & Execution Instructions

### Prerequisites
* Python 3.10+
* A Google Gemini API Key
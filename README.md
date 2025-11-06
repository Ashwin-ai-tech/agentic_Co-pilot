Project Structure:
------------------
    agentic_app.py            # Flask App Entry Point
    agentic_backend.py        # Orchestrator, KB Init, Session Mgmt, Pipeline Cache
    utils.py                  # LLM Call, Embedding, Reranking Utilities
    core_rag.py               # Core RAG Logic: Retrieval, Grading, Response Gen
    servicenow_integration.py # ServiceNow Client Logic & Initialization
    analytics_manager.py      # Analytics Tracking and Reporting
    database.py               # Database Connection and Initialization
    build_kb.py               # *** NEW: Offline KB Indexing Script ***
    agents/                   # Directory for Agent Logic
        __init__.py           # Makes 'agents' a Python package
        rag_specialist.py     # RAG Agent Logic (uses core_rag) - *UPDATED*
        servicenow_agent.py   # ServiceNow Tool Agent Logic
        clarification_agent.py# Clarification Logic
        simple_response_agent.py # Simple Greeting/Chitchat Logic
        simple_agent.py        #To generate simple response
        specialist_agent.py    #To generate code based responses
    demo-kb/                  # Knowledge Base JSON Files
    static/                   # Frontend HTML/CSS/JS Files
    analytics/                # Analytics Dashboard HTML/CSS/JS Files
    .env                      # Environment Variables (API Keys, etc.)
    requirements.txt          # Project Dependencies
    vector_store.npy          # *** NEW: Pre-computed Embeddings ***
    kb_data.json              # *** NEW: Pre-computed KB Chunks+Metadata ***
    ... etc.

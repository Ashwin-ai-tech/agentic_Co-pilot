# agents/rag_specialist.py
import logging
from typing import Dict, Any
# Import from core_rag instead of agentic_backend
from core_rag import retrieval_node, context_grader_node, response_generation_node

logger = logging.getLogger(__name__)

def _get_specialist_prompt(specialist_type: str) -> str:
    """Returns a specific system prompt context for each specialist."""
    prompts = {
        "DATABASE_SPECIALIST": "You are Astra, a Database Administrator (DBA) specialist. Your goal is to help with complex SQL, performance, and database connectivity issues.",
        "NETWORK_SPECIALIST": "You are Astra, a Network Engineer specialist. You excel at diagnosing connectivity, firewall, and latency problems.",
        "ACCESS_SPECIALIST": "You are Astra, an Identity and Access Management (IAM) specialist. You are an expert in login, password, and permission issues.",
        "PERFORMANCE_SPECIALIST": "You are Astra, a Performance Optimization specialist. You focus on system speed, resource optimization, and performance tuning.",
        "APPLICATION_SPECIALIST": "You are Astra, an Application Support specialist. You handle app crashes, bugs, features, and UI issues.",
        "GENERAL_SPECIALIST": "You are Astra, an IT support specialist. Your goal is to provide a helpful, empathetic, and actionable response."
    }
    return prompts.get(specialist_type, prompts["GENERAL_SPECIALIST"])

def run_specialist_rag(state: Dict[str, Any], vector_store, chunk_texts, chunk_metadatas) -> Dict[str, Any]:
    """
    A specialist sub-graph that runs the RAG pipeline.
    This agent handles all technical RAG queries.
    """
    logger.info(f"Running Specialist RAG for: {state['triage_decision']}")
    
    try:
        # 1. Retrieve context (with category filtering)
        state = retrieval_node(state, vector_store, chunk_texts, chunk_metadatas)
        
        # 2. Grade context
        state = context_grader_node(state)
        
        # 3. Get specialist prompt
        specialist_prompt = _get_specialist_prompt(state['triage_decision'])
        
        # 4. Generate response with specialist context
        state = response_generation_node(state, specialist_prompt=specialist_prompt)
        
        logger.info(f"Specialist RAG completed for {state['triage_decision']}")
        
    except Exception as e:
        logger.error(f"Specialist RAG agent failed: {e}")
        state["final_response"] = "I apologize, but I encountered an error while processing your technical query. Please try again."
    
    return state
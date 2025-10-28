# core_rag.py - Core RAG pipeline functions
import numpy as np
import logging
from typing import List, Dict, Optional, Any

# Import from utils
from utils import call_llm, get_cached_query_embedding, manual_cohere_rerank

logger = logging.getLogger(__name__)

# Define constants
TOP_K = 5

def retrieval_node(
    state: Dict[str, Any],
    vector_store: np.ndarray,
    kb_data: List[Dict]  # <-- KEY CHANGE 1: Combined data object
) -> Dict[str, Any]:
    """Retrieves context with cached query embeddings and rerank fallback."""
    query = state["query"]
    triage_decision = state["triage_decision"]

    # --- KEY CHANGE 2: Use new utils function output ---
    query_vector_list = get_cached_query_embedding(query.lower().strip())

    if not query_vector_list or not vector_store.size:
        logger.warning("Retrieval node: Invalid query embeddings or empty vector store.")
        state["retrieved_context"] = []
        return state

    query_vector = np.array(query_vector_list, dtype="float32") # No [0] needed
    # --------------------------------------------------

    # Normalize query; assume vector_store is pre-normalized
    try:
        # --- KEY CHANGE 3: Remove double normalization ---
        norm_query = query_vector / np.linalg.norm(query_vector)
        # We assume 'vector_store' is already normalized by build_kb.py
        similarities = np.dot(vector_store, norm_query)
        # ------------------------------------------------
        
        top_indices_initial = np.argsort(similarities)[-25:][::-1]

    except Exception as e:
        logger.error(f"Error during vector similarity calculation: {e}")
        state["retrieved_context"] = []
        return state

    # --- KEY CHANGE 4: Refactor all logic to use 'kb_data' ---
    top_indices = top_indices_initial
    if triage_decision.endswith('_SPECIALIST') and triage_decision != "GENERAL_SPECIALIST":
        target_category = triage_decision.replace('_SPECIALIST', '').title()
        filtered_indices = [
            idx for idx in top_indices_initial 
            # Use the new kb_data object for filtering
            if kb_data[idx].get('category', 'General').lower().startswith(target_category.lower())
        ]
        if filtered_indices: 
            logger.info(f"Filtered for category: {target_category}")
            top_indices = filtered_indices
        else:
            logger.warning(f"No matches for category {target_category}, using general results.")
    
    # Get the top candidate items from kb_data
    candidate_items = [kb_data[i] for i in top_indices[:20]]
    
    # Create primed docs for reranking
    primed_docs = [
        f"[Category: {item.get('category', 'General')}] [Title: {item.get('title', 'Untitled')}] {item['text_chunk']}"
        for item in candidate_items
    ]
    
    # Rerank
    rerank_results = manual_cohere_rerank(query, primed_docs, TOP_K)
    
    retrieved_context = []
    
    if rerank_results:
        logger.info("Reranking successful. Using reranked results.")
        for result in rerank_results:
            original_idx_pos = result['index']
            # Get the full original item from our candidate list
            original_item = candidate_items[original_idx_pos] 
            retrieved_context.append({
                "content": original_item['text_chunk'],
                "score": result['relevance_score'],
                "metadata": original_item # Pass the *entire* dict as metadata
            })
    else:
        logger.warning("Reranking failed or returned empty. Falling back to top vector search results.")
        for i in range(min(TOP_K, len(top_indices))):
            idx = top_indices[i]
            item = kb_data[idx] # Get the full item from kb_data
            retrieved_context.append({
                "content": item['text_chunk'],
                "score": similarities[idx],
                "metadata": item # Pass the *entire* dict as metadata
            })
    # ----------------------------------------------------------
    
    state["retrieved_context"] = retrieved_context
    return state

def context_grader_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Grades if the retrieved context is relevant to the query."""
    query = state["query"]
    retrieved_context = state["retrieved_context"]
    
    if not retrieved_context or retrieved_context[0]['score'] < 0.3:
        state["context_is_relevant"] = False
        return state

    context_str = "\n\n".join([ctx['content'] for ctx in retrieved_context[:2]])
    
    prompt = f"""You are a relevance-grading AI. Your task is to determine if the provided knowledge base context is **directly and specifically** relevant to answering the user's query.

Respond with only "Yes" or "No".

USER QUERY: "{query}"

KNOWLEDGE BASE CONTEXT:
---
{context_str}
---

CRITICAL RULE: If the user asks for a *user password reset* (e.g., "I forgot my password") and the context is about a *database password* or *database login credentials* (e.g., "Login failed for user"), the context is **NOT** relevant.

Is the context relevant? (Yes/No):
"""
    
    try:
        grade = call_llm(
            prompt,
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=10
        ).strip().lower()
        
        state["context_is_relevant"] = "yes" in grade
        logger.info(f"Context Grader: {'Relevant' if state['context_is_relevant'] else 'Not Relevant'}")
            
    except Exception as e:
        logger.error(f"Context grader failed: {e}")
        state["context_is_relevant"] = False
        
    return state

def response_generation_node(state: Dict[str, Any], specialist_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Generates final response based on context, intent, or a specialist prompt."""
    query = state["query"]
    retrieved_context = state["retrieved_context"]
    context_is_relevant = state.get("context_is_relevant", False)
    
    if state.get("specialist_response"):
        state["final_response"] = state["specialist_response"]
        return state

    if context_is_relevant:
        context_str = "\n\n".join([
            # This logic now works perfectly as 'metadata' is the full item dict
            f"Reference {i+1} (Source: {ctx['metadata'].get('source', 'N/A')} - {ctx['metadata'].get('section', 'N/A')}):\n{ctx['content']}"
            for i, ctx in enumerate(retrieved_context[:3])
        ])
        
        if specialist_prompt:
            base_prompt = specialist_prompt
        else:
            base_prompt = "You are Astra, an IT support specialist. Your goal is to provide a helpful, empathetic, and actionable response."
        
        prompt = f"""{base_prompt}
        
You have been given the following verified, relevant knowledge:
KNOWLEDGE:
{context_str}

USER QUERY: {query}

INSTRUCTIONS:
- Answer the user's query **using only the KNOWLEDGE provided**.
- **DO NOT** say "Reference 1" or "based on the knowledge". Just *use* the knowledge.
- Provide clear, step-by-step guidance.
- End with an offer for further assistance.

Response:
"""
        response = call_llm(prompt)
    
    else:
        prompt = f"""You are Astra, an IT support specialist. The user has asked: "{query}"

Since I don't have specific documentation for this issue, please provide general troubleshooting guidance:

1. Start with empathy (e.g., "I'm sorry to hear you're having trouble...").
2. Suggest 2-3 basic diagnostic steps (e.g., "Have you tried restarting the application?", "Can you check your internet connection?").
3. Recommend escalating to human support or logging a ticket if the issue persists.

Response:"""
        response = call_llm(prompt)
        
    state["final_response"] = response
    return state

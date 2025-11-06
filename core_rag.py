# core_rag.py - Core RAG pipeline functions
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from utils import _parse_ticket_details_from_history
import json

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

def _check_user_frustration(query: str, history: List[Dict]) -> bool:
    """
    Uses a quick LLM call to check if the user is stuck or frustrated.
    """
    if not history:
        return False # Not frustrated on the first turn

    # Combine the user's new query with the bot's last response
    last_bot_answer = history[-1].get("answer", "")
    
    # --- THIS IS THE NEW, SMARTER LOGIC ---
    try:
        prompt = f"""
        Analyze the following interaction. The bot just gave a solution, and the user replied.
        Does the user's reply express frustration, confusion, or that the solution failed?
        Respond with ONLY one word: "Yes" or "No".

        Bot's last message: "{last_bot_answer[-200:]}..." 
        User's new reply: "{query}"

        Analysis (Yes/No):
        """
        response = call_llm(prompt, model="llama-3.1-8b-instant", temperature=0, max_tokens=5).strip().lower()
        
        is_frustrated = "yes" in response
        logger.info(f"Frustration check: Bot said '...{last_bot_answer[-50:]}', User said '{query}'. Analysis: {response}")
        return is_frustrated
        
    except Exception as e:
        logger.error(f"Frustration check LLM call failed: {e}")
        # Fallback to simple keyword check on error
        keywords = ["no", "didn't work", "still broken", "stuck", "useless", "exhausted", "not solved"]
        return any(keyword in query.lower() for keyword in keywords)
    # --- END NEW LOGIC ---

def response_generation_node(state: Dict[str, Any], specialist_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates final response based on context, intent, or a specialist prompt.
    NOW includes conversational history and frustration-based ticketing.
    """
    query = state["query"]
    retrieved_context = state["retrieved_context"]
    context_is_relevant = state.get("context_is_relevant", False)
    
    # --- NEW: Get conversation history ---
    history = state.get('conversation_history', [])
    history_str = "\n".join([
        f"User: {turn.get('query', '')}\nBot: {turn.get('answer', '')}" 
        for turn in history[-3:] # Get last 3 turns
    ])
    # -------------------------------------
    
    if "metadata" not in state:
        state["metadata"] = {}
    state["metadata"]["show_ticket_offer"] = False
    state["metadata"]["pending_ticket_details_json"] = None
    
    if state.get("specialist_response"):
        state["final_response"] = state["specialist_response"]
        return state

    if context_is_relevant:
        context_str = "\n\n".join([
            f"Reference {i+1} (Source: {ctx['metadata'].get('source', 'N/A')} - {ctx['metadata'].get('section', 'N/A')}):\n{ctx['content']}"
            for i, ctx in enumerate(retrieved_context[:3])
        ])
        
        base_prompt = specialist_prompt or "You are Astra, an IT support specialist."
        
        # --- NEW CONTEXT-AWARE PROMPT ---
        prompt = f"""{base_prompt}
        
You are in a conversation with a user. Here is the recent history:
--- CONVERSATION HISTORY ---
{history_str}
--- END HISTORY ---

Based on the history, the user's *newest query* is: "{query}"

You have found the following relevant knowledge to answer the *newest query*:
--- KNOWLEDGE ---
{context_str}
--- END KNOWLEDGE ---

INSTRUCTIONS:
- Use the CONVERSATION HISTORY to understand the user's follow-up questions.
- Use the KNOWLEDGE to provide a direct, helpful answer to the *newest query*.
- **If the user is asking a follow-up about a previous step (e.g., "what is a hard reset?"), use the KNOWLEDGE to explain it.**
- Format your entire response using Markdown.
- **Use *only* the KNOWLEDGE provided.** Do not say "based on the knowledge".
- Conclude by asking if further assistance is needed or if they have more questions.

Response (in Markdown):
"""
        # --- END OF NEW PROMPT ---
        
        response = call_llm(prompt)
    
    else:
        # --- THIS IS YOUR NEW "SMART TICKET" LOGIC ---
        logger.warning(f"No relevant context found for query: {query}")
        
        # 1. Check for user frustration
        is_frustrated = _check_user_frustration(query, history)
        
        if is_frustrated:
            # 2. If frustrated, offer a stateful ticket
            logger.info("User seems frustrated. GENERATING STATEFUL TICKET OFFER.")
            try:
                ticket_details = _parse_ticket_details_from_history(history, query)
                state["metadata"]["show_ticket_offer"] = True
                state["metadata"]["pending_ticket_details_json"] = json.dumps(ticket_details)
                
                response = (
                    f"I'm sorry that the previous steps didn't help and I couldn't find a specific solution for: "
                    f"'{ticket_details.get('short_description', 'your issue')}'."
                    f"\n\nI understand this can be frustrating. **Would you like me to open a ticket** for you with these details?"
                )
            except Exception as e:
                logger.error(f"Error during RAG fallback ticket offer: {e}")
                response = "I'm sorry, I couldn't find a solution and I also ran into an error trying to offer a ticket. Please try rephrasing your issue."
        
        else:
            # 3. If NOT frustrated (first failure), just give general advice
            #    and DO NOT offer a ticket.
            logger.info("User is not frustrated. Generating general fallback.")
            
            prompt = f"""You are Astra, an IT support specialist. The user has asked: "{query}"
            
You don't have a specific document for this. 
Provide a general, helpful response.

INSTRUCTIONS:
- **Format your entire response using Markdown.**
- Address the issue with empathy.
- Provide a heading like `### General Troubleshooting`.
- Suggest 2-3 basic diagnostic steps as a **numbered list** (`1.`, `2.`, `3.`).
- **DO NOT offer to create a ticket.**
- Instead, conclude by asking the user: "Please let me know if these steps work, or if you're still stuck."

Response (in Markdown):
"""
            response = call_llm(prompt)
            # We do NOT set show_ticket_offer = True here.
            
    # --- END OF NEW LOGIC ---
        
    state["final_response"] = response
    return state

# utils.py - Shared utility functions
import os
import time
import logging
import requests
import ssl
import re
import json
import logging
from typing import List, Dict, Any
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# Renamed for clarity
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# --- SSL Configuration (As per user constraint) ---
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    logging.info("SSL default context overridden for unverified requests.")
except AttributeError:
    logging.warning("Could not override SSL default context (might not be needed).")

logger = logging.getLogger(__name__)

# --- LLM Call Utility ---
def call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.1, max_tokens: int = 800) -> str:
    """Enhanced LLM caller with better error handling."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not found.")
        return "LLM service key is missing."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        if "rate limit" in str(e).lower():
            return "The LLM service is currently experiencing high load. Please try again shortly."
        return "I apologize, but I'm having trouble connecting to the language model right now."

# --- Embedding Utilities ---
def manual_cohere_embed(texts: List[str], input_type: str) -> List[List[float]]:
    """Manual Cohere embedding with batching and error handling."""
    if not COHERE_API_KEY:
        logger.error("COHERE_API_KEY not found.")
        return []
    if not texts:
        return []

    all_embeddings = []
    BATCH_SIZE = 96

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        logger.info(f"Embedding batch {i//BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        payload = {
            "texts": batch,
            "model": "embed-english-v3.0",
            "input_type": input_type,
            "truncate": "END"
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {COHERE_API_KEY}"
        }

        retries = 3
        backoff_factor = 2
        for attempt in range(retries):
            try:
                response = requests.post("https://api.cohere.com/v1/embed",
                                         json=payload, headers=headers,
                                         verify=False, timeout=45) # verify=False as per user constraint

                if response.status_code == 429:
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                
                result = response.json()
                all_embeddings.extend(result.get("embeddings", []))
                time.sleep(1) # Small sleep to respect rate limits
                break

            except Exception as e:
                logger.error(f"Embedding batch failed on attempt {attempt+1}: {e}")
                if attempt + 1 == retries:
                    logger.error("Max retries reached. Failing this batch.")
                    # --- KEY CHANGE ---
                    # Raise an exception to stop build_kb.py
                    raise Exception("Embedding batch failed after max retries.")
                    # ------------------
                time.sleep(backoff_factor ** attempt)
    
    return all_embeddings

@lru_cache(maxsize=2048)
def get_cached_query_embedding(query: str) -> List[float]: # <-- Return type changed
    """
    Cached wrapper for embedding single search queries.
    Returns a single embedding vector.
    """
    logger.info(f"Embedding query (Cache MISS): {query[:50]}...")
    
    # --- KEY CHANGE ---
    # manual_cohere_embed returns List[List[float]], we want the first item
    embeddings = manual_cohere_embed([query], "search_query")
    
    if not embeddings:
        logger.error(f"Failed to embed query: {query}")
        return []
        
    return embeddings[0] # Return the first (and only) embedding
    # ------------------

# --- Reranking Utility ---
def manual_cohere_rerank(query: str, documents: List[str], top_n: int) -> List[Dict]:
    """Manual Cohere reranking with error handling."""
    if not COHERE_API_KEY:
        logger.error("COHERE_API_KEY not found.")
        return []
    if not documents:
        return []
        
    try:
        url = "https://api.cohere.com/v1/rerank"
        headers = {
            "accept": "application/json",
            "content-type": "application/json", 
            "Authorization": f"Bearer {COHERE_API_KEY}"
        }
        payload = {
            "model": "rerank-english-v3.0",
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents))
        }
        
        response = requests.post(url, json=payload, headers=headers, verify=False, timeout=30) # verify=False
        response.raise_for_status()
        return response.json().get('results', [])
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return []
    
def _parse_ticket_details_from_history(history: list, query: str) -> Dict[str, str]:
    """
    Uses an LLM to extract ticket details from the conversation.
    """
    logger.info("Parsing ticket details from conversation history...")
    
    # Create a concise history string
    history_str = "\n".join([f"User: {turn['query']}\nBot: {turn['answer']}" for turn in history[-3:]])
    history_str += f"\nUser: {query}" # Add the latest query
    
    prompt = f"""
    Analyze the following IT support conversation and extract the necessary details to create a ServiceNow incident.
    1.  **short_description**: A concise, 8-10 word summary of the user's problem.
    2.  **long_description**: A detailed description of the problem, including any error messages or specific symptoms mentioned.
    3.  **urgency**: Classify the urgency as '1' (High), '2' (Medium), or '3' (Low).
    
    Conversation:
    "{history_str}"
    
    Respond with ONLY a valid JSON object.
    
    Example:
    {{
      "short_description": "User cannot access shared drive 'Finance'",
      "long_description": "User reports getting an 'Access Denied' error when trying to open the //Server/Finance shared drive. This started today.",
      "urgency": "2"
    }}
    
    JSON:
    """
    
    try:
        response_text = call_llm(
            prompt,
            model="llama-3.1-8b-instant", # Use a fast model for extraction
            temperature=0.0,
            max_tokens=300
        )
        
        # Clean the response to get only the JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            details_dict = json.loads(json_match.group(0))
            # Ensure all keys are present
            details_dict.setdefault('short_description', query[:50] + "...")
            details_dict.setdefault('long_description', query)
            details_dict.setdefault('urgency', '3')
            return details_dict
        else:
            raise ValueError("LLM did not return valid JSON.")
            
    except Exception as e:
        logger.error(f"Failed to parse ticket details with LLM: {e}")
        # Fallback to simple details
        return {
            "short_description": f"Issue reported via Chatbot: {query[:50]}...",
            "long_description": query,
            "urgency": "3"
        }

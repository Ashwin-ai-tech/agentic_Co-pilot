# utils.py - Shared utility functions
import os
import time
import logging
import requests
import ssl
from typing import List, Dict, Any
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "llama-3.1-8b-instant")

# --- SSL Configuration ---
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
                                         verify=False, timeout=45)

                if response.status_code == 429:
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                
                result = response.json()
                all_embeddings.extend(result.get("embeddings", []))
                time.sleep(1)
                break

            except Exception as e:
                logger.error(f"Embedding batch failed on attempt {attempt+1}: {e}")
                if attempt + 1 == retries:
                    logger.error("Max retries reached. Failing this batch.")
                    return []
                time.sleep(backoff_factor ** attempt)
    
    return all_embeddings

@lru_cache(maxsize=2048)
def get_cached_query_embedding(query: str) -> List[List[float]]:
    """
    Cached wrapper for embedding single search queries.
    """
    logger.info(f"Embedding query (Cache MISS): {query[:50]}...")
    return manual_cohere_embed([query], "search_query")

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
        
        response = requests.post(url, json=payload, headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        return response.json().get('results', [])
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return []

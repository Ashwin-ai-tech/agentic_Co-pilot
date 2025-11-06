# In agents/specialist_agents.py

import logging
from typing import Dict, Any
from utils import call_llm  # Import the LLM utility

logger = logging.getLogger(__name__)

def run_code_generator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    A specialist agent that generates code snippets.
    """
    query = state["query"]
    history = state.get('conversation_history', [])
    history_str = "\n".join([
        f"User: {turn.get('query', '')}\nBot: {turn.get('answer', '')}" 
        for turn in history[-3:]
    ])

    # This prompt is highly specialized for code generation
    prompt = f"""
    You are Astra, a specialist Code Generation agent. Your role is to help users by writing safe, clear, and helpful code snippets.
    
    CONVERSATION HISTORY:
    {history_str}
    
    USER'S QUERY:
    "{query}"
    
    INSTRUCTIONS:
    1.  Analyze the user's query.
    2.  Generate the requested code snippet (e.g., Python, PowerShell, Bash, etc.).
    3.  **Use Markdown code blocks** for all code (e.g., ```python ... ```).
    4.  **Always provide a clear, step-by-step explanation** of what the code does.
    5.  **Add comments** inside the code to explain complex parts.
    6.  **Use placeholders** for sensitive information like passwords, API keys, or specific server names (e.g., `<YOUR_SERVER_NAME>`).
    7.  **Include a safety warning** if the code performs destructive actions (like deleting files or modifying registries).
    8.  Respond *only* with the code, explanation, and warnings. Do not add conversational fluff.
    
    Response (in Markdown):
    """
    
    try:
        response = call_llm(prompt, model="llama-3.1-8b-instant", temperature=0.1)
        state["final_response"] = response
    except Exception as e:
        logger.error(f"Code Generator Agent failed: {e}")
        state["final_response"] = "I'm sorry, I encountered an error while trying to generate the code for you."
        
    return state


def run_error_lookup_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    A specialist agent that looks up error codes and provides solutions.
    """
    query = state["query"]
    
    # This prompt is specialized for error lookup
    prompt = f"""
    You are Astra, a specialist Diagnostics agent. Your role is to identify error codes in a user's query and provide actionable solutions.
    
    USER'S QUERY:
    "{query}"
    
    INSTRUCTIONS:
    1.  **Identify the primary error code** or message (e.g., "0x80070005", "404 Not Found", "Connection Timed Out").
    2.  **Explain what this error code means** in simple terms.
    3.  Provide a heading `### Recommended Solutions`.
    4.  Give the user a **numbered list of 2-3 potential, actionable steps** to fix the problem.
    5.  Start with the simplest/most common solution first.
    6.  If you cannot find the error, simply state that you don't have information on that specific code.
    
    Response (in Markdown):
    """
    
    try:
        response = call_llm(prompt, model="llama-3.1-8b-instant", temperature=0.0)
        state["final_response"] = response
    except Exception as e:
        logger.error(f"Error Lookup Agent failed: {e}")
        state["final_response"] = "I'm sorry, I encountered an error while trying to look up that error code."
        
    return state

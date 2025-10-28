# agents/simple_response_agent.py
import logging
from typing import Dict, Any
# Import from utils instead of agentic_backend
from utils import call_llm

logger = logging.getLogger(__name__)

def run_simple_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles simple non-technical responses like greetings, gratitude, chitchat, and farewells.
    """
    triage_decision = state["triage_decision"]
    query = state["query"]

    logger.info(f"Simple Response Agent handling: {triage_decision}")

    response = "" # Initialize response variable

    try:
        if triage_decision == "GREETING":
            response = "Hello! I'm Astra, your IT support co-pilot. I'm here to help you with technical issues, troubleshooting, and IT guidance. How can I assist you today?"

        elif triage_decision == "GRATITUDE":
            response = "You're welcome! I'm glad I could help. If you have any other technical questions or run into more issues, don't hesitate to ask. Is there anything else you need assistance with?"

        elif triage_decision == "FAREWELL": # Ensure this condition is met
            response = "Goodbye! If you need further assistance, feel free to start a new chat."

        elif triage_decision == "CHITCHAT":
            # Use LLM for more natural conversation while staying on topic
            chat_prompt = f"""You are Astra, an IT support assistant. The user is making casual conversation: "{query}"

Keep your response brief, friendly, and professional. Gently steer the conversation back to IT support topics. Respond in 1-2 sentences.

Response:"""
            response = call_llm(chat_prompt, temperature=0.3, max_tokens=100)

        # --- IMPORTANT: Check if response was set ---
        # If the triage_decision didn't match any above (or if CHITCHAT failed), assign the fallback.
        # This 'else' should ideally only catch unexpected simple intents, NOT FAREWELL.
        if not response:
             logger.warning(f"Simple Response Agent: No specific response generated for decision '{triage_decision}'. Using fallback.")
             response = "I'm here to help with your IT support needs. How can I assist you with technical issues today?"
        # --------------------------------------------

        # Set the final response in the state
        state["final_response"] = response
        logger.info(f"Simple Response Agent: Handled {triage_decision} successfully. Response: '{response[:50]}...'")

    except Exception as e:
        logger.error(f"Simple Response Agent failed: {e}", exc_info=True)
        # Ensure a safe fallback even if errors occur
        state["final_response"] = "Hello! I'm here to help with IT support. How can I assist you today?"

    return state
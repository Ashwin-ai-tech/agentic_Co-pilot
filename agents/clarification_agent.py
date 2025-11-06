# agents/clarification_agent.py
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def run_clarification(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles vague queries by requesting clarification.
    This agent specializes in identifying and resolving ambiguous user requests.
    """
    query = state["query"].lower()
    logger.info(f"Clarification Agent handling vague query: {query[:50]}...")
    
    # Enhanced clarification prompts for different types of vagueness
    clarification_prompts = {
        'it': "What exactly is 'it' referring to? Please specify the system, application, or component that's having issues.",
        'this': "Could you clarify what 'this' refers to? What specific function, feature, or system isn't working properly?",
        'that': "What specific 'that' are you referring to? Please provide more context about what you're trying to accomplish.",
        'broken': "Could you describe what exactly is 'broken'? What were you trying to do when you noticed the issue, and what error messages did you see?",
        'not working': "What specifically isn't working? Please describe the expected behavior versus what you're actually experiencing.",
        'problem': "What specific problem are you experiencing? Please describe the symptoms, any error messages, and when the issue started.",
        'issue': "Could you describe the exact issue you're facing? What were you trying to do when it occurred, and what happened instead?",
        'error': "What specific error message are you seeing? Please share the exact error text and what you were doing when it appeared.",
        'help': "What specific area do you need help with? Please describe your technical issue in detail, including any relevant system information.",
        'something': "What exactly isn't working properly? Please be specific about the system, application, or process you're having trouble with."
    }
    
    # Default prompt for general vagueness
    prompt = "Could you please provide more details about your technical issue? What specifically are you trying to accomplish or troubleshoot? Please include any error messages, system information, or steps you've already tried."
    
    # Find the most relevant clarification prompt
    for key, value in clarification_prompts.items():
        if key in query:
            prompt = value
            logger.info(f"Clarification Agent: Using specific prompt for '{key}'")
            break
    
    # Set clarification state
    state["needs_clarification"] = True
    state["clarification_prompt"] = prompt
    state["final_response"] = prompt
    
    logger.info("Clarification Agent completed - clarification requested")
    return state

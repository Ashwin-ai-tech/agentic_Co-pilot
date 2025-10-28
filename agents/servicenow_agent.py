# agents/servicenow_agent.py
import logging
import re
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Import ServiceNow client - will be passed from main backend
try:
    from servicenow_integration import servicenow_client
    SERVICENOW_ENABLED = True
except ImportError:
    SERVICENOW_ENABLED = False
    logger.warning("ServiceNow integration not available")

def run_servicenow_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    A specialist agent that uses the ServiceNow tool.
    Handles ticket creation and status checking.
    """
    query = state["query"]
    decision = state["triage_decision"]
    logger.info(f"ServiceNow Agent handling: {decision}")

    if not SERVICENOW_ENABLED:
        state["specialist_response"] = "I'm sorry, but my connection to the ticketing system is currently disabled. I cannot create or check tickets at this time."
        return state

    try:
        if decision == "CREATE_TICKET":
            # Use the client's built-in analysis with conversation history
            history = state['conversation_history']
            logger.info(f"ServiceNow Agent: Creating ticket with {len(history)} history items")
            
            # Call the same method app.py uses
            ticket_result = servicenow_client.analyze_and_create_incident(
                conversation_history=history,
                use_llm=True,
                short_description=None,  # Let ServiceNow client generate from conversation
                description=None,
                urgency=None,
                category=None
            )
            
            # Handle response structure
            if "error" in ticket_result or ticket_result.get("status") == "error":
                error_msg = ticket_result.get('error') or ticket_result.get('message', 'Unknown error')
                response = f"I tried to create a ticket, but an error occurred: {error_msg}"
            else:
                # Extract ticket data from various possible response structures
                ticket_data = ticket_result.get('ticket_data', ticket_result)
                if 'result' in ticket_data:
                    ticket_data = ticket_data['result']
                
                ticket_number = ticket_data.get('number', 'N/A')
                response = f"I've successfully created a ticket for you. Your incident number is **{ticket_number}**. A support agent will look into this shortly."
                
                logger.info(f"ServiceNow Agent: Ticket created successfully - {ticket_number}")

        elif decision == "CHECK_TICKET_STATUS":
            # Extract ticket number from query
            match = re.search(r"(inc|req|task)[0-9]+", query, re.IGNORECASE)
            if match:
                ticket_id = match.group(0).upper()
                logger.info(f"ServiceNow Agent: Checking status for ticket {ticket_id}")
                
                status_result = servicenow_client.get_servicenow_ticket(ticket_id)
                
                # Handle various response formats
                if "error" in status_result or status_result.get("status") == "error":
                    error_msg = status_result.get('error') or status_result.get('message', 'Unknown error')
                    response = f"I tried to check the status of {ticket_id}, but an error occurred: {error_msg}"
                
                elif "ticket_data" in status_result:
                    ticket_data = status_result["ticket_data"]
                    state_val = ticket_data.get('state', 'N/A')
                    short_desc = ticket_data.get('short_description', 'N/A')
                    response = f"The status of ticket **{ticket_id}** is: **{state_val}**\n\n*Description:* {short_desc}"
                
                elif status_result.get("result") and isinstance(status_result["result"], list) and len(status_result["result"]) > 0:
                    ticket_data = status_result["result"][0]
                    state_val = ticket_data.get('state', 'N/A')
                    short_desc = ticket_data.get('short_description', 'N/A')
                    response = f"The status of ticket **{ticket_id}** is: **{state_val}**\n\n*Description:* {short_desc}"
                
                else:
                    logger.warning(f"ServiceNow Agent: Ticket {ticket_id} not found")
                    response = f"I'm sorry, I couldn't find any details for ticket **{ticket_id}**. Please double-check the ticket number."
            else:
                response = "I can help you check a ticket status. Could you please provide the ticket number (e.g., INC12345, REQ67890) you'd like me to check?"

        else:
            response = "I'm not sure how to handle that ticketing request."

    except Exception as e:
        logger.error(f"ServiceNow Agent failed: {e}")
        response = "I'm sorry, I encountered an error while contacting the ticketing system. Please try again later."

    state["specialist_response"] = response
    logger.info("ServiceNow Agent completed processing")
    return state
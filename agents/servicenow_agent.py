# agents/servicenow_agent.py
import logging
import re
import json
from typing import Dict, Any

# NEW: Import the LLM utility to parse ticket details
from utils import call_llm
from utils import _parse_ticket_details_from_history

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
    Handles ticket creation (2-step) and status checking.
    """
    query = state["query"]
    decision = state["triage_decision"]
    logger.info(f"ServiceNow Agent handling: {decision}")

    if not SERVICENOW_ENABLED:
        state["specialist_response"] = "I'm sorry, but my connection to the ticketing system is currently disabled. I cannot create or check tickets at this time."
        return state

    try:
        # -----------------------------------------------------------------
        # STEP 1: PARSE & CONFIRM
        # This block is now for parsing and asking for confirmation.
        # -----------------------------------------------------------------
        if decision == "CREATE_TICKET":
            logger.info(f"ServiceNow Agent: Parsing ticket details for confirmation...")
            
            # 1. Parse details from history
            history = state['conversation_history']
            ticket_details = _parse_ticket_details_from_history(history, state['query'])
            
            # 2. Add pending details to state metadata
            state["metadata"]["pending_ticket_details_json"] = json.dumps(ticket_details)
            
            # 3. Set metadata flags for the frontend pop-up
            # This tells run_agentic_rag() to pass "show_ticket_offer" to the frontend
            state["metadata"]["show_ticket_offer"] = True
            
            # 4. Create the confirmation response
            response = f"I've prepared a ticket for you with the following details:\n\n* **Summary:** {ticket_details['short_description']}\n* **Urgency:** {ticket_details['urgency']}\n\nWould you like me to create this ticket?"
            
            logger.info(f"ServiceNow Agent: Offering to create ticket. Details saved to session.")

        # -----------------------------------------------------------------
        # STEP 2: EXECUTE
        # This is the NEW block that runs after the user confirms "yes".
        # -----------------------------------------------------------------
        elif decision == "EXECUTE_TICKET":
            logger.info(f"ServiceNow Agent: Executing ticket creation...")
            
            # 1. Get the details from the state (put there by the orchestrator)
            pending_details_json = state.get("ticket_details")
            
            if not pending_details_json:
                logger.warning("EXECUTE_TICKET decision, but no ticket_details in state.")
                response = "I'm sorry, I seem to have lost the details for the ticket. Could you please describe your issue again?"
                state["specialist_response"] = response
                return state

            # 2. Load details and call the client
            ticket_details = json.loads(pending_details_json)
            
            ticket_result = servicenow_client.analyze_and_create_incident(
                conversation_history=state['conversation_history'], # Pass history for context
                use_llm=False, # We already parsed, so no need for LLM
                short_description=ticket_details.get('short_description'),
                description=ticket_details.get('long_description'),
                urgency=ticket_details.get('urgency'),
                category=None # Or parse this in the helper
            )
            
            # 3. Handle the response (NEW ROBUST LOGIC)
            if "error" in ticket_result or ticket_result.get("status") == "error":
                error_msg = ticket_result.get('error') or ticket_result.get('message', 'Unknown error')

                # Add the hibernation check we discussed
                if "hibernating" in error_msg.lower() or "<html>" in error_msg.lower():
                    logger.warning("ServiceNow instance is hibernating.")
                    response = "I tried to create the ticket, but it seems the ServiceNow system is currently hibernating. Please try again in 5-10 minutes."
                else:
                    response = f"I tried to create a ticket, but an error occurred: {error_msg}"

            else:
                # --- NEW ROBUST PARSING ---
                ticket_number = None

                # --- CORRECTED ROBUST PARSING ---
                ticket_number = None

                # 1. Check for 'incident_number' at the top level (based on your new log)
                if 'incident_number' in ticket_result:
                    ticket_number = ticket_result.get('incident_number')

                # 2. Check the other common structures
                elif 'result' in ticket_result and isinstance(ticket_result['result'], dict):
                    ticket_number = ticket_result['result'].get('number') or ticket_result['result'].get('incident_number')

                elif 'ticket_data' in ticket_result and isinstance(ticket_result['ticket_data'], dict):
                    ticket_number = ticket_result['ticket_data'].get('number') or ticket_result['ticket_data'].get('incident_number')
                    if not ticket_number and 'result' in ticket_result['ticket_data'] and isinstance(ticket_result['ticket_data']['result'], dict):
                        ticket_number = ticket_result['ticket_data']['result'].get('number') or ticket_result['ticket_data']['result'].get('incident_number')

                elif 'number' in ticket_result:
                    ticket_number = ticket_result.get('number')
                # --- END CORRECTED PARSING ---
                

                # --- NEW VALIDATION ---
                if ticket_number and ticket_number != 'N/A':
                    response = f"I've successfully created a ticket for you. Your incident number is **{ticket_number}**. A support agent will look into this shortly."
                    logger.info(f"ServiceNow Agent: Ticket created successfully - {ticket_number}")
                else:
                    # The API call succeeded (no error) but didn't return a number
                    logger.error(f"ServiceNow Agent: Ticket creation 'succeeded' but no ticket number was found. Full response: {ticket_result}")
                    response = "I've sent the ticket request to ServiceNow, but I did not receive a confirmation number. Please check the ServiceNow portal directly."
                # --- END NEW VALIDATION ---


        # -----------------------------------------------------------------
        # CHECK_TICKET_STATUS (Unchanged)
        # This logic is perfect as-is.
        # -----------------------------------------------------------------
        elif decision == "CHECK_TICKET_STATUS":
            # Extract ticket number from query
            match = re.search(r"(inc|req|task)[0-9]+", query, re.IGNORECASE)
            if match:
                ticket_id = match.group(0).upper()
                logger.info(f"ServiceNow Agent: Checking status for ticket {ticket_id}")
                
                status_result = servicenow_client.get_servicenow_ticket(ticket_id)
                
                # ... (All your existing, good status-handling logic remains here) ...
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
        logger.error(f"ServiceNow Agent failed: {e}", exc_info=True)
        response = "I'm sorry, I encountered an error while contacting the ticketing system. Please try again later."

    state["specialist_response"] = response
    logger.info("ServiceNow Agent completed processing")
    return state

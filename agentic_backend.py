# agentic_backend.py - Refactored Orchestrator + Specialist Agent System
import os, glob, json, sqlite3, numpy as np, time, hashlib, re, uuid, logging
import ssl
# --- SSL Configuration (As per user constraint) ---
ssl._create_default_https_context = ssl._create_unverified_context
# -------------------------
import requests
from typing import List, Dict, Tuple, Any, Optional, Union, TypedDict, Callable
from functools import lru_cache
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dotenv import load_dotenv
from groq import Groq
import cohere
from queue import Queue
from database import get_db
from analytics_manager import AnalyticsManager

# Import utility functions
# We still need them for the RAG pipeline (rerank, query embed, llm)
from utils import call_llm, manual_cohere_embed, get_cached_query_embedding, manual_cohere_rerank

# --- Import the client object, matching app.py ---
try:
    from servicenow_integration import servicenow_client
    SERVICENOW_ENABLED = True
    logging.info("ServiceNow integration client loaded successfully.")
except ImportError:
    logging.warning("ServiceNow integration client not found. Ticketing features will be disabled.")
    SERVICENOW_ENABLED = False
    # Create a mock client class
    class MockServiceNowClient:
        def analyze_and_create_incident(self, **kwargs):
            logger.warning("MockServiceNowClient: analyze_and_create_incident called")
            return {"error": "ServiceNow integration is not configured.", "status": "error"}
        def get_servicenow_ticket(self, ticket_id: str):
            logger.warning(f"MockServiceNowClient: get_servicenow_ticket called for {ticket_id}")
            return {"error": "ServiceNow integration is not configured.", "status": "error"}
    servicenow_client = MockServiceNowClient()

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant") # Use updated var name
KB_GLOB = os.getenv("KB_GLOB", "./demo-kb/**/*.json")
TOP_K = 5
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
DB_PATH = os.getenv("QUERY_DB", "query_history.sqlite")

# Initialize analytics manager
analytics_manager = AnalyticsManager(DB_PATH)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# --- DATA STRUCTURES & STATE MANAGEMENT ---
# ==============================================================================

@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    conversation_history: List[Dict] = field(default_factory=list)
    pending_clarification: Optional[Dict] = None
    feedback_pending: bool = False
    session_title: str = "Agentic Chat"
    theme_preference: str = "system"

class AgentState(TypedDict):
    """State for agentic workflow"""
    query: str
    session_id: str
    user_id: str
    conversation_history: List[Dict]
    triage_decision: str
    retrieved_context: Optional[List[Dict]]
    context_is_relevant: bool
    final_response: str
    metadata: Dict
    needs_clarification: bool
    clarification_prompt: str
    is_technical_rag: bool
    is_tool_use: bool
    specialist_response: str

# ==============================================================================
# --- KNOWLEDGE BASE & VECTOR STORE (NEW ALIGNED VERSION) ---
# ==============================================================================

# --- KEY CHANGE 1: 'load_and_chunk_knowledge_base' function is DELETED ---
# (It now lives in build_kb.py)

# --- KEY CHANGE 2: Global variables are updated ---
vector_store = None
kb_data = [] # Combined data list
# ------------------------------------------------

# --- KEY CHANGE 3: 'initialize_vector_store' is REPLACED ---
def initialize_vector_store():
    """
    Loads the pre-built vector store and KB data from disk.
    This is run ONCE at startup.
    """
    global vector_store, kb_data
    
    VECTOR_STORE_PATH = "vector_store.npy"
    KB_DATA_PATH = "kb_data.json"
    
    try:
        logger.info("Agentic Backend: Loading pre-built vector store...")
        vector_store = np.load(VECTOR_STORE_PATH)
        
        logger.info(f"Agentic Backend: Loading KB data...")
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
            
        if not vector_store.size or not kb_data:
            raise FileNotFoundError("Vector store or KB data is empty.")
            
        logger.info(f"✅ Knowledge Base loaded: {vector_store.shape[0]} vectors and {len(kb_data)} data chunks.")
        
    except FileNotFoundError:
        logger.error("--- FATAL ERROR ---")
        logger.error(f"Could not find '{VECTOR_STORE_PATH}' or '{KB_DATA_PATH}'.")
        logger.error("Please run the 'build_kb.py' script first to generate these files.")
        # Set to empty to prevent crash, but RAG will fail
        vector_store = np.array([])
        kb_data = []
    except Exception as e:
        logger.error(f"Error loading KB artifacts: {e}")
        vector_store = np.array([])
        kb_data = []
# ----------------------------------------------------------

# ==============================================================================
# --- CORE AGENTIC WORKFLOW NODES ---
# ==============================================================================

def orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]: # Using Dict if AgentState isn't available
    """
    Classifies intent to route to a Specialist Agent (RAG), a Tool (ServiceNow),
    or a Simple Response Agent. Includes refined prompt for farewells.
    """
    query = state["query"].lower().strip()
    logger.info(f"Orchestrator received query: {query[:100]}...") # Log query start

    # --- REFINED PROMPT ---
    classification_prompt = f"""
    Analyze the user's query and classify its primary intent. Respond with ONLY one word from the specific list below. Prioritize simple responses if applicable.

    --- Simple Responses ---
    GREETING            # User says hello, hi
    GRATITUDE           # User says thanks, thank you
    FAREWELL            # User says bye, goodbye, see you, thanks bye, take care
    CHITCHAT            # Casual conversation, unrelated to IT issues
    NEEDS_CLARIFICATION # Vague query like "it's broken", "help" with no detail

    --- Specialist RAG Agents ---
    DATABASE_SPECIALIST     # Database, SQL, connection timeout, query performance
    NETWORK_SPECIALIST      # Network, connectivity, firewall, latency, wifi, VPN
    ACCESS_SPECIALIST       # Login, authentication, password reset, permissions, access denied
    PERFORMANCE_SPECIALIST  # Slow system, application slowness, optimization needs
    APPLICATION_SPECIALIST  # Specific app crash, bug, feature request, UI problem
    GENERAL_SPECIALIST      # Any other technical IT issue not covered above

    --- Tool-Using Agents ---
    CREATE_TICKET           # User explicitly asks to create/log/open/file a ticket/incident
    CHECK_TICKET_STATUS     # User asks for status/update on an existing ticket (e.g., "status INC123")

    Query: "{query}"

    CRITICAL: If the query is clearly a simple greeting, farewell, or thank you, classify it as such, NOT as a technical specialist. Examples: "bye" -> FAREWELL, "hello there" -> GREETING.

    Classification:"""
    # --- END REFINED PROMPT ---

    try:
        # Call the LLM for classification
        classification = call_llm(
            classification_prompt,
            model="llama-3.1-8b-instant", # Or your preferred classification model
            temperature=0.0, # Low temperature for deterministic classification
            max_tokens=25 # Should be enough for one word + buffer
        ).strip().upper().replace("_", "") # Clean up potential underscores or extra spaces

        # --- Ensure FAREWELL is in valid_intents ---
        valid_intents = [
            "GREETING", "GRATITUDE", "CHITCHAT", "NEEDS_CLARIFICATION", "FAREWELL",
            "DATABASE SPECIALIST", "NETWORK SPECIALIST", "ACCESS SPECIALIST", # Handle potential spaces if LLM adds them
            "PERFORMANCE SPECIALIST", "APPLICATION SPECIALIST", "GENERAL SPECIALIST",
            "DATABASE_SPECIALIST", "NETWORK_SPECIALIST", "ACCESS_SPECIALIST", # Keep underscore versions too
            "PERFORMANCE_SPECIALIST", "APPLICATION_SPECIALIST", "GENERAL_SPECIALIST",
            "CREATE TICKET", "CHECK TICKET STATUS",
            "CREATE_TICKET", "CHECK_TICKET_STATUS"
        ]
        # --- Normalize classification (remove space if present) ---
        normalized_classification = classification.replace(" ", "_")

        if normalized_classification in valid_intents:
            state["triage_decision"] = normalized_classification
            logger.info(f"Orchestrator classified intent as: {normalized_classification}")
        else:
            # Fallback if the LLM gives an unexpected response
            logger.warning(f"Orchestrator received invalid classification '{classification}'. Defaulting to GENERAL_SPECIALIST.")
            state["triage_decision"] = "GENERAL_SPECIALIST"

    except Exception as e:
        logger.error(f"Orchestrator LLM call failed: {e}", exc_info=True)
        # Default to general specialist on error
        state["triage_decision"] = "GENERAL_SPECIALIST"

    # --- Set Routing Flags ---
    # Based on the final triage_decision
    decision = state["triage_decision"]
    if decision.endswith("_SPECIALIST"):
        state["is_technical_rag"] = True
        state["is_tool_use"] = False
        state["needs_clarification"] = False # Ensure flags are reset
    elif decision in ["CREATE_TICKET", "CHECK_TICKET_STATUS"]:
        state["is_technical_rag"] = False
        state["is_tool_use"] = True
        state["needs_clarification"] = False
    elif decision == "NEEDS_CLARIFICATION":
         state["is_technical_rag"] = False
         state["is_tool_use"] = False
         state["needs_clarification"] = True # Set clarification flag
    else: # Covers GREETING, GRATITUDE, CHITCHAT, FAREWELL
        state["is_technical_rag"] = False
        state["is_tool_use"] = False
        state["needs_clarification"] = False

    logger.info(f"Orchestrator final decision: {decision} (is_rag={state['is_technical_rag']}, is_tool={state['is_tool_use']}, needs_clarify={state['needs_clarification']})")
    return state

# ==============================================================================
# --- NEW: CACHED PIPELINE EXECUTION (ALIGNED VERSION) ---
# ==============================================================================

@lru_cache(maxsize=1024)
def _execute_rag_pipeline(query: str, session_id: str, user_id: str, history_json: str) -> AgentState:
    """
    This new function wraps the *entire* agentic workflow.
    It's cached, so repeated queries get an instant response.
    """
    logger.info(f"--- (Cache MISS) Executing pipeline for: {query[:50]}... ---")
    start_time = time.time()
    
    try:
        history = json.loads(history_json)
    except:
        history = []
        
    # 1. Initialize state
    state: AgentState = {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "conversation_history": history,
        "triage_decision": "",
        "retrieved_context": None,
        "context_is_relevant": False,
        "final_response": "",
        "metadata": {},
        "needs_clarification": False,
        "clarification_prompt": "",
        "is_technical_rag": False,
        "is_tool_use": False,
        "specialist_response": ""
    }
    
    # 2. Run Orchestrator
    state = orchestrator_node(state)
    
    # 3. Route to appropriate MODULAR AGENT
    if state["is_technical_rag"]:
        # Import here to avoid circular imports
        from agents import run_specialist_rag
        
        # --- KEY CHANGE 4: Pass the new 'kb_data' object ---
        state = run_specialist_rag(state, vector_store, kb_data)
        # ---------------------------------------------------
        
    elif state["is_tool_use"]:
        # Import here to avoid circular imports
        from agents import run_servicenow_agent
        from core_rag import response_generation_node
        state = run_servicenow_agent(state)
        state = response_generation_node(state)
        
    elif state["triage_decision"] == "NEEDS_CLARIFICATION":
        # Import here to avoid circular imports
        from agents import run_clarification
        state = run_clarification(state)
    
    else:
        # Import here to avoid circular imports
        from agents import run_simple_response
        state = run_simple_response(state)

    end_time = time.time()
    state["metadata"]["pipeline_time_ms"] = (end_time - start_time) * 1000

    print(f"DEBUG: Final response before returning from pipeline: '{state.get('final_response', 'NOT SET')}' for decision '{state.get('triage_decision')}'")
    
    return state

# ==============================================================================
# --- SESSION MANAGEMENT (Compatible with app.py) ---
# ==============================================================================

# (All session, DB, and compatibility functions from line 286 to 552
#  ... remain unchanged ...
#  e.g., get_session, save_session, run_agentic_rag, query_rag,
#  load_session, get_all_sessions_from_db, init_db, etc.)

def get_session(session_id: str, user_id: str = "default") -> Session:
    """Get or create session (compatible with app.py)."""
    try:
        con = get_db()
        cur = con.cursor()
        cur.execute("""
            SELECT user_id, created_at, last_activity, conversation_history, 
                   pending_clarification, feedback_pending, session_title, theme_preference
            FROM sessions WHERE session_id = ? AND is_active = 1
        """, (session_id,))
        
        row = cur.fetchone()
        now = datetime.now()
        
        if row:
            user_id, created_at, last_activity, history_json, clarification_json, feedback_pending, title, theme = row
            return Session(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.fromisoformat(created_at),
                last_activity=now,
                conversation_history=json.loads(history_json or '[]'),
                pending_clarification=json.loads(clarification_json) if clarification_json else None,
                feedback_pending=bool(feedback_pending),
                session_title=title or "Agentic Chat",
                theme_preference=theme or "system"
            )
        else:
            return Session(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                last_activity=now,
                session_title="Agentic Chat"
            )
            
    except Exception as e:
        logger.error(f"Session error: {e}")
        return Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )

def save_session(session: Session):
    """Save session to database."""
    try:
        con = get_db()
        cur = con.cursor()
        
        history_json = json.dumps(session.conversation_history)
        clarification_json = json.dumps(session.pending_clarification) if session.pending_clarification else None
        
        cur.execute("""
            INSERT OR REPLACE INTO sessions 
            (session_id, user_id, created_at, last_activity, conversation_history, 
             pending_clarification, feedback_pending, session_title, theme_preference, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            session.session_id, session.user_id, session.created_at.isoformat(),
            session.last_activity.isoformat(), history_json, clarification_json,
            int(session.feedback_pending), session.session_title, session.theme_preference
        ))
        con.commit()
    except Exception as e:
        logger.error(f"Save session error: {e}")

# ==============================================================================
# --- MAIN AGENTIC RAG ENTRY POINT (UPDATED) ---
# ==============================================================================

def run_agentic_rag(query: str, session_id: Optional[str] = None, user_id: str = "default") -> Dict[str, Any]:
    """
    Main agentic RAG function - wrapper around the cached pipeline with session management.
    Includes updated confidence logic.
    """
    start_time = time.time() # Start timing the whole request

    if not session_id:
        # Generate a new session ID if none is provided
        session_id = str(uuid.uuid4())[:12]
        logger.info(f"New session started: {session_id}")

    # Load or create the session object using the backend's session management
    session = get_session(session_id, user_id)

    try:
        # Normalize query for better cache hits and consistency
        normalized_query = query.lower().strip()
        # Prepare recent history for the pipeline cache key (limit size if needed)
        history_json = json.dumps(session.conversation_history[-5:]) # Pass recent history

        # --- CACHED PIPELINE CALL ---
        # Calls the function containing orchestrator, agent routing, RAG, etc.
        state = _execute_rag_pipeline(normalized_query, session_id, user_id, history_json)
        # --- END CACHED PIPELINE CALL ---

        # Check if the result came from the cache or was newly computed
        total_time = time.time() - start_time
        # Check for 'pipeline_time_ms' which is ONLY added on a cache miss within _execute_rag_pipeline
        is_cache_hit = "pipeline_time_ms" not in state["metadata"]

        if is_cache_hit:
            logger.info(f"--- (Cache HIT) Served from cache: {query[:50]}... ---")
            # Create fresh metadata for this cache-hit run, using total request time
            state["metadata"] = {"pipeline_time_ms": total_time * 1000, "cache_hit": True}
        else:
            # If it was a miss, 'pipeline_time_ms' already exists from _execute_rag_pipeline
            state["metadata"]["cache_hit"] = False
            logger.info(f"--- Pipeline execution took: {state['metadata']['pipeline_time_ms']:.2f} ms ---")

        # Determine if the Knowledge Base was used effectively
        used_kb = state.get("context_is_relevant", False)

        # --- UPDATED CONFIDENCE CALCULATION ---
        retrieved_context = state.get("retrieved_context") or [] # Use .get() for safety
        top_score = retrieved_context[0]['score'] if retrieved_context and 'score' in retrieved_context[0] else 0.0

        if state.get("is_tool_use"): # Use .get() for safety
            confidence = 1.0 # Tool use is considered deterministic
        elif used_kb: # KB context was retrieved and deemed relevant
            # Confidence based on top retrieved score, slightly boosted but capped
            # Ensures a minimum confidence if KB was used, prevents overly low scores
            confidence = min(max(0.5, top_score * 1.1), 0.98)
        elif state.get("is_technical_rag"): # Technical query but no relevant KB found (fallback answer generated)
            confidence = 0.35 # Lower confidence for generic fallback advice
        elif state.get("triage_decision") in ["GREETING", "GRATITUDE", "CHITCHAT", "FAREWELL"]: # Handle simple known intents
            confidence = 0.95 # Assign a high, but not perfect, confidence for these
        elif state.get("triage_decision") == "NEEDS_CLARIFICATION":
            confidence = 0.5  # Lower confidence seems appropriate when asking for more info
        else: # Default for any other unforeseen simple cases or potential errors
            confidence = 0.9 # Default high confidence for cases not covered above
        # --- END UPDATED CONFIDENCE CALCULATION ---

        # Update session object with the latest turn
        session.conversation_history.append({
            "query": query, # Store the original query
            "answer": state["final_response"],
            "timestamp": datetime.now().isoformat(),
            "used_kb": used_kb,
            "confidence": confidence # Store the calculated confidence
        })
        session.last_activity = datetime.now() # Update activity timestamp

        # Update session title if it's the default and this isn't just a short greeting
        if session.session_title == "Agentic Chat" and len(session.conversation_history) >= 1:
             first_query = session.conversation_history[0].get("query", "")
             # Only update if the first query is reasonably descriptive
             if first_query and len(first_query.split()) > 2:
                  session.session_title = first_query[:40] + "..." if len(first_query) > 40 else first_query

        # Save the updated session state to the database
        save_session(session)

        # Log the interaction details to the history table for analytics
        try:
            con = get_db()
            cur = con.cursor()
            cur.execute("""
                INSERT INTO history (session_id, query, answer, confidence, used_kb, exact_match, clarification_asked)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                query,
                state["final_response"],
                confidence,
                int(used_kb),
                0, # Assuming exact_match logic is elsewhere or not primary here
                int(state.get("needs_clarification", False)) # Log if clarification was asked
                ))
            con.commit()
        except Exception as e:
            logger.error(f"History logging error: {e}", exc_info=True) # Log full traceback

        # Prepare the final response object for the API caller (agentic_app.py)
        # Include necessary metadata for the frontend
        pipeline_time_ms = state["metadata"].get("pipeline_time_ms", total_time * 1000)
        retrieval_time_approx = pipeline_time_ms * 0.3 / 1000.0 # Approximate retrieval time in seconds

        return {
            "answer": state["final_response"],
            "confidence": confidence,
            "used_kb": used_kb,
            "exact_match": False, # Assuming False unless specific logic sets it
            "session_id": session_id, # Return the session ID
            "feedback_required": used_kb and confidence > 0.6, # Suggest feedback for KB answers
            "clarification_asked": state.get("needs_clarification", False),
            "total_time": total_time, # Total request time (including potential cache hit time)
            "retrieval_time": retrieval_time_approx, # Approximate retrieval portion
            "session_title": session.session_title, # Pass updated title back
            "theme_preference": session.theme_preference # Pass theme back
            # Add cache hit status for potential frontend display or debugging
            # "cache_hit": state["metadata"].get("cache_hit", False)
        }

    except Exception as e:
        logger.error(f"Agentic RAG main function error: {e}", exc_info=True) # Log full traceback
        # Return a standardized error response
        return {
            "answer": "I apologize, but I encountered an error while processing your request. Please try again.",
            "confidence": 0.0,
            "used_kb": False,
            "exact_match": False,
            "session_id": session_id, # Return session ID even on error
            "feedback_required": False,
            "clarification_asked": False,
            "total_time": time.time() - start_time,
            "retrieval_time": 0,
            "session_title": session.session_title if 'session' in locals() else "Agentic Chat", # Handle potential error before session load
            "theme_preference": session.theme_preference if 'session' in locals() else "system"
        }

# ==============================================================================
# --- COMPATIBILITY FUNCTIONS FOR app.py ---
# ==============================================================================

def query_rag(query: str, session_id: Optional[str] = None, user_id: str = "default") -> Dict[str, Any]:
    """Primary entry point for app.py - uses agentic backend."""
    return run_agentic_rag(query, session_id, user_id)

def load_session(session_id: str, user_id: str = "default") -> Optional[Dict[str, Any]]:
    """Compatibility function for app.py - loads session with frontend format"""
    try:
        session = get_session(session_id, user_id)
        return {
            "session_id": session.session_id,
            "title": session.session_title,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": len(session.conversation_history),
            "theme_preference": session.theme_preference,
            "conversation_history": session.conversation_history
        }
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}")
        return None

def get_all_sessions_from_db(user_id: str = "default") -> List[Dict]:
    """Get all sessions for sidebar."""
    try:
        con = get_db()
        cur = con.cursor()
        cur.execute("""
            SELECT session_id, session_title, created_at, last_activity, conversation_history, theme_preference
            FROM sessions WHERE user_id = ? AND is_active = 1 ORDER BY last_activity DESC
        """, (user_id,))
        
        sessions = []
        for row in cur.fetchall():
            session_id, title, created_at, last_activity, history_json, theme = row
            history = json.loads(history_json or '[]')
            sessions.append({
                "session_id": session_id,
                "title": title or "Agentic Chat",
                "created_at": created_at,
                "last_activity": last_activity,
                "message_count": len(history),
                "theme_preference": theme or "system"
            })
        return sessions
    except Exception as e:
        logger.error(f"Get sessions error: {e}")
        return []

def get_conversation_history(session_id: str, limit: int = 50) -> List[Dict]:
    """Get conversation history for a session."""
    try:
        con = get_db()
        cur = con.cursor()
        cur.execute("SELECT conversation_history FROM sessions WHERE session_id = ? AND is_active = 1", (session_id,))
        row = cur.fetchone()
        if row:
            history_json = row[0]
            history = json.loads(history_json or '[]')
            transformed_history = []
            for item in history:
                transformed_history.append({
                    "query": item.get("query"),
                    "answer": item.get("answer"),
                    "confidence": item.get("confidence", 0),
                    "used_kb": item.get("used_kb", False),
                    "exact_match": False,
                    "timestamp": item.get("timestamp", "")
                })
            return transformed_history[-limit:]
        
        cur.execute("""
            SELECT query, answer, confidence, used_kb, exact_match, timestamp
            FROM history WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?
        """, (session_id, limit))
        
        history = []
        for row in cur.fetchall():
            history.append({
                "query": row[0],
                "answer": row[1],
                "confidence": float(row[2]),
                "used_kb": bool(row[3]),
                "exact_match": bool(row[4]),
                "timestamp": row[5]
            })
        return history
    except Exception as e:
        logger.error(f"Get history error: {e}")
        return []

def create_new_session(user_id: str = "default") -> Dict[str, Any]:
    """Create a new session."""
    session_id = str(uuid.uuid4())[:12]
    session = Session(
        session_id=session_id,
        user_id=user_id,
        created_at=datetime.now(),
        last_activity=datetime.now(),
        session_title="Agentic Chat"
    )
    save_session(session)
    
    return {
        "session_id": session_id,
        "title": "Agentic Chat",
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "message_count": 0,
        "theme_preference": "system"
    }

def submit_feedback(session_id: str, query: str, answer: str, rating: int, comment: str = "") -> Dict[str, Any]:
    """Submit feedback."""
    try:
        con = get_db()
        cur = con.cursor()
        cur.execute("""
            INSERT INTO feedback (session_id, query, answer, rating, comment)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, query, answer, rating, comment))
        con.commit()
        return {"success": True}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return {"success": False}

def get_performance_stats() -> Dict[str, Any]:
    """Get performance statistics."""
    try:
        return analytics_manager.get_performance_stats()
    except Exception as e:
        logger.error(f"get_performance_stats error: {e}")
        return {
            "total_queries": 0,
            "kb_usage_rate": 0,
            "avg_confidence": 0,
            "active_sessions": 0
        }

def delete_session(session_id: str, user_id: str = "default") -> bool:
    """Delete session (marks as inactive)."""
    try:
        con = get_db()
        cur = con.cursor()
        cur.execute("UPDATE sessions SET is_active = 0 WHERE session_id = ? AND user_id = ?", 
                     (session_id, user_id))
        con.commit()
        return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        return False

def rename_session(session_id: str, new_title: str, user_id: str = "default") -> bool:
    """Rename session."""
    try:
        con = get_db()
        cur = con.cursor()
        cur.execute("UPDATE sessions SET session_title = ? WHERE session_id = ? AND user_id = ?", 
                     (new_title, session_id, user_id))
        con.commit()
        return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Rename session error: {e}")
        return False

def clear_session_history(session_id: str, user_id: str = "default") -> bool:
    """Clear session history."""
    try:
        session = get_session(session_id, user_id)
        if session:
            session.conversation_history = []
            save_session(session)
            return True
        return False
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return False

def update_theme_preference(session_id: str, theme: str, user_id: str = "default") -> bool:
    """Update theme preference."""
    try:
        session = get_session(session_id, user_id)
        if session:
            session.theme_preference = theme
            save_session(session)
            return True
        return False
    except Exception as e:
        logger.error(f"Theme update error: {e}")
        return False

def get_next_word_predictions(partial_text: str) -> List[str]:
    """Get next word predictions."""
    common_phrases = [
        "How do I troubleshoot", "How to fix", "Error when", "Problem with",
        "Cannot connect to", "Login issue", "Password reset", "Network connectivity",
        "Create a ticket for", "What is the status of INC"
    ]
    matches = [p for p in common_phrases if p.lower().startswith(partial_text.lower())]
    return matches[:3]

def init_db():
    """Initialize database tables."""
    try:
        con = get_db()
        cur = con.cursor()
        
        # History table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                session_id TEXT, 
                query TEXT,
                answer TEXT, 
                confidence REAL, 
                used_kb INTEGER, 
                exact_match INTEGER,
                clarification_asked INTEGER DEFAULT 0,
                user_feedback INTEGER,
                feedback_comment TEXT, 
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Feedback table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                session_id TEXT, 
                query TEXT,
                answer TEXT, 
                rating INTEGER, 
                comment TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sessions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, 
                user_id TEXT, 
                created_at DATETIME,
                last_activity DATETIME, 
                conversation_history TEXT, 
                pending_clarification TEXT,
                feedback_pending INTEGER DEFAULT 0,
                session_title TEXT DEFAULT 'Agentic Chat',
                is_active INTEGER DEFAULT 1,
                theme_preference TEXT DEFAULT 'system'
            )
        """)
        
        con.commit()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

# ==============================================================================
# --- INITIALIZATION ---
# ==============================================================================

# Initialize vector store after all imports are complete
initialize_vector_store()
logger.info("Refactored agentic backend (agentic_backend.py) initialized successfully!")

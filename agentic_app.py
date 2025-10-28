# agentic_app.py - WITH STREAMING MOVED TO ANALYTICS MANAGER
from flask import Flask, request, jsonify, Response, send_file, stream_with_context, send_from_directory
from servicenow_integration import servicenow_client
import json
import os
import time
import logging
from datetime import datetime
from typing import Optional
# Removed threading, ThreadPoolExecutor, Queue - now managed by analytics_manager
import database

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import functions from agentic_backend
# Note: analytics_manager is imported directly now
from agentic_backend import (
    query_rag,
    get_all_sessions_from_db,
    get_conversation_history,
    submit_feedback,
    get_performance_stats, # Still used by some endpoints
    create_new_session,
    get_session as load_session, # Renamed for clarity in app.py context
    delete_session,
    clear_session_history,
    rename_session,
    update_theme_preference,
    get_next_word_predictions,
    analytics_manager,  # Import the instance directly
    init_db,
    Session
)

# --- Flask App Setup ---
app = Flask(__name__, static_folder='static', static_url_path='')
database.init_app(app) # Initialize DB connection handling via database.py

# Helper function to get session data
# Note: This might be redundant if load_session is used directly, but kept for clarity
def get_session(session_id, user_id='default'):
    """Get session data - compatible with your backend"""
    try:
        # Calls the imported load_session (originally get_session from backend)
        return load_session(session_id, user_id)
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        return None

# --- Static File Serving ---
@app.route('/')
def serve_frontend():
    """Serves the main index.html file."""
    try:
        return send_from_directory('static', 'index.html')
    except Exception as e:
        logger.error(f"Error loading index.html: {str(e)}")
        return f"Error loading index.html: {str(e)}", 500

@app.route('/<path:path>')
def serve_static(path):
    """Serves other static files (CSS, JS, images)."""
    return send_from_directory('static', path)

# --- REAL-TIME ANALYTICS STREAM ENDPOINT (Uses Analytics Manager) ---
@app.route('/api/analytics/stream')
def analytics_live_stream():
    """Provides a Server-Sent Events stream for real-time analytics."""
    def generate():
        # Uses Queue from the standard library
        from queue import Queue
        q = Queue()
        # Subscribe using the analytics_manager's stream object
        analytics_manager.realtime_stream.add_subscriber(q)
        logger.info("Analytics stream client connected.")
        try:
            # Send initial state immediately using the manager's metrics
            initial_data = {'type': 'initial_state', 'metrics': analytics_manager.realtime_metrics}
            yield f"data: {json.dumps(initial_data)}\n\n"

            # Listen for updates broadcast by the manager's background thread
            while True:
                data = q.get() # Blocks until data is available
                yield f"data: {data}\n\n"
        except GeneratorExit:
            # Unsubscribe when the client disconnects
            analytics_manager.realtime_stream.remove_subscriber(q)
            logger.info("Analytics stream client disconnected.")

    # Return a streaming response
    return Response(generate(), mimetype='text/event-stream')

# --- SERVICENOW INTEGRATION ENDPOINTS ---
@app.route('/api/servicenow/create-ticket', methods=['POST'])
def create_servicenow_ticket():
    """Endpoint to create a ServiceNow ticket, potentially using conversation analysis."""
    try:
        data = request.get_json()

        session_id = data.get('session_id')
        # Allow direct input from form or analysis
        short_description = data.get('short_description')
        description = data.get('description')
        urgency = data.get('urgency')
        category = data.get('category')
        use_llm_analysis = data.get('use_llm_analysis', False) # Flag from frontend

        conversation_history = []
        if session_id:
            # Use the backend function to get history
            session_history_list = get_conversation_history(session_id)
            if session_history_list:
                conversation_history = session_history_list # Assuming it returns the list directly
                logger.info(f"Retrieved {len(conversation_history)} messages for ticket creation analysis")

        # Call the client's method which handles both direct input and analysis
        result = servicenow_client.analyze_and_create_incident(
            conversation_history=conversation_history,
            use_llm=use_llm_analysis, # Pass the flag
            # Pass direct kwargs which will be used if short_description is present
            short_description=short_description,
            description=description,
            urgency=urgency,
            category=category
        )

        logger.info(f"Ticket creation result: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error creating ServiceNow ticket: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Failed to create ticket: {str(e)}"
        }), 500

@app.route('/api/servicenow/get-ticket', methods=['POST'])
def get_servicenow_ticket():
    """Fetch ServiceNow ticket details by ticket number."""
    try:
        data = request.get_json()
        ticket_number = data.get('ticket_number')

        if not ticket_number:
            return jsonify({
                "status": "error",
                "message": "Ticket number is required"
            }), 400

        logger.info(f"Fetching ticket: {ticket_number}")

        # Use the imported client directly
        result = servicenow_client.get_servicenow_ticket(ticket_number)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error fetching ServiceNow ticket: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Failed to fetch ticket: {str(e)}"
        }), 500

@app.route('/api/servicenow/analyze-conversation', methods=['POST'])
def analyze_conversation():
    """Preview conversation analysis without creating a ticket."""
    try:
        data = request.json
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400

        # Get history using the backend function
        session_history = get_conversation_history(session_id)

        if not session_history:
            return jsonify({'error': 'No conversation history found or session invalid'}), 404 # Use 404

        logger.info(f"Analyzing {len(session_history)} messages for ticket preview")

        # Use the analyzer directly from the client instance
        llm_result = servicenow_client.ticket_analyzer.analyze_conversation_llm(session_history)

        if llm_result["status"] == "success":
            logger.info("LLM analysis successful for preview")
            return jsonify({
                "status": "success",
                "analysis": llm_result["analysis"],
                "source": "llm"
            })
        else:
            # Fallback to rule-based for preview as well
            logger.warning(f"LLM failed for preview, using rule-based: {llm_result.get('message', 'Unknown error')}")
            rule_based = servicenow_client.ticket_analyzer.analyze_conversation_rule_based(session_history)
            return jsonify({
                "status": "success",
                "analysis": rule_based,
                "source": "rule_based" # Make sure source indicates rule-based
            })

    except Exception as e:
        logger.error(f"Error analyzing conversation: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/api/servicenow/troubleshoot-ticket', methods=['POST'])
def troubleshoot_ticket():
    """Use ticket details to search the Knowledge Base via the RAG system."""
    try:
        data = request.get_json()
        ticket_data = data.get('ticket_data')

        if not ticket_data:
            return jsonify({'error': 'Ticket data is required'}), 400

        ticket_number = ticket_data.get('number', '')
        short_description = ticket_data.get('short_description', '')
        description = ticket_data.get('description', '')
        urgency = ticket_data.get('urgency', '')
        state = ticket_data.get('state', '')

        # Construct a query for the RAG system
        query = f"Troubleshoot ServiceNow ticket {ticket_number}: {short_description}. Details: {description}. Urgency: {urgency}, State: {state}"
        logger.info(f"Troubleshooting ticket with KB query: {query[:100]}...")

        # Call the main RAG query function from agentic_backend
        # Use a fresh session_id or handle appropriately
        result = query_rag(query, session_id=f"servicenow_troubleshoot_{ticket_number}", user_id="system_servicenow")

        return jsonify({
            'status': 'success',
            'ticket_number': ticket_number,
            'query_used': query,
            'kb_response': result # Return the full RAG response
        })

    except Exception as e:
        logger.error(f"Error troubleshooting ticket: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Failed to troubleshoot ticket: {str(e)}'
        }), 500

@app.route('/api/servicenow/test-connection', methods=['GET'])
def test_servicenow_connection():
    """Test ServiceNow connection status."""
    try:
        # Use the imported client directly
        result = servicenow_client.test_connection()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error testing ServiceNow connection: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Connection test failed: {str(e)}'
        }), 500

# --- CHAT ENDPOINTS ---
@app.route('/api/chat', methods=['POST'])
def chat_stream():
    """Main streaming chat endpoint."""
    data = request.json
    query = data.get('message', '')
    session_id = data.get('session_id') # Get session_id from request
    user_id = data.get('user_id', 'default') # Get user_id or default

    if not query:
        return jsonify({'error': 'Message is required'}), 400

    # If no session_id is provided by the client, the backend will generate one
    logger.info(f"Chat request received: Session='{session_id}', Query='{query[:50]}...'")

    def generate():
        try:
            start_time = time.time()

            # Call the main RAG function from agentic_backend
            # It now handles session creation/loading internally if session_id is None/valid
            result = query_rag(query, session_id, user_id)
            answer = result.get("answer", "Sorry, I encountered an issue.")
            # Important: Get the potentially *new* session_id back from the result
            returned_session_id = result.get("session_id", session_id)

            # Record metrics using the analytics manager instance
            analytics_manager.record_query_metrics({
                'query': query,
                'confidence': result.get('confidence', 0),
                'success': result.get('used_kb', False), # Use used_kb as proxy for success
                'response_time': result.get('total_time', time.time() - start_time),
                'kb_used': result.get('used_kb', False),
                'session_id': returned_session_id,
                'user_id': user_id,
                'clarification_asked': result.get('clarification_asked', False),
                'exact_match': result.get('exact_match', False),
                # Add retrieval/generation time if available from backend result metadata
                'retrieval_time': result.get('retrieval_time', 0),
                'generation_time': result.get('total_time', 0) - result.get('retrieval_time', 0) # Approximate
            })

            # Stream the response word by word (simulate typing)
            words = answer.split()
            for i, word in enumerate(words):
                yield f"data: {json.dumps({'type': 'token', 'content': word + (' ' if i < len(words) - 1 else '')})}\n\n"
                time.sleep(0.03) # Adjust typing speed if needed

            # Signal completion with full result metadata
            yield f"data: {json.dumps({
                'type': 'complete',
                'confidence': result.get('confidence', 0),
                'used_kb': result.get('used_kb', False),
                'session_id': returned_session_id, # Send back the correct session_id
                'feedback_required': result.get('feedback_required', False),
                'session_title': result.get('session_title', 'Agentic Chat'), # Include title
                'theme_preference': result.get('theme_preference', 'system') # Include theme
            })}\n\n"

        except Exception as e:
            logger.error(f"Error in chat stream generation: {e}", exc_info=True)
            # Send an error message to the client
            yield f"data: {json.dumps({'type': 'error', 'content': 'Sorry, an internal error occurred.'})}\n\n"

    # Return the streaming response using Flask's stream_with_context
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/query', methods=['POST'])
def api_query():
    """Original non-streaming query endpoint for backward compatibility or simpler clients."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        session_id = data.get('session_id')
        user_id = data.get('user_id', 'default')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        start_time = time.time()
        # Call the main RAG function
        result = query_rag(query, session_id, user_id)
        # Get the potentially new session_id
        returned_session_id = result.get("session_id", session_id)

        # Record metrics
        analytics_manager.record_query_metrics({
            'query': query,
            'confidence': result.get('confidence', 0),
            'success': result.get('used_kb', False),
            'response_time': result.get('total_time', time.time() - start_time),
            'kb_used': result.get('used_kb', False),
            'session_id': returned_session_id,
            'user_id': user_id,
            'clarification_asked': result.get('clarification_asked', False),
            'exact_match': result.get('exact_match', False),
            'retrieval_time': result.get('retrieval_time', 0),
            'generation_time': result.get('total_time', 0) - result.get('retrieval_time', 0)
        })

        # Return the complete result object from the backend
        return jsonify(result)

    except Exception as e:
        logger.error(f"API query error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


# --- SESSION MANAGEMENT ENDPOINTS ---
@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all active sessions for the sidebar."""
    try:
        user_id = request.args.get('user_id', 'default')
        # Use the backend function directly
        sessions = get_all_sessions_from_db(user_id)
        # The backend function already returns the format expected by frontend based on its code.
        return jsonify(sessions)
    except Exception as e:
        logger.error(f"Error getting sessions: {e}", exc_info=True)
        return jsonify([]), 500 # Return empty list on error

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session_history_route(session_id):
    """Get conversation history and session details for a specific session."""
    try:
        user_id = request.args.get('user_id', 'default') # Optional user_id check

        # Load the Session object from the backend
        session_info: Optional[Session] = load_session(session_id, user_id) # Type hint added for clarity

        # Use the backend function to get history list
        history = get_conversation_history(session_id) # Limit can be added if needed

        # Check if session exists and get details using attribute access
        session_title = 'Chat History'
        theme_preference = 'system'
        if session_info:
            # --- CORRECTED ACCESS ---
            session_title = session_info.session_title
            theme_preference = session_info.theme_preference
            # ------------------------
        else:
             # Handle case where session might not be found by load_session
             logger.warning(f"Session info not found for {session_id} during history request.")


        return jsonify({
            'session_id': session_id,
            'title': session_title,
            'theme_preference': theme_preference,
            'messages': history # Pass the history list directly
        })

    except Exception as e:
        logger.error(f"Error getting session history for {session_id}: {e}", exc_info=True)
        # Return a more informative error structure
        return jsonify({
            'session_id': session_id,
            'messages': [],
            'error': f'Internal server error: {str(e)}'
            }), 500

@app.route('/api/sessions/new', methods=['POST'])
def create_new_session_route():
    """Create a new session."""
    try:
        data = request.json or {}
        user_id = data.get('user_id', 'default')

        # Use the backend function
        session_info = create_new_session(user_id)
        return jsonify(session_info)

    except Exception as e:
        logger.error(f"Error creating new session: {e}", exc_info=True)
        return jsonify({'error': 'Failed to create session', 'message': str(e)}), 500

@app.route('/api/sessions/<session_id>/delete', methods=['POST'])
def delete_session_route(session_id):
    """Delete (mark as inactive) a session."""
    try:
        data = request.json or {} # Allow user_id in body or args
        user_id = data.get('user_id', request.args.get('user_id', 'default'))

        # Use the backend function
        success = delete_session(session_id, user_id)

        return jsonify({'success': success})

    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/sessions/<session_id>/clear', methods=['POST'])
def clear_session_route(session_id):
    """Clear the conversation history of a session."""
    try:
        data = request.json or {}
        user_id = data.get('user_id', request.args.get('user_id', 'default'))

        # Use the backend function
        success = clear_session_history(session_id, user_id)

        return jsonify({'success': success})

    except Exception as e:
        logger.error(f"Error clearing session {session_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/sessions/<session_id>/rename', methods=['POST'])
def rename_session_route(session_id):
    """Rename a session."""
    try:
        data = request.json
        new_title = data.get('new_title')
        user_id = data.get('user_id', 'default') # Get user_id from body

        if not new_title or not isinstance(new_title, str) or len(new_title.strip()) == 0:
            return jsonify({'error': 'New title is required and must be a non-empty string'}), 400

        # Use the backend function
        success = rename_session(session_id, new_title.strip(), user_id)
        return jsonify({'success': success})

    except Exception as e:
        logger.error(f"Error renaming session {session_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/sessions/<session_id>/theme', methods=['POST'])
def update_theme_route(session_id):
    """Update the theme preference for a session."""
    try:
        data = request.json
        theme = data.get('theme')
        user_id = data.get('user_id', 'default')

        # Basic validation for theme
        if theme not in ['light', 'dark', 'system']:
            return jsonify({'error': 'Invalid theme value. Must be light, dark, or system.'}), 400

        # Use the backend function
        success = update_theme_preference(session_id, theme, user_id)
        return jsonify({'success': success})

    except Exception as e:
        logger.error(f"Error updating theme for session {session_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500


# --- FEEDBACK ENDPOINT ---
@app.route('/api/feedback', methods=['POST'])
def submit_feedback_route():
    """Submit user feedback for a query/answer pair."""
    try:
        data = request.json
        session_id = data.get('session_id')
        query = data.get('query', '')
        answer = data.get('answer', '')
        # Frontend might send 0 (down) or 1 (up)
        frontend_rating = data.get('rating')
        comment = data.get('comment', '')

        if not session_id or frontend_rating is None:
            return jsonify({'success': False, 'error': 'Session ID and rating are required'}), 400

        # Convert frontend rating (0/1) to backend rating (1-5 scale)
        # Assuming 1 = thumbs down -> maps to 1
        # Assuming 1 = thumbs up -> maps to 5
        # Handle potential non-integer rating gracefully
        try:
            backend_rating = 5 if int(frontend_rating) == 1 else 1
        except (ValueError, TypeError):
             return jsonify({'success': False, 'error': 'Invalid rating value'}), 400

        # Use the backend submit_feedback function
        result = submit_feedback(session_id, query, answer, backend_rating, comment)

        # Analytics manager recording is handled within the backend submit_feedback if needed,
        # or can be explicitly called here if separated. Assuming it's separate:
        # analytics_manager.record_feedback(session_id, query, answer, backend_rating, comment)

        return jsonify({'success': result.get('success', False)})

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500


# --- ANALYTICS DASHBOARD ENDPOINTS ---
# Serve static files for the separate analytics dashboard
@app.route('/analytics')
def serve_analytics():
    """Serve the main HTML for the analytics dashboard."""
    try:
        # Assuming analytics dashboard files are in an 'analytics' subfolder within 'static'
        # Or change 'analytics' to the correct directory name if it's top-level
        return send_from_directory('analytics', 'an_index.html')
    except Exception as e:
        logger.error(f"Error loading analytics dashboard: {str(e)}")
        return f"Error loading analytics dashboard: {str(e)}", 500

@app.route('/analytics/<path:path>')
def serve_analytics_static(path):
    """Serve static files (CSS, JS) for the analytics dashboard."""
    return send_from_directory('analytics', path)

# API endpoints to provide data TO the analytics dashboard
@app.route('/api/analytics/admin-dashboard', methods=['GET'])
def get_admin_dashboard_data():
    """Get all data needed for the main admin dashboard view."""
    try:
        admin_data = analytics_manager.get_admin_dashboard()
        return jsonify(admin_data)
    except Exception as e:
        logger.error(f"Error getting admin dashboard data: {e}", exc_info=True)
        
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/api/analytics/details', methods=['GET'])
def get_detailed_analytics_data():
    """Get detailed analytics breakdown."""
    try:
        analytics_data = analytics_manager.get_comprehensive_analytics()
        # You might want to restructure this slightly based on dashboard needs
        return jsonify(analytics_data)
    except Exception as e:
        logger.error(f"Error getting detailed analytics: {e}", exc_info=True)
        return jsonify({}), 500 # Return empty object on error

@app.route('/api/analytics/trends', methods=['GET'])
def get_trend_analysis_data():
    """Get trend analysis data (e.g., queries/day over time)."""
    try:
        days = request.args.get('days', 30, type=int)
        trend_data = analytics_manager.get_trend_analysis(days=days)
        return jsonify(trend_data)
    except Exception as e:
        logger.error(f"Error getting trend analysis: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get trend analysis', 'message': str(e)}), 500

@app.route('/api/analytics/performance', methods=['GET'])
def get_system_performance_data():
    """Get detailed system performance metrics."""
    try:
        performance_data = analytics_manager.get_performance_stats()
        # Maybe add more details if needed from comprehensive analytics
        return jsonify(performance_data)
    except Exception as e:
        logger.error(f"Error getting system performance data: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get performance data', 'message': str(e)}), 500

@app.route('/api/analytics/query-patterns', methods=['GET'])
def get_query_patterns_data():
    """Get analysis of common query patterns."""
    try:
        analytics_data = analytics_manager.get_comprehensive_analytics()
        query_trends = analytics_data.get('query_trends', {})
        return jsonify({
            'top_queries': query_trends.get('top_queries', []),
            'categories': query_trends.get('categories', {}),
        })
    except Exception as e:
        logger.error(f"Error getting query patterns: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get query patterns', 'message': str(e)}), 500

@app.route('/api/analytics/user-dashboard', methods=['GET'])
def get_user_analytics_data():
    """Get analytics specific to a user (if implemented)."""
    try:
        user_id = request.args.get('user_id', 'default')
        user_dashboard_data = analytics_manager.get_user_dashboard(user_id)
        return jsonify(user_dashboard_data)
    except Exception as e:
        logger.error(f"Error getting user analytics for {user_id}: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/api/analytics/export', methods=['POST'])
def export_analytics_data():
    """Export analytics data to a file."""
    try:
        report_type = request.json.get('type', 'comprehensive')
        filename = analytics_manager.export_analytics_report(report_type)

        if filename and os.path.exists(filename):
             # Ensure the path is correct for send_file relative to the app's root
            return send_file(filename, as_attachment=True)
        else:
            logger.error(f"Export failed or file not found: {filename}")
            return jsonify({'error': 'Export failed or file not found'}), 500

    except Exception as e:
        logger.error(f"Error exporting analytics: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

# --- OTHER UTILITY ENDPOINTS ---
@app.route('/api/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    # Could add checks for DB connection, backend status etc.
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.1' # Example version
    })

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Endpoint for typeahead/autocomplete predictions."""
    try:
        partial_text = request.args.get('partial_text', '')
        if not partial_text:
            return jsonify({'predictions': []})

        # Use the backend function
        predictions = get_next_word_predictions(partial_text)

        return jsonify({'predictions': predictions})

    except Exception as e:
        logger.error(f"Error getting predictions: {e}", exc_info=True)
        return jsonify({'predictions': [], 'error': str(e)}), 500

# --- ADMIN ENDPOINTS (Optional - Secure appropriately in production) ---
@app.route('/api/admin/cleanup-analytics', methods=['POST'])
def cleanup_analytics_route():
    """Endpoint to trigger cleanup of old analytics data."""
    # IMPORTANT: Add authentication/authorization for admin endpoints
    try:
        # Maybe require an admin token in header or body
        # admin_token = request.headers.get('X-Admin-Token')
        # if admin_token != os.getenv('ADMIN_TOKEN'):
        #     return jsonify({'error': 'Unauthorized'}), 403

        analytics_manager.cleanup_old_data()
        return jsonify({'success': True, 'message': 'Analytics cleanup task initiated'})
    except Exception as e:
        logger.error(f"Error during analytics cleanup endpoint: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

# ===============================================
# App Initialization and Background Tasks
# ===============================================
# This block runs when the script is executed.
if __name__ == '__main__':
    # Initialize Database and Analytics Tables within Application Context
    with app.app_context():
        # init_db is called by database.init_app, but we need backend/analytics tables too
        try:
            init_db() # Ensure core DB tables (history, feedback, sessions) are ready
            analytics_manager.init_tables() # Ensure analytics-specific tables are ready
            logger.info("Database and analytics tables initialized successfully.")
        except Exception as e:
            logger.error(f"FATAL: Database initialization failed: {e}", exc_info=True)
            # Consider exiting if DB init fails critically
            # import sys
            # sys.exit(1)

    # Start the background thread for real-time metric updates via AnalyticsManager
    try:
        analytics_manager.init_streaming(app)
        logger.info("Real-time analytics streaming thread started.")
    except Exception as e:
        logger.error(f"Failed to start analytics streaming thread: {e}", exc_info=True)

    # Check for static files (optional but helpful for debugging deployment)
    static_files = ['index.html', 'style.css', 'script.js']
    logger.info("Checking for essential static files...")
    all_found = True
    for file in static_files:
        file_path = os.path.join(app.static_folder, file)
        if os.path.exists(file_path):
            logger.info(f"  ✅ Found: {file}")
        else:
            logger.warning(f"  ❌ Missing essential static file: {file_path}")
            all_found = False
    if not all_found:
         logger.warning("Missing static files might cause frontend issues.")

    # Start the Flask Development Server
    logger.info("--- Starting Astra IT Support Co-Pilot ---")
    logger.info(f"Environment: {'Development' if app.debug else 'Production'}")
    logger.info(f"ServiceNow Integration: {'ENABLED' if servicenow_client else 'DISABLED'}") # Check client existence
    logger.info(f"Real-Time Analytics: {'ACTIVE'}")
    logger.info(f"🚀 Server running on http://0.0.0.0:5000")
    logger.info("Access the frontend at http://localhost:5000")

    # Use host='0.0.0.0' to make it accessible on your network
    # debug=True enables auto-reloading and better error pages (DISABLE in production)
    app.run(debug=True, port=5000, host='0.0.0.0')

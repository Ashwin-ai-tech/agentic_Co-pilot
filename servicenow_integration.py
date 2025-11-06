# servicenow_integration.py
import requests
import json
import urllib3
import re
import os
from typing import Dict, List, Optional
from groq import Groq
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TicketAnalyzer:
    def __init__(self, groq_client=None):
        self.groq_client = groq_client

    def analyze_conversation_llm(self, conversation_history: List[Dict]) -> Dict:
        """Use LLM to analyze conversation and extract ticket details."""
        try:
            conversation_text = self._format_conversation_for_analysis(conversation_history)
            if not conversation_text.strip():
                return {"status": "error", "message": "No conversation content to analyze"}

            prompt = f"""
            Analyze this IT support conversation and extract key information for a ServiceNow ticket.

            CONVERSATION:
            {conversation_text}

            Provide a JSON response with:
            - short_description: Brief, clear title (max 10 words)
            - description: Detailed problem description summarizing the user's issue and what has been tried.
            - urgency: "1" (High), "2" (Medium), or "3" (Low) based on impact.
            - category: "software", "hardware", "network", "access", or "inquiry".
            - impact_analysis: Brief explanation of business impact.

            Base urgency on:
            - High (1): System down, critical business function blocked, security issue, multiple users affected.
            - Medium (2): Major feature broken, significant performance issue, single user blocked.
            - Low (3): Minor issue, workaround exists, cosmetic problem, information request.

            Respond ONLY with a valid JSON object, with no other text or markdown.
            """

            if not self.groq_client:
                return {"status": "error", "message": "LLM client not available"}

            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}, # Enforce JSON output
            )
            result_text = response.choices[0].message.content.strip()
            
            # Refined JSON parsing: Find the JSON block with regex
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not json_match:
                return {"status": "error", "message": "No valid JSON object found in LLM response."}
            
            result = json.loads(json_match.group(0))

            required_fields = ['short_description', 'description', 'urgency', 'category']
            if all(field in result for field in required_fields):
                return {"status": "success", "analysis": result}
            else:
                return {"status": "error", "message": "LLM response missing required fields"}

        except json.JSONDecodeError as e:
            return {"status": "error", "message": f"Failed to parse LLM response as JSON: {str(e)}"}
        except Exception as e:
            return {"status": "error", "message": f"LLM analysis failed: {str(e)}"}

    def analyze_conversation_rule_based(self, conversation_history: List[Dict]) -> Dict:
        """Fallback rule-based analysis when LLM fails."""
        conversation_text = self._format_conversation_for_analysis(conversation_history)
        return {
            "short_description": self._extract_short_description(conversation_text),
            "description": self._generate_description(conversation_text),
            "urgency": self._determine_urgency(conversation_text),
            "category": self._determine_category(conversation_text),
            "impact_analysis": "Automated analysis from conversation history",
            "source": "rule_based"
        }

    def _format_conversation_for_analysis(self, conversation_history: List[Dict]) -> str:
        if not conversation_history:
            return ""
        formatted = []
        for msg in conversation_history[-8:]:
            if msg.get('query'):
                formatted.append(f"User: {msg['query']}")
            if msg.get('answer'):
                answer = msg['answer'][:500] + "..." if len(msg['answer']) > 500 else msg['answer']
                formatted.append(f"Assistant: {answer}")
        return "\n".join(formatted)

    def _extract_short_description(self, text: str) -> str:
        first_user_line = next((line.replace('User:', '').strip() for line in text.split('\n') if line.startswith('User:')), None)
        if first_user_line:
            return first_user_line[:60] + "..." if len(first_user_line) > 60 else first_user_line
        return "IT Support Request from Astra Chat"

    def _determine_urgency(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ['down', 'crash', 'emergency', 'critical', 'outage', 'security', 'everyone']):
            return "1"
        if any(word in text_lower for word in ['slow', 'not working', 'error', 'broken', 'issue', 'failed', 'unable', 'blocked']):
            return "2"
        return "3"

    def _determine_category(self, text: str) -> str:
        text_lower = text.lower()
        category_map = {
            'software': ['software', 'application', 'program', 'app', 'browser'],
            'hardware': ['printer', 'computer', 'laptop', 'hardware', 'device', 'monitor'],
            'network': ['network', 'wifi', 'internet', 'vpn', 'connectivity'],
            'access': ['login', 'password', 'access', 'permission', 'account']
        }
        for category, keywords in category_map.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return "inquiry"
        
    def _generate_description(self, text: str) -> str:
        user_messages = [line.replace('User:', '').strip() for line in text.split('\n') if line.startswith('User:')]
        if user_messages:
            return f"User reported: {user_messages[0]}. Further details may be in the conversation log."
        return "Technical support issue identified through Astra chat conversation."

class ServiceNowIntegration:
    def __init__(self, instance_url: str, username: str, password: str, groq_api_key: str = None):
        if not all([instance_url, username, password]):
            raise ValueError("ServiceNow instance URL, username, and password are required.")
        self.instance_url = instance_url.rstrip('/')
        self.username = username
        self.password = password
        self.base_url = f"{self.instance_url}/api/now/table"
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.ticket_analyzer = TicketAnalyzer(self.groq_client)

    def create_incident(self, short_description: str, description: str, urgency: str = "2", category: str = "inquiry") -> Dict:
        """Creates a ServiceNow incident."""
        url = f"{self.base_url}/incident"
        payload = {
            "short_description": short_description,
            "description": description, # Refined: Use the clean, summarized description directly
            "urgency": urgency,
            "impact": self._get_impact_from_urgency(urgency),
            "category": category,
            "caller_id": self.username
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        try:
            response = requests.post(
                url, auth=(self.username, self.password), headers=headers,
                data=json.dumps(payload), verify=False, timeout=30
            )
            if response.status_code == 201:
                result = response.json().get("result", {})
                return {
                    "status": "success",
                    "incident_number": result.get("number"),
                    "sys_id": result.get("sys_id"),
                    "message": f"Incident {result.get('number')} created successfully."
                }
            else:
                return {"status": "error", "message": f"Failed with status {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Request failed: {str(e)}"}

    def get_servicenow_ticket(self, ticket_number: str) -> Dict:
        """Fetch a ServiceNow incident by its number."""
        # Implementation remains the same as your original, it's already good.
        # ... (Your existing get_servicenow_ticket code) ...
        # For brevity, I'll paste the original logic back in here.
        url = f"{self.instance_url}/api/now/table/incident"
        params = {"sysparm_query": f"number={ticket_number}", "sysparm_limit": 1}
        headers = {"Accept": "application/json"}
        try:
            response = requests.get(
                url, auth=(self.username, self.password), headers=headers, 
                params=params, verify=False, timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                if data["result"]:
                    ticket = data["result"][0]
                    return {
                        "status": "success",
                        "number": ticket.get("number"),
                        "short_description": ticket.get("short_description"),
                        "description": ticket.get("description", "N/A"),
                        "state": ticket.get("state"), "urgency": ticket.get("urgency"),
                        "priority": ticket.get("priority"),
                        "assigned_to": ticket.get("assigned_to", {}).get("display_value", "Unassigned") if ticket.get("assigned_to") else "Unassigned",
                        "sys_created_on": ticket.get("sys_created_on")
                    }
                else:
                    return {"status": "not_found", "message": "Ticket not found."}
            else:
                return {"status": "error", "message": f"Error: {response.status_code} - {response.text}"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Request failed: {str(e)}"}


    def _get_impact_from_urgency(self, urgency: str) -> str:
        return {"1": "1", "2": "2", "3": "3"}.get(urgency, "2")

    def analyze_and_create_incident(self, conversation_history: List[Dict], **kwargs) -> Dict:
        """Analyze conversation and create incident with intelligent details."""
        if not conversation_history and not kwargs.get('short_description'):
            return {"status": "error", "message": "No conversation or data provided for analysis."}

        # If form values are provided directly from the frontend, use them.
        if kwargs.get('short_description'):
            analysis = {
                "short_description": kwargs['short_description'],
                "description": kwargs.get('description', ''),
                "urgency": kwargs.get('urgency', "2"),
                "category": kwargs.get('category', "inquiry")
            }
        else: # Otherwise, perform analysis
            analysis_result = self.ticket_analyzer.analyze_conversation_llm(conversation_history)
            if analysis_result["status"] == "success":
                analysis = analysis_result["analysis"]
            else: # Fallback to rule-based on LLM failure
                analysis = self.ticket_analyzer.analyze_conversation_rule_based(conversation_history)

        return self.create_incident(
            short_description=analysis["short_description"],
            description=analysis["description"],
            urgency=analysis.get("urgency", "2"),
            category=analysis.get("category", "inquiry")
        )

    def test_connection(self) -> Dict:
        """Test ServiceNow connection."""
        url = f"{self.base_url}/incident?sysparm_limit=1"
        try:
            response = requests.get(
                url, auth=(self.username, self.password), headers={"Accept": "application/json"},
                verify=False, timeout=10
            )
            if 200 <= response.status_code < 300:
                return {"status": "success", "message": f"Connection successful (Status: {response.status_code})"}
            elif response.status_code == 401:
                return {"status": "error", "message": "Authentication failed - check credentials"}
            return {"status": "error", "message": f"Connection test failed with status: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Connection failed: {str(e)}"}

# --- Singleton Instance ---
# Load environment variables and create a single client instance
load_dotenv()

servicenow_client = ServiceNowIntegration(
    instance_url=os.getenv("SERVICENOW_INSTANCE"),
    username=os.getenv("SERVICENOW_USERNAME"),
    password=os.getenv("SERVICENOW_PASSWORD"),
    groq_api_key=os.getenv("GROQ_API_KEY")
)

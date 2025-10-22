"""
LLM helper (Azure OpenAI) for the MCP client.
- Uses Azure OpenAI chat completions endpoint (REST).
- Set AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_KEY, AZURE_API_VERSION in env or edit placeholders.
- The model/deployment should be set to your Azure deployment (e.g., gpt-4o).
"""

import os
import requests
import json
import re

# =============================
# 🔧 Azure OpenAI Configuration
# =============================
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', "")  # e.g. https://your-resource.openai.azure.com
AZURE_DEPLOYMENT = os.getenv('AZURE_DEPLOYMENT', "gpt-4o")      # e.g. gpt-4o
AZURE_API_KEY = os.getenv('AZURE_API_KEY', "")
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION', "2024-02-01")

# =============================
# 🧠 Prompts
# =============================
PROMPT_SYSTEM = (
    "You are an intent recognition assistant for IndiGo's MCP AI system.\n"
    "You will receive a user query related to flight operations.\n"
    "Your job: Return ONLY a valid JSON object with two keys: 'tool' and 'args'.\n\n"
    "Allowed tools: search_flight_data, flight_statistics, list_stations, raw_mongodb_query, health_check.\n\n"
    "Example responses:\n"
    "{ \"tool\": \"search_flight_data\", \"args\": {\"registration\": \"VTLLP\"} }\n"
    "{ \"tool\": \"flight_statistics\", \"args\": {\"station\": \"DEL\"} }\n"
    "{ \"tool\": \"health_check\", \"args\": {} }\n\n"
    "⚠️ Do NOT include any explanation or text outside the JSON.\n"
)

PROMPT_TEMPLATE = """{system}\nUser: {user}\n\nReturn JSON only:"""

# =============================
# 🚀 Azure Chat Request
# =============================
def call_azure_chat(user_text: str, max_tokens: int = 500):
    if not AZURE_OPENAI_ENDPOINT or not AZURE_DEPLOYMENT or not AZURE_API_KEY or not AZURE_API_VERSION:
        raise RuntimeError('❌ Missing Azure OpenAI configuration (endpoint, deployment, key, or version).')

    url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {'api-key': AZURE_API_KEY, 'Content-Type': 'application/json'}

    payload = {
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user_text}
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "response_format": {"type": "json_object"}  # ✅ Ensures strict JSON from model
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        raise RuntimeError(f"❌ Azure OpenAI call failed: {e}")

# =============================
# 🧩 JSON Parsing Helpers
# =============================
def parse_llm_json(raw: str):
    """
    Safely parse JSON output from model.
    Even if Azure adds extra newlines or text, we extract the JSON part.
    """
    try:
        # Extract JSON content between first '{' and last '}'
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError("No valid JSON object found.")
        clean_json = match.group(0)
        obj = json.loads(clean_json)

        # Validate required keys
        if "tool" not in obj or "args" not in obj:
            raise ValueError("Missing required keys ('tool', 'args') in JSON.")
        return obj

    except Exception as e:
        raise RuntimeError(f"[LLM parse error] {e}\nRaw output:\n{raw}")

# =============================
# 🔍 Main Entry for MCP Client
# =============================
def parse_user_query(user_text: str):
    prompt = PROMPT_TEMPLATE.format(system=PROMPT_SYSTEM, user=user_text)
    raw = call_azure_chat(prompt)
    return parse_llm_json(raw)












(venv) C:\Users\Krishna.x.Jaiswal\Downloads\flight_mcp_package>python client_mcp.py
🛫 MCP Client (LLM-driven) — type exit to quit
You: Give me all flights which are diverted to AMD 
-> Calling tool: search_flight_data with args: {"diverted_to": "AMD"}
[Server call error] HTTPConnectionPool(host='localhost', port=8000): Read timed out. (read timeout=30)
You: 

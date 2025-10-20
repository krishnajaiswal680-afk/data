(venv) C:\Users\Krishna.x.Jaiswal\Downloads\flight_mcp_package>python client_mcp.py                                       
🛫 MCP Client (LLM-driven) — type exit to quit
You: ram                    
[LLM parse error] '\n  "tool"'
You:

i am sharring code please update it and send me agin 

"""LLM helper (Azure OpenAI) for the MCP client.
- Uses Azure OpenAI chat completions endpoint (REST).
- Set AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_KEY in env or edit placeholders.
- The model/deployment should be set to your azure deployment (e.g., gpt-4o).
"""
import os
import requests
import json

AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', "")  # e.g., https://your-resource.openai.azure.com
AZURE_DEPLOYMENT = os.getenv('AZURE_DEPLOYMENT', "gpt-4o")  # e.g., gpt-4o
AZURE_API_KEY = os.getenv('AZURE_API_KEY', "")
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION', ")

PROMPT_SYSTEM = ("You are an assistant that receives a user request about flight operations and returns exactly a JSON "
                 "object with keys 'tool' and 'args'. Allowed tools: search_flight_data, flight_statistics, list_stations, raw_mongodb_query, health_check.\n"
                 "Return only JSON and nothing else.\n")

PROMPT_TEMPLATE = """{system}\nUser: {user}\n\nReturn a JSON object exactly like: {\n  \"tool\": <one of the allowed tool names as a string>,\n  \"args\": <an object with arguments for the tool, or {} if none>\n}\nIf the user requests a raw MongoDB query, choose 'raw_mongodb_query' and put the JSON in args.query_json.\nIf unsure, return {\"tool\": \"search_flight_data\", \"args\": {}}\n"""

def call_azure_chat(user_text: str, max_tokens: int = 500):
    if '<AZURE_ENDPOINT>' in AZURE_OPENAI_ENDPOINT or '<DEPLOYMENT_NAME>' in AZURE_DEPLOYMENT or '<AZURE_API_KEY>' in AZURE_API_KEY:
        raise RuntimeError('Please set AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT and AZURE_API_KEY environment variables.')
    url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {'api-key': AZURE_API_KEY, 'Content-Type': 'application/json'}
    payload = {
        'messages': [
            {'role': 'system', 'content': PROMPT_SYSTEM},
            {'role': 'user', 'content': user_text}
        ],
        'max_tokens': max_tokens,
        'temperature': 0.0
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # extract assistant message
    try:
        content = data['choices'][0]['message']['content']
    except Exception:
        content = json.dumps(data)
    return content

def parse_llm_json(raw: str):
    # extract JSON object between first { and last }
    try:
        start = raw.index('{')
        end = raw.rindex('}')
        obj = json.loads(raw[start:end+1])
        if 'tool' not in obj or 'args' not in obj:
            raise ValueError('Missing keys tool/args')
        return obj
    except Exception as e:
        raise RuntimeError(f'Failed to parse JSON from model output: {e}\nRaw:\n{raw}')

def parse_user_query(user_text: str):
    prompt = PROMPT_TEMPLATE.format(system=PROMPT_SYSTEM, user=user_text)
    raw = call_azure_chat(prompt)
    return parse_llm_json(raw)




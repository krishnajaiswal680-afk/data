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





























"""Flight Data MCP Server
- Registers MCP tools for flight data search and statistics.
- Exposes a HTTP endpoint /call to accept tool calls as JSON: { "tool": "<name>", "args": {...} }
- Uses Motor (async MongoDB) and FastAPI to accept HTTP requests.
- Replace MONGODB_URL with your connection string if needed.
"""
import os
import asyncio
import json
from typing import Any, Dict
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

# Config (replace via env vars)
MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb+srv://joc-dbuser:n5cqzIJSl319TZN6@mongo-airlineops-az-ddb01-pl-0.npznw.mongodb.net/')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'jocflightdb_30jan')
COLLECTION_FLIGHT = os.getenv('COLLECTION_METAR', 'flight')
LISTEN_HOST = os.getenv('HOST', '0.0.0.0')
LISTEN_PORT = int(os.getenv('PORT', '8000'))

# FastAPI app
app = FastAPI(title='Flight MCP Server')

# Global Mongo client
_client = None
_db = None

async def get_db():
    global _client, _db
    if _client is None:
        _client = AsyncIOMotorClient(MONGODB_URL)
        _db = _client[DATABASE_NAME]
    return _db

def format_flight_doc(doc: Dict[str, Any]) -> str:
    fl = doc.get('flightLegState', {})
    carrier = fl.get('carrier', '') or ''
    flight_no = fl.get('flightNumber', '') or ''
    start = fl.get('startStation', 'N/A')
    end = fl.get('endStation', 'N/A')
    date = fl.get('dateOfOrigin', 'N/A')
    sched_start = fl.get('scheduledStartTime', 'N/A')
    sched_end = fl.get('scheduledEndTime', 'N/A')
    actual = fl.get('operation', {}).get('actualTimes', {})
    offblock = actual.get('offBlock') or actual.get('takeoffTime') or 'N/A'
    landing = actual.get('landingTime') or actual.get('inBlock') or 'N/A'
    pax_list = fl.get('pax', {}).get('passengerCount', [])
    pax_count = None
    # try to find CheckInCount or total
    for p in pax_list:
        if p.get('code') in ('CheckInCount','DefaultTotalCount','TotalCount'):
            pax_count = p.get('count'); break
    if pax_count is None and pax_list:
        # fallback to first count
        pax_count = pax_list[0].get('count')
    aircraft = fl.get('aircraft', {})
    reg = aircraft.get('registration','N/A')
    atype = aircraft.get('type','N/A')
    delay = fl.get('delays', {}).get('total', 'N/A')

    return (
        f"✈️ {carrier}{flight_no} | {start} → {end} | Date: {date}\n"
        f"  Scheduled: {sched_start} → {sched_end}\n"
        f"  Actual: {offblock} → {landing}\n"
        f"  Delay: {delay}\n"
        f"  Pax: {pax_count}\n"
        f"  Aircraft: {reg} ({atype})\n"
    )

# --- MCP tools implementation ---
async def tool_search_flight_data(**kwargs) -> str:
    """Search flights with flexible filters.
    Supported kwargs (examples): carrier, flight_number, date_of_origin, start_station, end_station,
    delayed_only (bool), pax_min, pax_max, limit
    """
    db = await get_db()
    query = {}
    # map known args
    carrier = kwargs.get('carrier') or kwargs.get('airline')
    flight_number = kwargs.get('flight_number') or kwargs.get('flightNumber')
    date_of_origin = kwargs.get('date_of_origin') or kwargs.get('dateOfOrigin')
    start_station = kwargs.get('start_station') or kwargs.get('startStation') or kwargs.get('from')
    end_station = kwargs.get('end_station') or kwargs.get('endStation') or kwargs.get('to')
    delayed_only = kwargs.get('delayed_only') or kwargs.get('delayed') or False
    pax_min = kwargs.get('pax_min')
    pax_max = kwargs.get('pax_max')
    limit = int(kwargs.get('limit', 20))

    if carrier:
        query['flightLegState.carrier'] = carrier.upper()
    if flight_number:
        # support string/int
        try:
            query['flightLegState.flightNumber'] = int(flight_number)
        except Exception:
            query['flightLegState.flightNumber'] = flight_number
    if date_of_origin:
        query['flightLegState.dateOfOrigin'] = date_of_origin
    if start_station:
        query['flightLegState.startStation'] = start_station.upper()
    if end_station:
        query['flightLegState.endStation'] = end_station.upper()

    # run initial find
    cursor = db[COLLECTION_FLIGHT].find(query).sort('flightLegState.dateOfOrigin', -1).limit(limit)
    results = await cursor.to_list(length=limit)

    # post-filter delayed or pax if required
    filtered = []
    for doc in results:
        fl = doc.get('flightLegState', {})
        # compute delay: compare actual offBlock/takeoff with scheduledStartTime
        scheduled = fl.get('scheduledStartTime')
        actual = fl.get('operation', {}).get('actualTimes', {}).get('offBlock') or fl.get('operation', {}).get('actualTimes', {}).get('takeoffTime')
        is_delayed = False
        try:
            if scheduled and actual:
                # compare ISO strings
                sched_dt = datetime.fromisoformat(scheduled.replace('Z','+00:00'))
                actual_dt = datetime.fromisoformat(actual.replace('Z','+00:00'))
                is_delayed = actual_dt > sched_dt
        except Exception:
            is_delayed = False
        if delayed_only and not is_delayed:
            continue
        # pax checks
        pax_ok = True
        if pax_min is not None or pax_max is not None:
            pax_count = None
            pax_list = fl.get('pax', {}).get('passengerCount', [])
            for p in pax_list:
                if p.get('code') in ('CheckInCount','DefaultTotalCount','TotalCount'):
                    pax_count = p.get('count'); break
            if pax_count is None and pax_list:
                pax_count = pax_list[0].get('count')
            if pax_min is not None and (pax_count is None or pax_count < int(pax_min)):
                pax_ok = False
            if pax_max is not None and (pax_count is None or pax_count > int(pax_max)):
                pax_ok = False
        if not pax_ok:
            continue
        filtered.append(doc)

    if not filtered:
        return f"No flights found for filters: {query} (delayed_only={delayed_only}, pax_min={pax_min}, pax_max={pax_max})"

    out = f"Found {len(filtered)} flight(s)\n\n"
    for d in filtered:
        out += format_flight_doc(d) + "\n"
    return out

async def tool_flight_statistics(**kwargs) -> str:
    db = await get_db()
    total = await db[COLLECTION_FLIGHT].count_documents({})
    unique_aircraft = len(await db[COLLECTION_FLIGHT].distinct('flightLegState.aircraft.registration'))
    delayed_count = await db[COLLECTION_FLIGHT].count_documents({'flightLegState.delays.total': {'$exists': True}})
    # sample aggregation: max flights in a day in last n months - simple estimate
    stats = f"Total Flights: {total}\nUnique aircraft regs: {unique_aircraft}\nFlights with delays recorded: {delayed_count}\n"
    return stats

async def tool_list_stations(**kwargs) -> str:
    db = await get_db()
    starts = await db[COLLECTION_FLIGHT].distinct('flightLegState.startStation')
    ends = await db[COLLECTION_FLIGHT].distinct('flightLegState.endStation')
    starts_sorted = sorted([s for s in starts if s])
    ends_sorted = sorted([s for s in ends if s])
    return f"Departures ({len(starts_sorted)}): {', '.join(starts_sorted)}\nArrivals ({len(ends_sorted)}): {', '.join(ends_sorted)}"

async def tool_raw_query(query_json: str, limit: int = 20) -> str:
    db = await get_db()
    try:
        q = json.loads(query_json)
    except Exception as e:
        return f"Invalid JSON query: {e}"
    cursor = db[COLLECTION_FLIGHT].find(q).limit(limit)
    results = await cursor.to_list(length=limit)
    if not results:
        return f"No documents found for query: {q}"
    out = f"Raw Query Results ({len(results)})\n\n"
    for d in results:
        out += format_flight_doc(d) + "\n"
    return out

async def tool_health_check(**kwargs) -> str:
    return "Flight MCP Server OK"

# tool registry map
TOOL_MAP = {
    'search_flight_data': tool_search_flight_data,
    'flight_statistics': tool_flight_statistics,
    'list_stations': tool_list_stations,
    'raw_mongodb_query': tool_raw_query,
    'health_check': tool_health_check
}

class CallPayload(BaseModel):
    tool: str
    args: Dict[str, Any] = {}

@app.post('/call')
async def call_tool(payload: CallPayload):
    tool = payload.tool
    args = payload.args or {}
    if tool not in TOOL_MAP:
        raise HTTPException(status_code=400, detail=f"Tool '{tool}' not found.")
    fn = TOOL_MAP[tool]
    try:
        res = await fn(**args)
        return JSONResponse({'result': res})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health():
    return {'status': 'ok'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)


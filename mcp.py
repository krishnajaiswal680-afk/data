# app.py


import asyncio
import json
import streamlit as st
from dotenv import load_dotenv
from client import FlightOpsMCPClient

load_dotenv()

st.set_page_config(page_title="FlightOps Smart Agent (Groq MCP)", layout="wide")

st.title("✈️ FlightOps — Groq + MCP Chatbot")
st.caption("Ask any flight operations question. The LLM plans tool calls → MCP server executes → Groq summarizes.")

# Create global event loop if not exists
if "event_loop" not in st.session_state:
    st.session_state.event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.event_loop)

loop = st.session_state.event_loop

# Initialize MCP client once per session
if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = FlightOpsMCPClient()
mcp_client = st.session_state.mcp_client

# Connect once per app session
if "mcp_connected" not in st.session_state:
    try:
        loop.run_until_complete(mcp_client.connect())
        st.session_state.mcp_connected = True
        st.success("✅ Connected to MCP server")
    except Exception as e:
        st.error(f"❌ Could not connect to MCP server.\n\n{e}")
        st.stop()

with st.sidebar:
    st.markdown("## Server / LLM Info")
    st.write("**MCP Server:**", mcp_client.base_url)
    st.write("**LLM Model:**", "Groq - llama3-70b-8192")

st.markdown("### 💬 Example questions")
st.write("- Why was flight **6E215** delayed on **June 23, 2024**?")
st.write("- Show **aircraft** and **delay info** for **6E215**.")
st.write("- What were **operation times** for **6E215 on 2024-06-23**?")
st.write("---")

user_query = st.text_area("Your question:", height=100, key="query_box")

if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    st.info("🧠 Thinking with Groq LLM to plan the query...")
    with st.spinner("Generating tool plan and fetching results..."):
        try:
            # ✅ Use the same event loop, don't recreate
            result = loop.run_until_complete(mcp_client.run_query(user_query))
        except Exception as e:
            st.error(f"❌ Error during query:\n{e}")
            st.stop()

    plan = result.get("plan", [])
    if not plan:
        st.warning("LLM did not produce a valid tool plan.")
        st.json(result)
        st.stop()

    st.subheader("🗂️ LLM Tool Plan")
    st.json(plan)

    results = result.get("results", [])
    if results:
        st.subheader("🔧 MCP Tool Results")
        for step in results:
            tool_name = list(step.keys())[0]
            tool_result = step[tool_name]
            st.markdown(f"**Tool:** `{tool_name}`")
            st.json(tool_result)

    summary = result.get("summary", {}).get("summary", "")
    if summary:
        st.subheader("📝 Final Summary")
        st.write(summary)
    else:
        st.warning("No summary returned by Groq.")

    st.session_state.last_result = result

# Show previous result
if "last_result" in st.session_state:
    with st.expander("📦 Previous Results"):
        st.json(st.session_state.last_result)










client.py


import os
import json
import logging
import asyncio
from typing import List, Dict, Any
 
from dotenv import load_dotenv
from openai import AzureOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
 
from tool_registry import TOOLS
 
# Color reset function
def reset_terminal_colors():
    """Reset terminal colors to prevent light text issues"""
    print("\033[0m", end="")
 
# Load environment variables
load_dotenv()
 
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
 
# Azure OpenAI configuration
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
 
if not AZURE_OPENAI_KEY:
    raise RuntimeError("❌ AZURE_OPENAI_KEY not set in environment")
 
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("FlightOps.MCPClient")
 
# Initialize Azure OpenAI client
client_azure = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
 
def _build_tool_prompt() -> str:
    """Convert TOOLS dict into compact text to feed the LLM."""
    lines = []
    for name, meta in TOOLS.items():
        arg_str = ", ".join(meta["args"])
        lines.append(f"- {name}({arg_str}): {meta['desc']}")
    return "\n".join(lines)
 
SYSTEM_PROMPT_PLAN = f"""
You are an assistant that converts user questions into MCP tool calls.
Use only these tools exactly as defined below:
 
{_build_tool_prompt()}
 
Rules:
1. Output only valid JSON.
2. Always return a top-level key 'plan' as a list.
3. If user asks something general like 'details of flight', use get_flight_basic_info.
4. Do not invent tool names.
5. If carrier or date not mentioned, omit them instead of writing 'unknown'.
6. only use "tool" as key not "name"
"""
 
SYSTEM_PROMPT_SUMMARIZE = """
You are an assistant that summarizes tool outputs into a concise answer.
Focus on clarity and readability.
"""
 
class FlightOpsMCPClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or MCP_SERVER_URL).rstrip("/")
        self.session: ClientSession = None
        self._client_context = None
 
    async def connect(self):
        """Connect to the MCP server using streamable-http transport."""
        try:
            logger.info(f"Connecting to MCP server at {self.base_url}")
           
            # streamablehttp_client returns a context manager
            self._client_context = streamablehttp_client(self.base_url)
            read_stream, write_stream, _ = await self._client_context.__aenter__()
           
            # Create session
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
           
            # Initialize the connection
            await self.session.initialize()
            logger.info("✅ Connected to MCP server successfully")
           
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            reset_terminal_colors()  # Reset colors on error
            raise
 
    async def disconnect(self):
        """Disconnect from the MCP server."""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self._client_context:
                await self._client_context.__aexit__(None, None, None)
            logger.info("Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            reset_terminal_colors()  # Always reset colors
 
    def _call_azure_openai(self, messages: list, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        """Internal helper for Azure OpenAI chat completions."""
        try:
            completion = client_azure.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            reset_terminal_colors()  # Reset colors on error
            return json.dumps({"error": str(e)})
 
    # ---------- MCP Server Interaction ----------
 
    async def list_tools(self) -> dict:
        """List available tools from the MCP server."""
        try:
            if not self.session:
                await self.connect()
           
            tools_list = await self.session.list_tools()
           
            # Convert MCP tools response to dictionary format
            tools_dict = {}
            for tool in tools_list.tools:
                tools_dict[tool.name] = {
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
           
            return {"tools": tools_dict}
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            reset_terminal_colors()  # Reset colors on error
            return {"error": str(e)}
 
    async def invoke_tool(self, tool_name: str, args: dict) -> dict:    
        """Invoke a tool by name with arguments via MCP protocol."""
        try:
            if not self.session:
                await self.connect()
           
            logger.info(f"Calling tool: {tool_name} with args: {args}")
           
            # Call the tool using MCP session
            result = await self.session.call_tool(tool_name, args)    
           
            # Extract content from result
            if result.content:
                # MCP returns content as a list of Content objects
                content_items = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        try:
                            # Try to parse as JSON
                            content_items.append(json.loads(item.text))
                        except json.JSONDecodeError:
                            content_items.append(item.text)
               
                # If single item, return it directly
                if len(content_items) == 1:
                    return content_items[0]
                return {"results": content_items}
           
            return {"error": "No content in response"}
           
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {e}")
            reset_terminal_colors()  # Reset colors on error
            return {"error": str(e)}
 
    # ---------- LLM Wrappers ----------
 
    def plan_tools(self, user_query: str) -> dict:
        """Use Azure OpenAI to generate a plan of tool calls."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PLAN},
            {"role": "user", "content": user_query},
        ]
        content = self._call_azure_openai(messages, temperature=0.1)
        try:
            plan = json.loads(content)
            if isinstance(plan, dict) and "plan" in plan:
                return plan
            else:
                return {"plan": []}
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM plan output.")
            reset_terminal_colors()  # Reset colors on warning
            return {"plan": []}
 
    def summarize_results(self, user_query: str, plan: list, results: list) -> dict:
        """Use Azure OpenAI to summarize results into human-friendly output."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
            {"role": "user", "content": f"Question:\n{user_query}"},
            {"role": "assistant", "content": f"Plan:\n{json.dumps(plan, indent=2)}"},
            {"role": "assistant", "content": f"Results:\n{json.dumps(results, indent=2)}"},
        ]
        summary = self._call_azure_openai(messages, temperature=0.3)
        return {"summary": summary}
 
    # ---------- Orchestration ----------
 
    async def run_query(self, user_query: str) -> dict:
        """
        Full flow:
        1. Use LLM to plan tool calls.
        2. Execute tools sequentially on MCP server.
        3. Summarize results via LLM.
        """
        try:
            logger.info(f"User query: {user_query}")
            plan_data = self.plan_tools(user_query)
            plan = plan_data.get("plan", [])
 
            if not plan:
                return {"error": "LLM did not produce a valid tool plan."}
 
            results = []
            for step in plan:
                tool = step.get("tool")
                args = step.get("arguments", {})
 
                # Clean up 'unknown' or empty args
                args = {
                    k: v for k, v in args.items()
                    if v is not None and str(v).strip() != "" and str(v).lower() != "unknown"
                }
 
                if not tool:
                    continue
 
                logger.info(f"Invoking tool: {tool} with args: {args}")
                resp = await self.invoke_tool(tool, args)
                results.append({tool: resp})
 
            # Summarize results
            summary = self.summarize_results(user_query, plan, results)
            return {"plan": plan, "results": results, "summary": summary}
        except Exception as e:
            logger.error(f"Error in run_query: {e}")
            reset_terminal_colors()  # Reset colors on error
            return {"error": str(e)}
        finally:
            reset_terminal_colors()  # Always reset color







# server.py
import os
import logging
import json
from typing import Optional, Any, Dict
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
load_dotenv() 

from mcp.server.fastmcp import FastMCP

HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_PORT", "8000"))
TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")

MONGODB_URL = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("MONGO_DB")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("flightops.mcp.server")

mcp = FastMCP("FlightOps MCP Server")

_mongo_client: Optional[AsyncIOMotorClient] = None
_db = None
_col = None

async def get_mongodb_client():
    """Initialize and return the global Motor client, DB and collection."""
    global _mongo_client, _db, _col
    if _mongo_client is None:
        logger.info("Connecting to MongoDB: %s", MONGODB_URL)
        _mongo_client = AsyncIOMotorClient(MONGODB_URL)
        _db = _mongo_client[DATABASE_NAME]
        _col = _db[COLLECTION_NAME]
    return _mongo_client, _db, _col

def normalize_flight_number(flight_number: Any) -> Optional[int]:
    """Convert flight_number to int. MongoDB stores it as int."""
    if flight_number is None or flight_number == "":
        return None
    if isinstance(flight_number, int):
        return flight_number
    try:
        return int(str(flight_number).strip())
    except (ValueError, TypeError):
        logger.warning(f"Could not normalize flight_number: {flight_number}")
        return None

def validate_date(date_str: str) -> Optional[str]:
    """
    Validate date_of_origin string. Accepts common formats.
    Returns normalized ISO date string YYYY-MM-DD if valid, else None.
    """
    if not date_str or date_str == "":
        return None
    
    # Handle common date formats
    formats = [
        "%Y-%m-%d",      # 2024-06-23
        "%d-%m-%Y",      # 23-06-2024
        "%Y/%m/%d",      # 2024/06/23
        "%d/%m/%Y",      # 23/06/2024
        "%B %d, %Y",     # June 23, 2024
        "%d %B %Y",      # 23 June 2024
        "%b %d, %Y",     # Jun 23, 2024
        "%d %b %Y"       # 23 Jun 2024
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None

def make_query(carrier: str, flight_number: Optional[int], date_of_origin: str) -> Dict:
    """
    Build MongoDB query matching the actual database schema.
    """
    query = {}
    
    # Add carrier if provided
    if carrier:
        query["flightLegState.carrier"] = carrier
    
    # Add flight number as integer (as stored in DB)
    if flight_number is not None:
        query["flightLegState.flightNumber"] = flight_number
    
    # Add date if provided
    if date_of_origin:
        query["flightLegState.dateOfOrigin"] = date_of_origin
    
    logger.info(f"Built query: {json.dumps(query)}")
    return query

def response_ok(data: Any) -> str:
    """Return JSON string for successful response."""
    return json.dumps({"ok": True, "data": data}, indent=2, default=str)

def response_error(msg: str, code: int = 400) -> str:
    """Return JSON string for error response."""
    return json.dumps({"ok": False, "error": {"message": msg, "code": code}}, indent=2)

async def _fetch_one_async(query: dict, projection: dict) -> str:          #  Point of concern
    """
    Consistent async DB fetch and error handling.
    Returns JSON string response.
    """
    try:
        _, _, col = await get_mongodb_client()
        logger.info(f"Executing query: {json.dumps(query)}")
        
        result = await col.find_one(query, projection)
        
        if not result:
            logger.warning(f"No document found for query: {json.dumps(query)}")
            return response_error("No matching document found.", code=404)
        
        # Remove _id and _class to keep output clean
        if "_id" in result:
            result.pop("_id")
        if "_class" in result:
            result.pop("_class")
        
        logger.info(f"Query successful")
        return response_ok(result)
    except Exception as exc:
        logger.exception("DB query failed")
        return response_error(f"DB query failed: {str(exc)}", code=500)

# --- MCP Tools ---

@mcp.tool()
async def health_check() -> str:
    """
    Simple health check for orchestrators and clients.
    Attempts a cheap DB ping.
    """
    try:
        _, _, col = await get_mongodb_client()
        doc = await col.find_one({}, {"_id": 1})
        return response_ok({"status": "ok", "db_connected": doc is not None})
    except Exception as e:
        logger.exception("Health check DB ping failed")
        return response_error("DB unreachable", code=503)

@mcp.tool()
async def get_flight_basic_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Fetch basic flight information including carrier, flight number, date, stations, times, and status.
    
    Args:
        carrier: Airline carrier code (e.g., "6E", "AI")
        flight_number: Flight number as string (e.g., "215")
        date_of_origin: Date in YYYY-MM-DD format (e.g., "2024-06-23")
    """
    logger.info(f"get_flight_basic_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    # Normalize inputs
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    if date_of_origin and not dob:
        return response_error("Invalid date_of_origin format. Expected YYYY-MM-DD or common date formats", 400)
    
    query = make_query(carrier, fn, dob)
    
    # Project basic flight information
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.suffix": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.seqNumber": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.startStationICAO": 1,
        "flightLegState.endStationICAO": 1,
        "flightLegState.scheduledStartTime": 1,
        "flightLegState.scheduledEndTime": 1,
        "flightLegState.flightStatus": 1,
        "flightLegState.operationalStatus": 1,
        "flightLegState.flightType": 1,
        "flightLegState.blockTimeSch": 1,
        "flightLegState.blockTimeActual": 1,
        "flightLegState.flightHoursActual": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_operation_times(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Return estimated and actual operation times for a flight including takeoff, landing, block times.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_operation_times: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    if date_of_origin and not dob:
        return response_error("Invalid date format.", 400)
    
    query = make_query(carrier, fn, dob)
    
    projection = {
       
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.scheduledStartTime": 1,
        "flightLegState.scheduledEndTime": 1,
        "flightLegState.operation.estimatedTimes": 1,
        "flightLegState.operation.actualTimes": 1,
        "flightLegState.taxiOutTime": 1,
        "flightLegState.taxiInTime": 1,
        "flightLegState.blockTimeSch": 1,
        "flightLegState.blockTimeActual": 1,
        "flightLegState.flightHoursActual": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_equipment_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Get aircraft equipment details including aircraft type, registration (tail number), and configuration.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_equipment_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    query = make_query(carrier, fn, dob)
    
    projection = {
        
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.equipment.plannedAircraftType": 1,
        "flightLegState.equipment.aircraft": 1,
        "flightLegState.equipment.aircraftConfiguration": 1,
        "flightLegState.equipment.aircraftRegistration": 1,
        "flightLegState.equipment.assignedAircraftTypeIATA": 1,
        "flightLegState.equipment.assignedAircraftTypeICAO": 1,
        "flightLegState.equipment.assignedAircraftTypeIndigo": 1,
        "flightLegState.equipment.assignedAircraftConfiguration": 1,
        "flightLegState.equipment.tailLock": 1,
        "flightLegState.equipment.onwardFlight": 1,
        "flightLegState.equipment.actualOnwardFlight": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_delay_summary(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Summarize delay reasons, durations, and total delay time for a specific flight.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_delay_summary: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    query = make_query(carrier, fn, dob)
    
    projection = {
   
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.scheduledStartTime": 1,
        "flightLegState.operation.actualTimes.offBlock": 1,
        "flightLegState.delays": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_fuel_summary(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Retrieve fuel summary including planned vs actual fuel for takeoff, landing, and total consumption.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_fuel_summary: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    query = make_query(carrier, fn, dob)
    
    projection = {
       
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.operation.fuel": 1,
        "flightLegState.operation.flightPlan.offBlockFuel": 1,
        "flightLegState.operation.flightPlan.takeoffFuel": 1,
        "flightLegState.operation.flightPlan.landingFuel": 1,
        "flightLegState.operation.flightPlan.holdFuel": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_passenger_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Get passenger count and connection information for the flight.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_passenger_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    query = make_query(carrier, fn, dob)
    
    projection = {
        
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.pax": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_crew_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Get crew connections and details for the flight.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    logger.info(f"get_crew_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    query = make_query(carrier, fn, dob)
    
    projection = {
        
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.crewConnections": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def raw_mongodb_query(query_json: str, limit: int = 10) -> str:
    """
    Run a raw MongoDB query string (JSON) against collection (for debugging).
    Returns up to `limit` documents.
    
    Args:
        query_json: MongoDB query as JSON string (e.g., '{"flightLegState.carrier": "6E"}')
        limit: Maximum number of documents to return (default 10, max 50)
    """
    try:
        _, _, col = await get_mongodb_client()
        try:
            query = json.loads(query_json)
        except json.JSONDecodeError as e:
            return response_error(f"Invalid JSON query: {str(e)}. Example: '{{\"flightLegState.carrier\": \"6E\"}}'", 400)
        
        limit = min(max(1, int(limit)), 50)
        cursor = col.find(query).sort("flightLegState.dateOfOrigin", -1).limit(limit)
        docs = []
        
        async for doc in cursor:
            if "_id" in doc:
                doc.pop("_id")
            if "_class" in doc:
                doc.pop("_class")
            docs.append(doc)
        
        if not docs:
            return response_error("No documents found for given query.", 404)
        
        return response_ok({"count": len(docs), "documents": docs})
    except Exception as exc:
        logger.exception("raw_mongodb_query failed")
        return response_error(f"raw query failed: {str(exc)}", 500)

# --- Run MCP Server ---
if __name__ == "__main__":
    logger.info("Starting FlightOps MCP Server on %s:%s (transport=%s)", HOST, PORT, TRANSPORT)
    logger.info("MongoDB URL: %s, Database: %s, Collection: %s", MONGODB_URL, DATABASE_NAME, COLLECTION_NAME)
    mcp.run(transport="streamable-http")






# tool_registry.py

TOOLS = {
    "get_flight_basic_info": {
        "args": ["carrier", "flight_number", "date_of_origin"],
        "desc": "Fetch basic flight information including carrier, flight number, stations, scheduled times, and flight status.",
    },
    "get_equipment_info": {
        "args": ["carrier", "flight_number", "date_of_origin"],
        "desc": "Get aircraft equipment details: aircraft type, tail number (registration), and configuration.",
    },
    "get_operation_times": {
        "args": ["carrier", "flight_number", "date_of_origin"],
        "desc": "Return estimated and actual operation times: takeoff, landing, departure, arrival, and block times.",
    },
    "get_fuel_summary": {
        "args": ["carrier", "flight_number", "date_of_origin"],
        "desc": "Retrieve fuel summary including planned vs actual fuel consumption for the flight.",
    },
    "get_delay_summary": {
        "args": ["carrier", "flight_number", "date_of_origin"],
        "desc": "Get delay information including delay reasons, durations, and total delay time.",
    },
    "health_check": {
        "args": [],
        "desc": "Check the health status of the MCP server and database connection.",
    },
    "raw_mongodb_query": {
        "args": ["query_json", "limit"],
        "desc": "Run a raw MongoDB query (JSON format) for debugging purposes.",
    },
}



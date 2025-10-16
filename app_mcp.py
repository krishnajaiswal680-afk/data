schema of fight data



_id
"6E#2024-06-23#215#SXR#BOM"

flightLegState
Object
carrier
"6E"
dateOfOrigin
"2024-06-23"
flightNumber
215
suffix
""
seqNumber
2
startStation
"SXR"
endStation
"BOM"
scheduledStartTime
"2024-06-23T14:30:00Z"
scheduledEndTime
"2024-06-23T17:35:00Z"
endTerminal
"2"
operationalStatus
"S"

returnEvents
Object

returnEvent
Array (empty)

handling
Object
serviceType
"J"
aircraftOwner
"6E"
cockpitEmployer
"6E"
cabinEmployer
"6E"

equipment
Object
plannedAircraftType
"320"

aircraft
Object
registration
"VTIVY"
type
"32S"
aircraftConfiguration
"Y186VV10"

onwardFlight
Object
carrier
"6E"
flightNumber
911
suffix
""
tailLock
false
assignedAircraftTypeIATA
"32N"
assignedAircraftTypeICAO
"A320"
assignedAircraftTypeIndigo
"320"
plannedaircraftTypeICAO
"A320"
plannedaircraftTypeIndigo
"320"
assignedAircraftConfiguration
"Y186"
aircraftRegistration
"VTIVY"

actualOnwardFlight
Object
carrier
"6E"
flightNumber
911
suffix
""
startStation
"BOM"
endStation
"AMD"
scheduledStartTime
"2024-06-23T19:15:00Z"
scheduledEndTime
"2024-06-23T20:30:00Z"

pax
Object

passengerCount
Array (9)

0
Object
code
"CheckInCount"
count
187

1
Object
code
"CheckedInTotalChildCount"
count
9

2
Object
code
"CheckedInTotalFemaleCount"
count
75

3
Object
code
"CheckedInTotalMaleCount"
count
103

4
Object
code
"DefaultTotalChildCount"
count
1

5
Object
code
"DefaultTotalFemaleCount"
count
4

6
Object

7
Object

8
Object

nextLeg
Array (6)

0
Object

1
Object

2
Object

3
Object

4
Object

5
Object

operation
Object

estimatedTimes
Object
offBlock
"2024-06-23T15:25:00Z"
inBlock
"2024-06-23T18:21:00Z"
takeoffTime
"2024-06-23T15:38:00Z"
landingTime
"2024-06-23T18:07:00Z"

actualTimes
Object
offBlock
"2024-06-23T15:23:00Z"
inBlock
"2024-06-23T18:22:00Z"
takeoffTime
"2024-06-23T15:37:00Z"
landingTime
"2024-06-23T18:16:00Z"
doorClose
"2024-06-23T15:22:00Z"

estimatedDepSlot
Object
isSlotCancelled
"false"

fuel
Object
offBlock
10700
takeoff
10600
landing
4400
inBlock
4300
autoland
false

flightPlan
Object
estimatedElapsedTime
"PT2H30M"
acTakeoffWeight
72340
offBlockFuel
10500
takeoffFuel
10300
landingFuel
5125
holdFuel
2059
holdTime
"PT57M"
routeDistance
951

alternates
Object

departure
Array (empty)

intermediate
Array (empty)

arrival
Array (1)

delays
Object

delay
Array (2)
total
"PT53M"

training
Object

trainingTag
Array (empty)

annotations
Object

annotation
Array (empty)

codeShares
Object

codeShare
Array (1)

0
Object
flightStatus
""
startStationICAO
"VISR"
endStationICAO
"VABB"
startCountry
"IN"
endCountry
"IN"
Show 17 more fields in flightLegState
_class
"com.indigo.nosql.document.FlightDocument"





_id
"6E#2024-06-23#6395#IXC#AMD"

flightLegState
Object
_class
"com.indigo.nosql.document.FlightDocument"











app.py







from typing import Any, List, Dict, Optional
import asyncio
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
from motor.motor_asyncio import AsyncIOMotorClient
import json

# Initialize FastMCP server
mcp = FastMCP("metar-weather")

# MongoDB configuration
import os
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "metar_data")
COLLECTION_METAR = os.getenv("COLLECTION_METAR", "metar_data")

# Global MongoDB client
client = None
db = None

async def get_mongodb_client():
    """Get MongoDB client connection."""
    global client, db
    if client is None:
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client[DATABASE_NAME]
    return client, db

def format_metar_data(metar_doc: Dict) -> str:
    """Format METAR data into a readable string."""
    station = metar_doc.get('stationICAO', 'Unknown')
    iata = metar_doc.get('stationIATA', 'N/A')
    timestamp = metar_doc.get('timestamp', 'Unknown')
    
    result = f"🛩️  Station: {station}"
    if iata:
        result += f" ({iata})"
    result += f"\n Last Updated: {timestamp}\n"
    
    if metar_doc.get('hasMetarData') and 'metar' in metar_doc:
        metar = metar_doc['metar']
        raw_data = metar.get('rawData', 'N/A')
        result += f" Raw METAR: {raw_data}\n"
        
        if 'decodedData' in metar and 'observation' in metar['decodedData']:
            obs = metar['decodedData']['observation']
            result += f"\n Weather Conditions:\n"
            result += f"   Temperature: {obs.get('airTemperature', 'N/A')}\n"
            result += f"   Dewpoint: {obs.get('dewpointTemperature', 'N/A')}\n"
            result += f"   Wind: {obs.get('windSpeed', 'N/A')} from {obs.get('windDirection', 'N/A')}\n"
            result += f"   Visibility: {obs.get('horizontalVisibility', 'N/A')}\n"
            result += f"   Pressure: {obs.get('observedQNH', 'N/A')}\n"
            
            if obs.get('cloudLayers'):
                result += f"   Clouds: {', '.join(obs['cloudLayers'])}\n"
            
            if obs.get('weatherConditions'):
                result += f"   Weather: {obs['weatherConditions']}\n"
    
    if metar_doc.get('hasTaforData') and 'tafor' in metar_doc:
        tafor = metar_doc['tafor']
        raw_taf = tafor.get('rawData', 'N/A')
        result += f"\n📊 TAF: {raw_taf}\n"
    
    return result

@mcp.tool()
async def search_metar_data(
    station_icao: str = None,
    station_iata: str = None,
    weather_condition: str = None,
    temperature_min: float = None,
    temperature_max: float = None,
    visibility_min: int = None,
    visibility_max: int = None,
    wind_speed_min: float = None,
    wind_speed_max: float = None,
    pressure_min: float = None,
    pressure_max: float = None,
    cloud_type: str = None,
    fir_region: str = None,
    hours_back: int = None,
    limit: int = 10
) -> str:
    """Generic search for METAR data with multiple optional filters.

    Args:
        station_icao: Filter by ICAO code (e.g., 'VOTP')
        station_iata: Filter by IATA code (e.g., 'TIR')
        weather_condition: Search in raw METAR data (e.g., 'rain', 'fog', 'CB')
        temperature_min: Minimum temperature in Celsius
        temperature_max: Maximum temperature in Celsius
        visibility_min: Minimum visibility in meters
        visibility_max: Maximum visibility in meters
        wind_speed_min: Minimum wind speed in m/s
        wind_speed_max: Maximum wind speed in m/s
        pressure_min: Minimum pressure in hPa
        pressure_max: Maximum pressure in hPa
        cloud_type: Search for cloud types in raw data (e.g., 'CB', 'SCT', 'OVC')
        fir_region: Filter by FIR region (e.g., 'Chennai', 'Mumbai')
        hours_back: Look back N hours from now
        limit: Maximum results to return (default: 10, max: 50)
    """
    try:
        _, db = await get_mongodb_client()
        
        # Build the query
        query = {}
        
        # Station filters
        if station_icao:
            query["stationICAO"] = station_icao.upper()
        if station_iata:
            query["stationIATA"] = station_iata.upper()
        
        # FIR region filter
        if fir_region:
            query["metar.firRegion"] = {"$regex": fir_region, "$options": "i"}
        
        # Time filter
        if hours_back:
            time_threshold = datetime.now() - timedelta(hours=hours_back)
            query["timestamp"] = {"$gte": time_threshold}
        
        # Weather condition filter (search in raw METAR data)
        if weather_condition:
            query["metar.rawData"] = {"$regex": weather_condition, "$options": "i"}
        
        # Cloud type filter (search in raw METAR data)
        if cloud_type:
            query["metar.rawData"] = {"$regex": cloud_type, "$options": "i"}
        
        # Temperature filters
        if temperature_min is not None or temperature_max is not None:
            temp_query = {}
            if temperature_min is not None:
                temp_query["$gte"] = str(temperature_min)
            if temperature_max is not None:
                temp_query["$lte"] = str(temperature_max)
            query["metar.decodedData.observation.airTemperature"] = temp_query
        
        # Visibility filters
        if visibility_min is not None or visibility_max is not None:
            vis_query = {}
            if visibility_min is not None:
                vis_query["$gte"] = str(visibility_min)
            if visibility_max is not None:
                vis_query["$lte"] = str(visibility_max)
            query["metar.decodedData.observation.horizontalVisibility"] = vis_query
        
        # Wind speed filters
        if wind_speed_min is not None or wind_speed_max is not None:
            wind_query = {}
            if wind_speed_min is not None:
                wind_query["$gte"] = str(wind_speed_min)
            if wind_speed_max is not None:
                wind_query["$lte"] = str(wind_speed_max)
            query["metar.decodedData.observation.windSpeed"] = wind_query
        
        # Pressure filters
        if pressure_min is not None or pressure_max is not None:
            pressure_query = {}
            if pressure_min is not None:
                pressure_query["$gte"] = str(pressure_min)
            if pressure_max is not None:
                pressure_query["$lte"] = str(pressure_max)
            query["metar.decodedData.observation.observedQNH"] = pressure_query
        
        # Limit results
        limit = min(limit, 50)
        
        # Execute the query
        cursor = db[COLLECTION_METAR].find(query).sort("timestamp", -1).limit(limit)
        results = await cursor.to_list(length=limit)
        
        if not results:
            filters = []
            if station_icao: filters.append(f"ICAO: {station_icao}")
            if station_iata: filters.append(f"IATA: {station_iata}")
            if weather_condition: filters.append(f"Weather: {weather_condition}")
            if temperature_min: filters.append(f"Temp ≥ {temperature_min}°C")
            if temperature_max: filters.append(f"Temp ≤ {temperature_max}°C")
            if visibility_min: filters.append(f"Visibility ≥ {visibility_min}m")
            if visibility_max: filters.append(f"Visibility ≤ {visibility_max}m")
            if wind_speed_min: filters.append(f"Wind ≥ {wind_speed_min} m/s")
            if wind_speed_max: filters.append(f"Wind ≤ {wind_speed_max} m/s")
            if pressure_min: filters.append(f"Pressure ≥ {pressure_min} hPa")
            if pressure_max: filters.append(f"Pressure ≤ {pressure_max} hPa")
            if cloud_type: filters.append(f"Cloud: {cloud_type}")
            if fir_region: filters.append(f"FIR: {fir_region}")
            if hours_back: filters.append(f"Last {hours_back}h")
            
            return f"No METAR data found with filters: {', '.join(filters)}"
        
        # Format results
        result = f"🔍 METAR Search Results ({len(results)} documents found):\n"
        applied_filters = [f"{k}: {v}" for k, v in locals().items() if v is not None and k not in ['db', 'cursor', 'results', 'limit', 'hours_back', 'query']]
        if applied_filters:
            result += f"Filters: {', '.join(applied_filters)}\n"
        result += "=" * 80 + "\n\n"
        
        for i, doc in enumerate(results, 1):
            result += f"--- Result {i} ---\n"
            result += format_metar_data(doc)
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error executing search: {str(e)}"

@mcp.tool()
async def list_available_stations() -> str:
    """List all available weather stations with their codes.

    Returns:
        String containing all available ICAO and IATA codes
    """
    try:
        _, db = await get_mongodb_client()
        
        # Get unique ICAO codes
        icao_codes = await db[COLLECTION_METAR].distinct("stationICAO")
        icao_codes.sort()
        
        # Get unique IATA codes (non-null)
        iata_codes = await db[COLLECTION_METAR].distinct("stationIATA")
        iata_codes = [code for code in iata_codes if code is not None]
        iata_codes.sort()
        
        # Get station count
        total_stations = await db[COLLECTION_METAR].count_documents({})
        
        result = f" Available Weather Stations ({total_stations} total reports)\n"
        result += "=" * 50 + "\n\n"
        
        result += f"  ICAO Codes ({len(icao_codes)} stations):\n"
        for i, code in enumerate(icao_codes, 1):
            result += f"   {i:3d}. {code}"
            if i % 10 == 0:  # New line every 10 codes
                result += "\n"
            else:
                result += "  "
        if len(icao_codes) % 10 != 0:
            result += "\n"
        
        result += f"\nIATA Codes ({len(iata_codes)} stations):\n"
        for i, code in enumerate(iata_codes, 1):
            result += f"   {i:3d}. {code}"
            if i % 10 == 0:  # New line every 10 codes
                result += "\n"
            else:
                result += "  "
        if len(iata_codes) % 10 != 0:
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving station list: {str(e)}"

@mcp.tool()
async def get_metar_statistics() -> str:
    """Get statistics about the METAR database.

    Returns:
        String containing database statistics
    """
    try:
        _, db = await get_mongodb_client()
        
        # Get basic counts
        total_metar = await db[COLLECTION_METAR].count_documents({})
        
        # Get unique station counts
        unique_icao = len(await db[COLLECTION_METAR].distinct("stationICAO"))
        unique_iata = len([code for code in await db[COLLECTION_METAR].distinct("stationIATA") if code is not None])
        
        # Get date range
        earliest = await db[COLLECTION_METAR].find({}, {"timestamp": 1}).sort("timestamp", 1).limit(1).to_list(1)
        latest = await db[COLLECTION_METAR].find({}, {"timestamp": 1}).sort("timestamp", -1).limit(1).to_list(1)
        
        # Get data availability
        with_metar = await db[COLLECTION_METAR].count_documents({"hasMetarData": True})
        with_taf = await db[COLLECTION_METAR].count_documents({"hasTaforData": True})
        
        result = f" METAR Database Statistics\n"
        result += "=" * 40 + "\n\n"
        
        result += f"Document Counts:\n"
        result += f"   METAR Reports: {total_metar:,}\n\n"
        
        result += f" Station Information:\n"
        result += f"   Unique ICAO Codes: {unique_icao}\n"
        result += f"   Unique IATA Codes: {unique_iata}\n\n"
        
        result += f" Data Range:\n"
        if earliest:
            result += f"   Earliest: {earliest[0]['timestamp']}\n"
        if latest:
            result += f"   Latest: {latest[0]['timestamp']}\n\n"
        
        result += f"  Availability:\n"
        result += f"   Reports with METAR: {with_metar:,} ({with_metar/total_metar*100:.1f}%)\n"
        result += f"   Reports with TAF: {with_taf:,} ({with_taf/total_metar*100:.1f}%)\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving statistics: {str(e)}"

@mcp.tool()
async def raw_mongodb_query(query_json: str, limit: int = 10) -> str:
    """Execute a raw MongoDB query against the METAR database.

    Args:
        query_json: JSON string containing MongoDB query (e.g., '{"stationICAO": "VOTP"}')
        limit: Maximum number of results to return (default: 10, max: 50)
    """
    try:
        _, db = await get_mongodb_client()
        
        # Parse the query JSON
        try:
            query = json.loads(query_json)
        except json.JSONDecodeError as e:
            return f"Invalid JSON query: {str(e)}\n\nExample: '{{\"stationICAO\": \"VOTP\"}}'"
        
        # Limit the number of results
        limit = min(limit, 50)  # Cap at 50 results
        
        # Execute the query
        cursor = db[COLLECTION_METAR].find(query).sort("timestamp", -1).limit(limit)
        results = await cursor.to_list(length=limit)
        
        if not results:
            return f"No documents found matching query: {query_json}"
        
        # Format results
        result = f"🔍 Raw MongoDB Query Results ({len(results)} documents found):\n"
        result += f"Query: {query_json}\n"
        result += "=" * 60 + "\n\n"
        
        for i, doc in enumerate(results, 1):
            result += f"--- Result {i} ---\n"
            result += format_metar_data(doc)
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error executing query: {str(e)}"

@mcp.tool() 
async def health_check(): 
    return "OK"

import time

if __name__ == "__main__":
    # Initialize and run the server
    print("METAR MCP Server starting...")
    print(f"MongoDB URL: {MONGODB_URL}")
    print(f"Database: {DATABASE_NAME}")
    print(f"Collection: {COLLECTION_METAR}")
    print("Server ready! Waiting for MCP protocol messages...")
    mcp.run(transport='stdio')

    # If mcp.run() returns immediately, keep the process alive with an infinite sleep loop
    while True:
        time.sleep(60)

# http_app.py


from typing import Any, List, Dict, Optional
import asyncio
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
from motor.motor_asyncio import AsyncIOMotorClient
import json

# Initialize FastMCP server
mcp = FastMCP("metar-weather")

# MongoDB configuration
import os
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "metar_data")
COLLECTION_METAR = os.getenv("COLLECTION_METAR", "metar_data")

# Global MongoDB client
client = None
db = None

async def get_mongodb_client():
    """Get MongoDB client connection."""
    global client, db
    if client is None:
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client[DATABASE_NAME]
    return client, db

def format_metar_data(metar_doc: Dict) -> str:
    """Format METAR data into a readable string."""
    station = metar_doc.get('stationICAO', 'Unknown')
    iata = metar_doc.get('stationIATA', 'N/A')
    timestamp = metar_doc.get('timestamp', 'Unknown')
    
    result = f"🛩️  Station: {station}"
    if iata:
        result += f" ({iata})"
    result += f"\n Last Updated: {timestamp}\n"
    
    if metar_doc.get('hasMetarData') and 'metar' in metar_doc:
        metar = metar_doc['metar']
        raw_data = metar.get('rawData', 'N/A')
        result += f" Raw METAR: {raw_data}\n"
        
        if 'decodedData' in metar and 'observation' in metar['decodedData']:
            obs = metar['decodedData']['observation']
            result += f"\n Weather Conditions:\n"
            result += f"   Temperature: {obs.get('airTemperature', 'N/A')}\n"
            result += f"   Dewpoint: {obs.get('dewpointTemperature', 'N/A')}\n"
            result += f"   Wind: {obs.get('windSpeed', 'N/A')} from {obs.get('windDirection', 'N/A')}\n"
            result += f"   Visibility: {obs.get('horizontalVisibility', 'N/A')}\n"
            result += f"   Pressure: {obs.get('observedQNH', 'N/A')}\n"
            
            if obs.get('cloudLayers'):
                result += f"   Clouds: {', '.join(obs['cloudLayers'])}\n"
            
            if obs.get('weatherConditions'):
                result += f"   Weather: {obs['weatherConditions']}\n"
    
    if metar_doc.get('hasTaforData') and 'tafor' in metar_doc:
        tafor = metar_doc['tafor']
        raw_taf = tafor.get('rawData', 'N/A')
        result += f"\nTAF: {raw_taf}\n"
    
    return result

@mcp.tool()
async def search_metar_data(
    station_icao: str = None,
    station_iata: str = None,
    weather_condition: str = None,
    temperature_min: float = None,
    temperature_max: float = None,
    visibility_min: int = None,
    visibility_max: int = None,
    wind_speed_min: float = None,
    wind_speed_max: float = None,
    pressure_min: float = None,
    pressure_max: float = None,
    cloud_type: str = None,
    fir_region: str = None,
    hours_back: int = None,
    limit: int = 10
) -> str:
    """Generic search for METAR data with multiple optional filters.

    Args:
        station_icao: Filter by ICAO code (e.g., 'VOTP')
        station_iata: Filter by IATA code (e.g., 'TIR')
        weather_condition: Search in raw METAR data (e.g., 'rain', 'fog', 'CB')
        temperature_min: Minimum temperature in Celsius
        temperature_max: Maximum temperature in Celsius
        visibility_min: Minimum visibility in meters
        visibility_max: Maximum visibility in meters
        wind_speed_min: Minimum wind speed in m/s
        wind_speed_max: Maximum wind speed in m/s
        pressure_min: Minimum pressure in hPa
        pressure_max: Maximum pressure in hPa
        cloud_type: Search for cloud types in raw data (e.g., 'CB', 'SCT', 'OVC')
        fir_region: Filter by FIR region (e.g., 'Chennai', 'Mumbai')
        hours_back: Look back N hours from now
        limit: Maximum results to return (default: 10, max: 50)
    """
    try:
        _, db = await get_mongodb_client()
        
        # Build the query
        query = {}
        
        # Station filters
        if station_icao:
            query["stationICAO"] = station_icao.upper()
        if station_iata:
            query["stationIATA"] = station_iata.upper()
        
        # FIR region filter
        if fir_region:
            query["metar.firRegion"] = {"$regex": fir_region, "$options": "i"}
        
        # Time filter
        if hours_back:
            time_threshold = datetime.now() - timedelta(hours=hours_back)
            query["timestamp"] = {"$gte": time_threshold}
        
        # Weather condition filter (search in raw METAR data)
        if weather_condition:
            query["metar.rawData"] = {"$regex": weather_condition, "$options": "i"}
        
        # Cloud type filter (search in raw METAR data)
        if cloud_type:
            query["metar.rawData"] = {"$regex": cloud_type, "$options": "i"}
        
        # Temperature filters
        if temperature_min is not None or temperature_max is not None:
            temp_query = {}
            if temperature_min is not None:
                temp_query["$gte"] = str(temperature_min)
            if temperature_max is not None:
                temp_query["$lte"] = str(temperature_max)
            query["metar.decodedData.observation.airTemperature"] = temp_query
        
        # Visibility filters
        if visibility_min is not None or visibility_max is not None:
            vis_query = {}
            if visibility_min is not None:
                vis_query["$gte"] = str(visibility_min)
            if visibility_max is not None:
                vis_query["$lte"] = str(visibility_max)
            query["metar.decodedData.observation.horizontalVisibility"] = vis_query
        
        # Wind speed filters
        if wind_speed_min is not None or wind_speed_max is not None:
            wind_query = {}
            if wind_speed_min is not None:
                wind_query["$gte"] = str(wind_speed_min)
            if wind_speed_max is not None:
                wind_query["$lte"] = str(wind_speed_max)
            query["metar.decodedData.observation.windSpeed"] = wind_query
        
        # Pressure filters
        if pressure_min is not None or pressure_max is not None:
            pressure_query = {}
            if pressure_min is not None:
                pressure_query["$gte"] = str(pressure_min)
            if pressure_max is not None:
                pressure_query["$lte"] = str(pressure_max)
            query["metar.decodedData.observation.observedQNH"] = pressure_query
        
        # Limit results
        limit = min(limit, 50)
        
        # Execute the query
        cursor = db[COLLECTION_METAR].find(query).sort("timestamp", -1).limit(limit)
        results = await cursor.to_list(length=limit)
        
        if not results:
            filters = []
            if station_icao: filters.append(f"ICAO: {station_icao}")
            if station_iata: filters.append(f"IATA: {station_iata}")
            if weather_condition: filters.append(f"Weather: {weather_condition}")
            if temperature_min: filters.append(f"Temp ≥ {temperature_min}°C")
            if temperature_max: filters.append(f"Temp ≤ {temperature_max}°C")
            if visibility_min: filters.append(f"Visibility ≥ {visibility_min}m")
            if visibility_max: filters.append(f"Visibility ≤ {visibility_max}m")
            if wind_speed_min: filters.append(f"Wind ≥ {wind_speed_min} m/s")
            if wind_speed_max: filters.append(f"Wind ≤ {wind_speed_max} m/s")
            if pressure_min: filters.append(f"Pressure ≥ {pressure_min} hPa")
            if pressure_max: filters.append(f"Pressure ≤ {pressure_max} hPa")
            if cloud_type: filters.append(f"Cloud: {cloud_type}")
            if fir_region: filters.append(f"FIR: {fir_region}")
            if hours_back: filters.append(f"Last {hours_back}h")
            
            return f"No METAR data found with filters: {', '.join(filters)}"
        
        # Format results
        result = f"🔍 METAR Search Results ({len(results)} documents found):\n"
        applied_filters = [f"{k}: {v}" for k, v in locals().items() if v is not None and k not in ['db', 'cursor', 'results', 'limit', 'hours_back', 'query']]
        if applied_filters:
            result += f"Filters: {', '.join(applied_filters)}\n"
        result += "=" * 80 + "\n\n"
        
        for i, doc in enumerate(results, 1):
            result += f"--- Result {i} ---\n"
            result += format_metar_data(doc)
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error executing search: {str(e)}"

@mcp.tool()
async def list_available_stations() -> str:
    """List all available weather stations with their codes.

    Returns:
        String containing all available ICAO and IATA codes
    """
    try:
        _, db = await get_mongodb_client()
        
        # Get unique ICAO codes
        icao_codes = await db[COLLECTION_METAR].distinct("stationICAO")
        icao_codes.sort()
        
        # Get unique IATA codes (non-null)
        iata_codes = await db[COLLECTION_METAR].distinct("stationIATA")
        iata_codes = [code for code in iata_codes if code is not None]
        iata_codes.sort()
        
        # Get station count
        total_stations = await db[COLLECTION_METAR].count_documents({})
        
        result = f" Available Weather Stations ({total_stations} total reports)\n"
        result += "=" * 50 + "\n\n"
        
        result += f"  ICAO Codes ({len(icao_codes)} stations):\n"
        for i, code in enumerate(icao_codes, 1):
            result += f"   {i:3d}. {code}"
            if i % 10 == 0:  # New line every 10 codes
                result += "\n"
            else:
                result += "  "
        if len(icao_codes) % 10 != 0:
            result += "\n"
        
        result += f"\nIATA Codes ({len(iata_codes)} stations):\n"
        for i, code in enumerate(iata_codes, 1):
            result += f"   {i:3d}. {code}"
            if i % 10 == 0:  # New line every 10 codes
                result += "\n"
            else:
                result += "  "
        if len(iata_codes) % 10 != 0:
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving station list: {str(e)}"

@mcp.tool()
async def get_metar_statistics() -> str:
    """Get statistics about the METAR database.

    Returns:
        String containing database statistics
    """
    try:
        _, db = await get_mongodb_client()
        
        # Get basic counts
        total_metar = await db[COLLECTION_METAR].count_documents({})
        
        # Get unique station counts
        unique_icao = len(await db[COLLECTION_METAR].distinct("stationICAO"))
        unique_iata = len([code for code in await db[COLLECTION_METAR].distinct("stationIATA") if code is not None])
        
        # Get date range
        earliest = await db[COLLECTION_METAR].find({}, {"timestamp": 1}).sort("timestamp", 1).limit(1).to_list(1)
        latest = await db[COLLECTION_METAR].find({}, {"timestamp": 1}).sort("timestamp", -1).limit(1).to_list(1)
        
        # Get data availability
        with_metar = await db[COLLECTION_METAR].count_documents({"hasMetarData": True})
        with_taf = await db[COLLECTION_METAR].count_documents({"hasTaforData": True})
        
        result = f" METAR Database Statistics\n"
        result += "=" * 40 + "\n\n"
        
        result += f"Document Counts:\n"
        result += f"   METAR Reports: {total_metar:,}\n\n"
        
        result += f" Station Information:\n"
        result += f"   Unique ICAO Codes: {unique_icao}\n"
        result += f"   Unique IATA Codes: {unique_iata}\n\n"
        
        result += f" Data Range:\n"
        if earliest:
            result += f"   Earliest: {earliest[0]['timestamp']}\n"
        if latest:
            result += f"   Latest: {latest[0]['timestamp']}\n\n"
        
        result += f"  Availability:\n"
        result += f"   Reports with METAR: {with_metar:,} ({with_metar/total_metar*100:.1f}%)\n"
        result += f"   Reports with TAF: {with_taf:,} ({with_taf/total_metar*100:.1f}%)\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving statistics: {str(e)}"

@mcp.tool()
async def raw_mongodb_query(query_json: str, limit: int = 10) -> str:
    """Execute a raw MongoDB query against the METAR database.

    Args:
        query_json: JSON string containing MongoDB query (e.g., '{"stationICAO": "VOTP"}')
        limit: Maximum number of results to return (default: 10, max: 50)
    """
    try:
        _, db = await get_mongodb_client()
        
        # Parse the query JSON
        try:
            query = json.loads(query_json)
        except json.JSONDecodeError as e:
            return f"Invalid JSON query: {str(e)}\n\nExample: '{{\"stationICAO\": \"VOTP\"}}'"
        
        # Limit the number of results
        limit = min(limit, 50)  # Cap at 50 results
        
        # Execute the query
        cursor = db[COLLECTION_METAR].find(query).sort("timestamp", -1).limit(limit)
        results = await cursor.to_list(length=limit)
        
        if not results:
            return f"No documents found matching query: {query_json}"
        
        # Format results
        result = f"Raw MongoDB Query Results ({len(results)} documents found):\n"
        result += f"Query: {query_json}\n"
        result += "=" * 60 + "\n\n"
        
        for i, doc in enumerate(results, 1):
            result += f"--- Result {i} ---\n"
            result += format_metar_data(doc)
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error executing query: {str(e)}"

@mcp.tool() 
async def health_check(): 
    return "OK"

if __name__ == "__main__":
    # Initialize and run the server with HTTP transport
    print("METAR MCP Server starting with HTTP transport...")
    print(f"MongoDB URL: {MONGODB_URL}")
    print(f"Database: {DATABASE_NAME}")
    print(f"Collection: {COLLECTION_METAR}")
    print("Server ready! Waiting for HTTP requests...")
    mcp.run(transport='streamable-http')


import httpx # Changed from requests
import json
import os
import asyncio # Import asyncio for async operations
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Body, Header, Depends, status, Query
from pydantic import BaseModel, Field
from fastapi.responses import Response
from fastapi.security import APIKeyHeader

# Load environment variables
load_dotenv()

# ---
## Data Models & State (No Change)
# ---

@dataclass
class HappyHourState:
    """State shared between agents"""
    user_locations: List[str] = None
    coordinates_list: List[Dict[str, float]] = None
    location_details_list: List[Dict] = None
    venues_found: List[Dict] = None
    whistle_alerts_created: List[Dict] = None
    tavily_search_results: List[Dict] = None
    search_radius_km: float = 5.0
    alert_radius_km: float = 1.0
    max_venues_per_location: int = 15
    create_whistle_alerts: bool = False
    error_messages: List[str] = None
    processing_step: str = "initialized"
    
    def __post_init__(self):
        if self.user_locations is None:
            self.user_locations = []
        if self.coordinates_list is None:
            self.coordinates_list = []
        if self.location_details_list is None:
            self.location_details_list = []
        if self.venues_found is None:
            self.venues_found = []
        if self.whistle_alerts_created is None:
            self.whistle_alerts_created = []
        if self.tavily_search_results is None:
            self.tavily_search_results = []
        if self.error_messages is None:
            self.error_messages = []

@dataclass
class VenueOffer:
    """Structured venue offer data"""
    name: str
    latitude: float
    longitude: float
    venue_type: str
    happy_hour_details: str
    distance_km: float
    offer_valid_until: str
    tags: List[str]
    location_source: str

# ---
## Utility Services (Modified for Async)
# ---

class LocationService:
    """OpenStreetMap location services"""
    
    @staticmethod
    async def geocode_location(location_input: str) -> Optional[Dict]:
        """Get coordinates from location string"""
        try:
            # Nominatim geocoding is synchronous, so we run it in a thread pool
            # to avoid blocking the event loop. This is the correct way to call
            # synchronous code from an async function.
            geocoder = Nominatim(user_agent="happy_hour_finder_v2")
            location = await asyncio.to_thread(geocoder.geocode, location_input, timeout=10)
            
            if not location:
                return None
            return {
                "lat": location.latitude,
                "lng": location.longitude,
                "full_address": location.address,
                "city": location.address.split(',')[0] if ',' in location.address else location_input,
                "country": location.address.split(',')[-1].strip() if ',' in location.address else "Unknown"
            }
        except Exception as e:
            print(f"Geocoding error for {location_input}: {e}")
            return None
    
    @staticmethod
    async def find_venues_osm(lat: float, lng: float, radius_km: float = 5.0, max_venues: int = 15) -> List[Dict]:
        """Find venues using OpenStreetMap Overpass API"""
        radius_meters = int(radius_km * 1000)
        overpass_query = f"""
        [out:json][timeout:30];
        (
          node["amenity"~"^(bar|pub|restaurant|cafe|nightclub|biergarten)$"](around:{radius_meters},{lat},{lng});
          way["amenity"~"^(bar|pub|restaurant|cafe|nightclub|biergarten)$"](around:{radius_meters},{lat},{lng});
          node["leisure"="adult_gaming_centre"](around:{radius_meters},{lat},{lng});
          node["shop"="alcohol"](around:{radius_meters},{lat},{lng});
        );
        out center tags;
        """
        try:
            async with httpx.AsyncClient() as client: # Use AsyncClient
                response = await client.post("https://overpass-api.de/api/interpreter", data=overpass_query, timeout=30)
                response.raise_for_status()
            data = response.json()
            venues = []
            processed_names = set()
            
            for element in data.get('elements', []):
                tags = element.get('tags', {})
                name = tags.get('name')
                if not name or name in processed_names:
                    continue
                processed_names.add(name)
                
                # Get coordinates
                if element['type'] == 'node':
                    venue_lat = element['lat']
                    venue_lng = element['lon']
                elif element['type'] == 'way' and 'center' in element:
                    venue_lat = element['center']['lat']
                    venue_lng = element['center']['lon']
                else:
                    continue
                
                # Calculate distance - geodesic is synchronous, so no await here
                distance = geodesic((lat, lng), (venue_lat, venue_lng)).kilometers
                if distance > radius_km:
                    continue
                
                # Check amenity type
                amenity = tags.get('amenity', tags.get('leisure', tags.get('shop', 'venue')))
                if amenity not in ['bar', 'pub', 'restaurant', 'nightclub', 'biergarten', 'cafe']:
                    continue
                
                venues.append({
                    "name": name,
                    "latitude": round(venue_lat, 6),
                    "longitude": round(venue_lng, 6),
                    "venue_type": amenity,
                    "distance_km": round(distance, 2),
                    "tags": tags,
                    "source": "openstreetmap"
                })
            
            # Sort by distance and limit results
            sorted_venues = sorted(venues, key=lambda x: x['distance_km'])
            return sorted_venues[:max_venues]
        except httpx.RequestError as e: # Catch httpx specific exceptions
            print(f"Error fetching OSM data: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during OSM data processing: {e}")
            return []


class TavilyAPIService:
    """Tavily API integration service for additional search"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    async def search(self, query: str, max_results: int = 2) -> List[Dict]:
        """Perform a search using Tavily API"""
        if not self.api_key:
            print("Tavily API key not set.")
            return []
        
        headers = {"Content-Type": "application/json"}
        data = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": max_results
        }
        
        try:
            async with httpx.AsyncClient() as client: # Use AsyncClient
                response = await client.post(self.base_url, headers=headers, json=data, timeout=15)
                response.raise_for_status()
            return response.json().get('results', [])
        except httpx.RequestError as e: # Catch httpx specific exceptions
            print(f"Tavily search error: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during Tavily search: {e}")
            return []

class WhistleAPIService:
    """Whistle API integration service"""

    def __init__(self, api_key: str, alert_radius_km: float = 1.0):
        self.api_key = os.getenv("WHISTLE_API_KEY") # Ensure this is read from env
        self.alert_radius_km = alert_radius_km
        self.base_url = "http://dowhistle.herokuapp.com/v3/whistle"
        # headers are now created per request or passed as a parameter for dynamic usage
    
    async def create_alert(self, venue_data: Dict) -> Dict:
        """Create a Whistle alert for a happy hour venue"""
        if not self.api_key or self.api_key == "":
            print("❌ Whistle API Key is missing or is the placeholder. Cannot create alert.")
            return {"error": "Whistle API Key not configured or is placeholder.", "success": False}

        headers = { # Define headers here or pass via init if constant
            "Authorization": f"{self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            combined_tags = list(set([] + venue_data.get('tags', [])))
            expiry_time = datetime.now() + timedelta(hours=24)
            expiry_isoformat = expiry_time.isoformat()
            
            alert_payload = {
                "whistle": {
                    "provider": True,
                    "tags": combined_tags, 
                    "alertRadius": self.alert_radius_km,
                    "description": f"Happy Hour at {venue_data['name']}: {venue_data.get('happy_hour_details', 'Special offers available')}",
                    "expiry": expiry_isoformat, 
                    "latitude": venue_data['latitude'],
                    "longitude": venue_data['longitude']
                }
            }
            
            print(f"Sending alert payload to Whistle API: {json.dumps(alert_payload, indent=2)}")

            async with httpx.AsyncClient() as client: # Use AsyncClient
                response = await client.post(self.base_url, headers=headers, json=alert_payload, timeout=15)
                response.raise_for_status()
            
            response_data = response.json()
            
            if response.status_code >= 200 and response.status_code < 300: # Check for success range
                alert_id = response_data.get('id') or response_data.get('alertId') or response_data.get('whistleId', 'created_successfully')
                return {
                    "success": True,
                    "alert_id": alert_id,
                    "message": response_data.get('message', 'Alert created successfully'),
                    "expiry": expiry_isoformat, 
                    "created_at": datetime.now().isoformat(),
                    "alert_radius_km": self.alert_radius_km
                }
            else:
                error_message = response_data.get('message', f"HTTP status {response.status_code}")
                return {"error": error_message, "success": False, "status_code": response.status_code}

        except httpx.RequestError as e: # Catch httpx specific exceptions
            print(f"Error creating Whistle alert: {e}")
            return {"error": str(e), "success": False}
        except Exception as e:
            print(f"An unexpected error occurred during Whistle alert creation: {e}")
            return {"error": str(e), "success": False}


# ---
## Simple Sequential Workflow (Agents - Modified for Async)
# ---

class DataCollectorAgent:
    """Agent 1: Collects happy hour offers near multiple locations"""
    
    def __init__(self):
        self.location_service = LocationService()
    
    async def process(self, state: HappyHourState) -> HappyHourState: # Made async
        print("🔍 DATA COLLECTOR AGENT: Starting venue search for multiple locations...")
        
        all_venues = []
        location_tasks = []

        # Create tasks for geocoding all locations concurrently
        for location_name in state.user_locations:
            location_tasks.append(self.location_service.geocode_location(location_name))
        
        # Run geocoding tasks concurrently
        geocoded_results = await asyncio.gather(*location_tasks, return_exceptions=True)

        venue_search_tasks = []
        processed_locations_details = []

        for i, location_details_or_error in enumerate(geocoded_results):
            location_name = state.user_locations[i]
            if isinstance(location_details_or_error, Exception) or location_details_or_error is None:
                error_msg = f"Could not geocode location: {location_name} (Error: {location_details_or_error})"
                state.error_messages.append(error_msg)
                print(f"❌ {error_msg}")
                continue
            
            location_details = location_details_or_error
            state.coordinates_list.append({"latitude": location_details["lat"], "longitude": location_details["lng"]})
            state.location_details_list.append(location_details)
            processed_locations_details.append(location_details) # Store for venue search
            print(f"✅ OSM Geocoding success for '{location_name}': {location_details['full_address']}")

            # Create tasks for finding venues for each successfully geocoded location
            venue_search_tasks.append(
                self.location_service.find_venues_osm(
                    location_details["lat"], 
                    location_details["lng"], 
                    state.search_radius_km,
                    state.max_venues_per_location
                )
            )

        if not venue_search_tasks:
            state.processing_step = "data_collection_complete"
            print("\n📊 DATA COLLECTOR: No valid locations to search for venues.")
            return state

        # Run venue search tasks concurrently
        venues_per_location_results = await asyncio.gather(*venue_search_tasks, return_exceptions=True)

        for i, raw_venues_or_error in enumerate(venues_per_location_results):
            original_location_name = processed_locations_details[i]['city'] # Use city from geocoded details
            
            if isinstance(raw_venues_or_error, Exception) or not raw_venues_or_error:
                error_msg = f"No venues found near {original_location_name} within {state.search_radius_km}km (Error: {raw_venues_or_error})"
                state.error_messages.append(error_msg)
                print(f"⚠️ {error_msg}")
                continue

            raw_venues = raw_venues_or_error
            print(f"✅ OSM Venue Search success for {original_location_name}: Found {len(raw_venues)} potential venues.")
            
            for venue in raw_venues:
                if self._has_happy_hour_potential(venue):
                    enhanced_venue = self._enhance_venue_with_happy_hour_data(venue)
                    enhanced_venue['location_source'] = original_location_name # Attach the source
                    all_venues.append(enhanced_venue)
        
        state.venues_found = all_venues
        state.processing_step = "data_collection_complete"
        print(f"\n📊 DATA COLLECTOR: Total venues found across all locations: {len(all_venues)}")
        return state
    
    def _enhance_venue_with_happy_hour_data(self, venue: Dict) -> Dict:
        venue_type = venue["venue_type"]
        happy_hour_map = {
            "bar": {"offer": "Happy Hour: 25% off cocktails, wine & beer", "typical_hours": "4:00 PM - 7:00 PM", "tags": ["cocktails", "beer", "wine", "happy_hour", "bar"]},
            "pub": {"offer": "$2 off all drinks + discounted appetizers", "typical_hours": "3:00 PM - 6:00 PM", "tags": ["beer", "pub_food", "happy_hour", "appetizers"]},
            "restaurant": {"offer": "Discounted drinks with meal orders", "typical_hours": "4:30 PM - 6:30 PM", "tags": ["dining", "drinks", "happy_hour", "restaurant"]},
            "nightclub": {"offer": "Half-price drinks before 8 PM", "typical_hours": "5:00 PM - 8:00 PM", "tags": ["nightlife", "cocktails", "happy_hour", "club"]},
            "biergarten": {"offer": "$1 off all beers + pretzel specials", "typical_hours": "4:00 PM - 7:00 PM", "tags": ["beer", "outdoor", "german", "happy_hour"]},
            "cafe": {"offer": "20% off coffee & pastries", "typical_hours": "2:00 PM - 5:00 PM", "tags": ["coffee", "pastries", "afternoon", "cafe"]}
        }
        
        offer_data = happy_hour_map.get(venue_type, {
            "offer": "Happy Hour specials available", 
            "typical_hours": "varies", 
            "tags": ["happy_hour", venue_type]
        })
        
        today_end = datetime.now() + timedelta(hours=24)
        venue.update({
            "happy_hour_details": f"{offer_data['offer']} ({offer_data['typical_hours']})",
            "offer_valid_until": today_end.isoformat(),
            "tags": offer_data["tags"],
            "has_happy_hour": True
        })
        return venue
    
    def _has_happy_hour_potential(self, venue: Dict) -> bool:
        return venue["venue_type"] in ["bar", "pub", "restaurant", "nightclub", "biergarten", "cafe"]


class TavilySearchAgent:
    """Agent 2: Uses Tavily to find more details"""

    def __init__(self):
        self.tavily_service = TavilyAPIService(os.getenv("TAVILY_API_KEY"))

    async def process(self, state: HappyHourState) -> HappyHourState: # Made async
        print("🌐 TAVILY SEARCH AGENT: Searching for additional happy hour details...")
        
        if not state.venues_found:
            print("Tavily Search: No venues found to search for.")
            state.processing_step = "tavily_search_skipped_no_venues"
            return state

        tavily_results_summary = []
        updated_venues = []
        search_tasks = []

        for venue in state.venues_found:
            city = venue.get('location_source', '')
            query = f"{venue['name']} happy hour {city}"
            search_tasks.append(self.tavily_service.search(query, max_results=1))
        
        # Run all Tavily searches concurrently
        tavily_search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        for i, search_result_or_error in enumerate(tavily_search_results):
            venue = state.venues_found[i]
            
            if isinstance(search_result_or_error, Exception) or not search_result_or_error:
                tavily_results_summary.append({
                    "venue_name": venue['name'], 
                    "status": "not_found", 
                    "details_found": False, 
                    "source": "tavily",
                    "error": str(search_result_or_error) if isinstance(search_result_or_error, Exception) else None
                })
                updated_venues.append(venue)
                print(f"ℹ Tavily for '{venue['name']}': No additional details found.")
            else:
                tavily_results_summary.append({
                    "venue_name": venue['name'], 
                    "status": "enriched", 
                    "details_found": True, 
                    "source": "tavily",
                    "top_result_title": search_result_or_error[0].get('title'), 
                    "top_result_url": search_result_or_error[0].get('url')
                })
                updated_venues.append({**venue, "tavily_enriched": True})
                print(f"✅ Tavily success for '{venue['name']}': Found external details.")
        
        state.tavily_search_results = tavily_results_summary
        state.venues_found = updated_venues
        state.processing_step = "tavily_search_complete"
        print("🌐 TAVILY SEARCH AGENT: Completed search for venues.")
        return state


class WhistleAPICreatorAgent:
    """Agent 3: Creates Whistle API alerts"""
    
    def __init__(self, alert_radius_km: float = 1.0):
        self.whistle_service = WhistleAPIService(api_key=None, alert_radius_km=alert_radius_km)
    
    async def process(self, state: HappyHourState) -> HappyHourState: # Made async
        print("📡 WHISTLE API CREATOR AGENT: Creating alerts...")
        
        if not state.create_whistle_alerts:
            print("⏩ Whistle alerts creation skipped (user opted out)")
            state.processing_step = "whistle_alerts_skipped_user_choice"
            return state
        
        if not state.venues_found:
            state.error_messages.append("No venues to create alerts for.")
            state.processing_step = "whistle_alerts_skipped_no_venues"
            return state
        
        alert_creation_tasks = []
        eligible_venues = [venue for venue in state.venues_found if venue.get("has_happy_hour", False)]

        for venue in eligible_venues:
            alert_creation_tasks.append(self.whistle_service.create_alert(venue))
        
        # Run all alert creations concurrently
        alert_results = await asyncio.gather(*alert_creation_tasks, return_exceptions=True)

        alerts_created = []
        successful_alerts_count = 0

        for i, alert_result_or_error in enumerate(alert_results):
            venue = eligible_venues[i] # Get the corresponding venue
            
            if isinstance(alert_result_or_error, Exception) or not alert_result_or_error.get("success", False):
                error_msg = str(alert_result_or_error) if isinstance(alert_result_or_error, Exception) else alert_result_or_error.get("error", "Unknown error during alert creation")
                alerts_created.append({
                    "venue_name": venue["name"], 
                    "location_source": venue.get("location_source", ""),
                    "status": "failed", 
                    "error": error_msg
                })
                print(f"❌ Whistle Alert failed for '{venue['name']}': {error_msg}")
            else:
                successful_alerts_count += 1
                alerts_created.append({
                    "venue_name": venue["name"], 
                    "location_source": venue.get("location_source", ""),
                    "alert_id": alert_result_or_error["alert_id"], 
                    "status": "created", 
                    "expiry": alert_result_or_error["expiry"],
                    "alert_radius_km": alert_result_or_error.get("alert_radius_km", state.alert_radius_km)
                })
                print(f"✅ Whistle Alert success for '{venue['name']}': ID {alert_result_or_error['alert_id']}")
        
        state.whistle_alerts_created = alerts_created
        state.processing_step = "whistle_alerts_complete"
        print(f"📡 WHISTLE CREATOR: Successfully created {successful_alerts_count} alerts.")
        return state

# ---
## Main Application - Simple Sequential Workflow (Modified for Async and Pagination)
# ---

class HappyHourFinder:
    """Main application class - Simple sequential workflow"""
    
    def __init__(self):
        self.data_collector = DataCollectorAgent()
        self.tavily_searcher = TavilySearchAgent()
        # whistle_creator is initialized in find_happy_hours because alert_radius_km is dynamic

    async def find_happy_hours(self, locations: List[str], radius_km: float = 5.0, 
                                max_venues: int = 15, create_alerts: bool = False, 
                                alert_radius_km: float = 1.0, 
                                page: int = 1, page_size: int = 10) -> Dict: # Added pagination parameters
        """Find happy hour venues and optionally create Whistle alerts with pagination"""
        
        self.whistle_creator = WhistleAPICreatorAgent(alert_radius_km=alert_radius_km)
        
        state = HappyHourState(
            user_locations=locations,
            search_radius_km=radius_km,
            max_venues_per_location=max_venues,
            create_whistle_alerts=create_alerts,
            alert_radius_km=alert_radius_km
        )
        
        print("🍺 HAPPY HOUR FINDER - Enhanced Sequential Asynchronous Implementation")
        print("=" * 70)
        print(f"📍 Searching locations: {', '.join(locations)}")
        print(f"🎯 Search radius: {radius_km}km")
        print(f"📊 Max venues per location: {max_venues}")
        print(f"📡 Create Whistle alerts: {'Yes' if create_alerts else 'No'}")
        if create_alerts:
            print(f"🔊 Alert radius: {alert_radius_km}km")
        print(f"📄 Pagination: Page {page}, Size {page_size}") # Print pagination info
        print()
        
        try:
            # Step 1: Collect data (now async)
            state = await self.data_collector.process(state)
            if not state.venues_found and not state.error_messages:
                return self._format_results(state, page, page_size) # Pass pagination info
                
            # Step 2: Search with Tavily (now async)
            state = await self.tavily_searcher.process(state)
            
            # Step 3: Create Whistle alerts (if requested, now async)
            if create_alerts:
                state = await self.whistle_creator.process(state)
                
            return self._format_results(state, page, page_size) # Pass pagination info
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            state.error_messages.append(f"Unexpected error: {str(e)}")
            return self._format_results(state, page, page_size) # Pass pagination info
    
    def _format_results(self, state: HappyHourState, page: int, page_size: int) -> Dict:
        """Format final results for API response with pagination"""
        venues_for_display = [
            v for v in state.venues_found 
            if v.get("has_happy_hour", False)
        ]
        
        # Apply pagination
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_venues = venues_for_display[start_index:end_index]

        overall_success = (
            len(state.error_messages) == 0 and 
            len(venues_for_display) > 0
        )

        total_venues = len(venues_for_display)
        total_pages = (total_venues + page_size - 1) // page_size # Calculate total pages
        
        return {
            "success": overall_success,
            "search_metadata": {
                "locations": state.user_locations, 
                "coordinates_list": state.coordinates_list,
                "search_radius_km": state.search_radius_km,
                "max_venues_per_location": state.max_venues_per_location,
                "create_whistle_alerts": state.create_whistle_alerts,
                "alert_radius_km": state.alert_radius_km,
                "timestamp": datetime.now().isoformat()
            },
            "processing_summary": {
                "locations_processed": len(state.location_details_list),
                "osm_venue_search": f"found {len(state.venues_found)} venues total" if state.venues_found else "no venues found",
                "tavily_enrichment": f"processed {len(state.tavily_search_results)} venues ({sum(1 for r in state.tavily_search_results if r.get('details_found'))} enriched)" if state.tavily_search_results else "skipped",
                "whistle_alert_creation": f"created {len([a for a in state.whistle_alerts_created if a.get('status') == 'created'])} alerts" if state.whistle_alerts_created else "no alerts created"
            },
            "results_summary": {
                "total_venues_found": total_venues, # Total venues before pagination
                "venues_eligible_for_happy_hour": len(venues_for_display),
                "whistle_alerts_created_count": len([a for a in state.whistle_alerts_created if a.get("status") == "created"]) if state.whistle_alerts_created else 0
            },
            "pagination": { # Added pagination details
                "current_page": page,
                "page_size": page_size,
                "total_items": total_venues,
                "total_pages": total_pages,
                "items_on_page": len(paginated_venues)
            },
            "happy_hour_venues": paginated_venues, # Return paginated venues
            "whistle_alerts": state.whistle_alerts_created,
            "errors": state.error_messages
        }

# ---
## FastAPI Application (Modified for Async and Pagination)
# ---

app = FastAPI(
    title="Happy Hour Finder API",
    description="An API to find happy hour venues and optionally create Whistle alerts.",
    version="1.0.0",
)

# Define API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Function to validate the API key
def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == os.getenv("FASTAPI_API_KEY"):
        return api_key
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

class HappyHourRequest(BaseModel):
    locations: List[str] = Field(..., description="List of locations (cities/addresses) to search for happy hours.")
    search_radius_km: float = Field(5.0, ge=0.1, description="Search radius in kilometers around each location.")
    max_venues_per_location: int = Field(15, ge=1, description="Maximum number of venues to return per location.")
    create_whistle_alerts: bool = Field(False, description="Whether to create Whistle alerts for found happy hour venues.")
    alert_radius_km: float = Field(1.0, ge=0.1, description="Radius in kilometers for Whistle alerts.")

@app.get("/")
async def read_root():
    """
    Root endpoint for the Happy Hour Finder API.
    Provides a welcome message and directs to the main API endpoint.
    """
    return {
        "message": "Welcome to the Happy Hour Finder API!",
        "instructions": "Use the /find_happy_hours endpoint with a POST request to search for happy hours."
    }

@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    """
    Returns a 204 No Content response for favicon.ico requests.
    This prevents unnecessary 404 errors in server logs from browsers.
    """
    return Response(status_code=204)

@app.post("/find_happy_hours", response_model=Dict)
async def find_happy_hours_endpoint( # Made async
    request: HappyHourRequest = Body(...),
    api_key: str = Depends(get_api_key),
    page: int = Query(1, ge=1, description="Page number for results pagination."), # Added pagination query parameter
    page_size: int = Query(10, ge=1, le=100, description="Number of results per page.") # Added pagination query parameter
):
    """
    Finds happy hour venues based on the provided locations and parameters.
    Optionally creates Whistle alerts for the found venues.
    Results are paginated.
    This endpoint requires a valid API Key in the 'X-API-Key' header.
    """
    try:
        finder = HappyHourFinder()
        # Await the asynchronous find_happy_hours method, passing pagination parameters
        results = await finder.find_happy_hours(
            locations=request.locations,
            radius_km=request.search_radius_km,
            max_venues=request.max_venues_per_location,
            create_alerts=request.create_whistle_alerts,
            alert_radius_km=request.alert_radius_km,
            page=page, # Pass page to finder
            page_size=page_size # Pass page_size to finder
        )
        return {
            "status": "success" if results["success"] else "error",
            "message": "Happy Hour search complete." if results["success"] else "Happy Hour search failed.",
            "data": {
                "locations": results["search_metadata"]["locations"], 
                "coordinates_list": results["search_metadata"]["coordinates_list"],
                "venues": [
                    {
                        "name": v["name"], 
                        "location": {"lat": v["latitude"], "lng": v["longitude"]}, 
                        "distance_km": v["distance_km"],
                        "offer": v["happy_hour_details"], 
                        "venue_type": v["venue_type"], 
                        "tags": v["tags"], 
                        "valid_until": v["offer_valid_until"],
                        "location_source": v.get("location_source", "")
                    }
                    for v in results["happy_hour_venues"] # This now contains paginated results
                ]
            },
            "alerts_created": results["whistle_alerts"], 
            "processing_summary": results["processing_summary"],
            "pagination": results["pagination"], # Include pagination details in the response
            "timestamp": results["search_metadata"]["timestamp"], 
            "errors": results["errors"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# ---
## Enhanced CLI Interface (Modified for Async and Pagination)
# ---

def get_user_inputs(): # No change here, this is synchronous input
    """Get all user inputs for the enhanced happy hour finder"""
    print("🍺 HAPPY HOUR FINDER - Enhanced Configuration")
    print("=" * 50)
    
    print("\n📍 LOCATIONS:")
    cities_input = input("Enter cities/locations (comma-separated): ").strip()
    if not cities_input:
        print("❌ Please enter at least one location")
        return None
    
    cities = [city.strip() for city in cities_input.split(',') if city.strip()]
    if not cities:
        print("❌ Please enter valid locations")
        return None
    
    print("\n📊 VENUE LIMIT:")
    max_venues_str = input("How many venues per location? (default: 15): ").strip()
    max_venues = int(max_venues_str) if max_venues_str else 15
    if max_venues <= 0:
        print("❌ Max venues must be a positive number.")
        return None
    
    print("\n🎯 SEARCH RADIUS:")
    search_radius_str = input("Search radius in km? (default: 5.0): ").strip()
    search_radius = float(search_radius_str) if search_radius_str else 5.0
    if search_radius <= 0:
        print("❌ Search radius must be a positive number.")
        return None
    
    print("\n🔔 WHISTLE ALERTS:")
    create_alerts_input = input("Create Whistle alerts for happy hour venues? (yes/no, default: no): ").strip().lower()
    create_alerts = create_alerts_input == "yes"

    alert_radius = 1.0
    if create_alerts:
        alert_radius_str = input("Alert radius in km? (default: 1.0): ").strip()
        alert_radius = float(alert_radius_str) if alert_radius_str else 1.0
        if alert_radius <= 0:
            print("❌ Alert radius must be a positive number if creating alerts.")
            return None

    print("\n📄 PAGINATION:")
    page_str = input("Enter page number (default: 1): ").strip()
    page = int(page_str) if page_str else 1
    if page <= 0:
        print("❌ Page number must be a positive integer.")
        return None

    page_size_str = input("Enter page size (number of venues per page, default: 10): ").strip()
    page_size = int(page_size_str) if page_size_str else 10
    if page_size <= 0:
        print("❌ Page size must be a positive integer.")
        return None

    return {
        "locations": cities,
        "search_radius_km": search_radius,
        "max_venues_per_location": max_venues,
        "create_whistle_alerts": create_alerts,
        "alert_radius_km": alert_radius,
        "page": page,
        "page_size": page_size
    }

async def main(): # Changed to async main function
    """Main function to run the Happy Hour Finder CLI."""
    inputs = get_user_inputs()
    if not inputs:
        return

    finder = HappyHourFinder()
    results = await finder.find_happy_hours( # Await the async method
        locations=inputs["locations"],
        radius_km=inputs["search_radius_km"],
        max_venues=inputs["max_venues_per_location"],
        create_alerts=inputs["create_whistle_alerts"],
        alert_radius_km=inputs["alert_radius_km"],
        page=inputs["page"], # Pass page to finder
        page_size=inputs["page_size"] # Pass page_size to finder
    )

    print("\n" + "=" * 70)
    print("✨ HAPPY HOUR SEARCH RESULTS ✨")
    print("=" * 70)

    if results["success"]:
        print("\n✅ Search completed successfully!")
        print(f"Total venues found: {results['results_summary']['total_venues_found']}")
        print(f"Venues with potential happy hours: {results['results_summary']['venues_eligible_for_happy_hour']}")
        if results["search_metadata"]["create_whistle_alerts"]:
            print(f"Whistle alerts created: {results['results_summary']['whistle_alerts_created_count']}")

        print("\n--- Happy Hour Venues (Paginated) ---")
        pagination_info = results["pagination"]
        print(f"Page {pagination_info['current_page']} of {pagination_info['total_pages']} (Items on this page: {pagination_info['items_on_page']})")
        
        if results["happy_hour_venues"]:
            for i, venue in enumerate(results["happy_hour_venues"]):
                print(f"\n{i+1}. {venue['name']}")
                print(f"   Type: {venue['venue_type'].replace('_', ' ').title()}")
                print(f"   Location Source: {venue['location_source']}")
                print(f"   Distance: {venue['distance_km']:.2f} km")
                print(f"   Happy Hour: {venue['offer']}")
                print(f"   Valid Until: {venue['valid_until']}")
                print(f"   Tags: {', '.join(venue['tags'])}")
        else:
            print(f"No happy hour venues found on page {pagination_info['current_page']}.")

        if results["whistle_alerts"]:
            print("\n--- Whistle Alert Details ---")
            for i, alert in enumerate(results["whistle_alerts"]):
                status_icon = "✅" if alert.get("status") == "created" else "❌"
                print(f"\n{status_icon} Alert for {alert['venue_name']} ({alert['location_source']}):")
                print(f"   Status: {alert['status']}")
                if alert.get("alert_id"):
                    print(f"   Alert ID: {alert['alert_id']}")
                if alert.get("expiry"):
                    print(f"   Expiry: {alert['expiry']}")
                if alert.get("alert_radius_km"):
                    print(f"   Alert Radius: {alert['alert_radius_km']} km")
                if alert.get("error"):
                    print(f"   Error: {alert['error']}")

    else:
        print("\n❌ Search failed or completed with errors.")
        if results["errors"]:
            print("\n--- Errors ---")
            for error in results["errors"]:
                print(f"- {error}")

    print("\n--- Processing Summary ---")
    for step, summary in results["processing_summary"].items():
        print(f"- {step.replace('_', ' ').title()}: {summary}")


# This block ensures that `main()` is called when the script is executed directly.
# `if __name__ == "__main__":` is standard Python practice.
# `asyncio.run(main())` is the correct way to run an async function from a synchronous context.
if __name__ == "__main__":
    asyncio.run(main())
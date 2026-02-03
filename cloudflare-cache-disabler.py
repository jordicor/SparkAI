import os
import requests
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Get data from environment variables
api_key = os.getenv("CLOUDFLARE_API_KEY")
email = os.getenv("CLOUDFLARE_EMAIL")
zone_ids = [os.getenv("CLOUDFLARE_ZONE_ID"), os.getenv("CLOUDFLARE_ZONE_ID_2")]

# Headers for authentication
headers = {
    "X-Auth-Email": email,
    "X-Auth-Key": api_key,
    "Content-Type": "application/json"
}

# Payload to activate development mode
data = {
    "value": "on"
}

def activate_development_mode(zone_id):
    # Endpoint to activate development mode
    development_mode_url = f"https://api.cloudflare.com/client/v4/zones/{zone_id}/settings/development_mode"
    
    # Make PATCH request to activate development mode
    response = requests.patch(development_mode_url, headers=headers, json=data, timeout=(5, 15))
    
    # Verify if the response is successful
    if response.status_code == 200:
        print(f"Development mode successfully activated for zone {zone_id}.")
    else:
        print(f"Error activating development mode for zone {zone_id}: {response.status_code}")
        print(response.json())

# Activate development mode for both domains
for zone_id in zone_ids:
    activate_development_mode(zone_id)
import os
import base64
import hmac
import hashlib
import logging
import time
from typing import Tuple
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import jwt  # PyJWT
from jwt import PyJWTError

import uvicorn
from pathlib import Path

# Load environment variables
load_dotenv()

app = FastAPI()

# Debug configuration from environment
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Configure logger
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

# Secret key from environment
SECRET_KEY = os.getenv("APP_SECRET_KEY")
ALGORITHM = "HS256"

# Base directory for users
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/users')

# PEPPER for user hash generation from environment
PEPPER = os.getenv("PEPPER")

def debug(message: str):
    if DEBUG:
        logger.debug(message)

def generate_user_hash(username: str) -> Tuple[str, str, str]:
    data_to_hash = username + PEPPER
    hash_digest = hashlib.sha1(data_to_hash.encode()).hexdigest()
    return hash_digest[:3], hash_digest[3:7], hash_digest

@app.get("/image")
async def get_image(image_path: str, token: str, request: Request):
    logger.debug("enters get_image")
    try:
        # Decode the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Verify the token expiration
        exp = payload.get('exp', 0)
        username = payload.get('username')
        
        logger.debug(f"username: {username}")
        
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token: Username missing")
        
        current_time = int(time.time())
        if current_time > exp:
            raise HTTPException(status_code=401, detail="Token expired")
        
        # Generate the user hash
        hash_prefix1, hash_prefix2, user_hash = generate_user_hash(username)
        token_user_dir = os.path.join(hash_prefix1, hash_prefix2, user_hash)
        
        # Build the full image path
        full_image_path = os.path.join(BASE_DIR, token_user_dir, image_path.lstrip('/'))
        debug(f"Full image path: {full_image_path}")
        
        # Check if the file exists
        if not os.path.isfile(full_image_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get the MIME type of the file
        mime_type = "application/octet-stream"  # Default value
        import mimetypes
        mime_type_guess, _ = mimetypes.guess_type(full_image_path)
        if mime_type_guess:
            mime_type = mime_type_guess
        
        # Calculate the time until expiration
        time_until_expiration = exp - current_time
        if time_until_expiration < 0:
            time_until_expiration = 0
        
        # Configure the headers
        headers = {
            "Content-Type": mime_type,
            "Cache-Control": f"public, max-age={time_until_expiration}",
            "Expires": time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(exp))
        }
        
        # Send the file
        return FileResponse(full_image_path, headers=headers)
        
    except jwt.ExpiredSignatureError:
        debug("Token expired.")
        # Build the alternative path
        scheme = request.url.scheme
        host = request.headers.get('host')
        alternative_url = f"{scheme}://{host}/static/{image_path.lstrip('/')}"
        debug(f"Alternative path: {alternative_url}")
        return JSONResponse(status_code=401, content={"detail": "Token expired"})
    except jwt.PyJWTError:
        debug("Invalid token.")
        return JSONResponse(status_code=401, content={"detail": "Invalid token"})
    except HTTPException as http_exc:
        if http_exc.status_code == 404:
            debug("File not found.")
            return JSONResponse(status_code=404, content={"detail": "File not found"})
        else:
            return JSONResponse(status_code=http_exc.status_code, content={"detail": http_exc.detail})
    except Exception as e:
        debug(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Run the application using Uvicorn
# To run the server, use the following command in the terminal:
# uvicorn filename:app --host 0.0.0.0 --port 8000

if __name__ == '__main__':

    static_directory = Path("data/static")

    # Build the full paths for SSL files
    ssl_keyfile = os.path.join(static_directory, 'sec', 'privkey.pem')
    ssl_certfile = os.path.join(static_directory, 'sec', 'cert.pem')        
    
    uvicorn.run(
        "app:app",
        host='0.0.0.0',
        port=7789,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        log_level="debug",
        workers=1
    )

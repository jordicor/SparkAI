import os
import time
import sys
from datetime import timedelta

# Fixed cache path
CACHE_DIR = "data/cache"

def delete_old_files(age_limit):
    # Get the current time
    current_time = time.time()
    
    # Iterate through the folder and file structure
    for root, dirs, files in os.walk(CACHE_DIR):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            
            # Get the file age
            file_age = current_time - os.path.getmtime(file_path)
            
            # If the file is older than the limit, delete it
            if file_age > age_limit:
                os.remove(file_path)
                print(f"Deleted: {file_path}")

def parse_time_argument(arg):
    value = int(arg[:-1])
    unit = arg[-1]
    
    if unit == 'h':
        return timedelta(hours=value).total_seconds()
    elif unit == 'd':
        return timedelta(days=value).total_seconds()
    elif unit == 'w':
        return timedelta(weeks=value).total_seconds()
    elif unit == 'm':
        return timedelta(days=value*30).total_seconds()
    else:
        raise ValueError(f"Invalid time unit: {unit}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: clear-cache <time>")
        sys.exit(1)

    time_arg = sys.argv[1]
    
    try:
        # Measure the start time
        start_time = time.time()
        
        # Process deletion of old files
        age_limit = parse_time_argument(time_arg)
        delete_old_files(age_limit)
        
        # Measure end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nProcessed in {duration:.2f} seconds.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

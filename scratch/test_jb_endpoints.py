import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")
headers = {"Authorization": f"Api-Key {API_KEY}", "Content-Type": "application/json"}

endpoints = [
    "https://www.jblanked.com/news/api/mql5/calendar/today/",
    "https://www.jblanked.com/news/api/forex-factory/calendar/today/",
    "https://www.jblanked.com/news/api/fxstreet/calendar/today/",
    "https://www.jblanked.com/news/api/mql5/calendar/week/",
]

for url in endpoints:
    print(f"Testing {url}...")
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"Success! Found {len(data)} events.")
            break
        else:
            print(f"Msg: {r.text}")
    except Exception as e:
        print(f"Err: {e}")

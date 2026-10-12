import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://www.jblanked.com/news/api/forex-factory/calendar/week/"
api_key = os.getenv("NEWS_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Api-Key {api_key}",
}

print(f"Testing with Key: {api_key[:5]}...")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(f"Success! Found {len(data)} events.")
    if data:
        print(f"Sample: {data[0]}")
else:
    print(f"Failed: {response.status_code}")
    print(response.text)

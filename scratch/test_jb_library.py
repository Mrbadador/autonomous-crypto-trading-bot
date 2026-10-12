import os
from jb_news.news import JBNews, NEWS_SOURCE_FOREX_FACTORY
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

jb = JBNews()
print(f"Testing jb-news library with key: {API_KEY[:5]}...")

# Try loading this week's calendar (based on docs, 'today=True' might be limited)
# The docs show jb.calendar(API_KEY, today=True, news_source=NEWS_SOURCE)
try:
    if jb.calendar(API_KEY, today=False, news_source=NEWS_SOURCE_FOREX_FACTORY):
        print(f"Success! Found {len(jb.calendar_info)} events.")
        for event in jb.calendar_info[:5]:
            print(f"Event: {event.name} | Impact: {event.impact} | Date: {event.date}")
    else:
        print("Failed to load calendar.")
except Exception as e:
    print(f"Error: {e}")

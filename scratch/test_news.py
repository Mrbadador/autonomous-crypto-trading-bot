import asyncio
import aiohttp
import socket
import xml.etree.ElementTree as ET
from datetime import datetime, UTC, timedelta
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("test_news")

async def fetch_high_impact_news():
    url = "https://nfs.forexfactory.com/ff_calendar_thisweek.xml"
    try:
        # DNS Fix: Force IPv4 and use explicit resolvers
        resolver = aiohttp.AsyncResolver(nameservers=["1.1.1.1", "8.8.8.8"])
        connector = aiohttp.TCPConnector(family=socket.AF_INET, resolver=resolver, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, timeout=15) as resp:
                print(f"Status: {resp.status}")
                if resp.status != 200:
                    return
                text = await resp.text()
                root = ET.fromstring(text)
                
                count = 0
                for event in root.findall('event'):
                    impact = event.find('impact').text
                    if impact == 'High':
                        title = event.find('title').text
                        date_str = event.find('date').text
                        time_str = event.find('time').text
                        
                        print(f"Event: {title} | Date: {date_str} | Time: {time_str}")
                        count += 1
                print(f"Total High Impact: {count}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(fetch_high_impact_news())

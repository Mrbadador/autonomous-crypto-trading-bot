import os
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load the new key from your .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

async def test_groq():
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "Reply with exactly these words: 'API and Network are fully operational!'"}]
    }
    
    import socket
    resolver = aiohttp.ThreadedResolver()
    connector = aiohttp.TCPConnector(resolver=resolver, family=socket.AF_INET)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"\n✅ SUCCESS! AI Response: {data['choices'][0]['message']['content']}")
                elif response.status == 401:
                    print("\n❌ ERROR 401: Unauthorized. Your API key is invalid or not saving properly in the .env file.")
                else:
                    print(f"\n⚠️ FAILED! Status: {response.status}")
                    print(await response.text())
        except Exception as e:
            print(f"\n🚨 NETWORK/DNS ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_groq())

import asyncio
from playwright.async_api import async_playwright

async def get_error():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True)
        page = await b.new_page()
        await page.goto("http://localhost:8501")
        await page.wait_for_timeout(5000)
        try:
            err = await page.locator('[data-testid="stException"]').inner_text()
            print("EXCEPTION:")
            print(err)
        except Exception as e:
            print("No exception element found:", e)
        await b.close()

asyncio.run(get_error())

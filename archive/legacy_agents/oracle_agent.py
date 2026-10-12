import requests
import time
import random
import logging
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, UTC
from db_manager import DatabaseManager

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.FileHandler("logs/oracle.log"), logging.StreamHandler()]
)
log = logging.getLogger("oracle")

db = DatabaseManager()

def fetch_fear_greed():
    """Fetches the Bitcoin Fear & Greed Index from Alternative.me"""
    try:
        response = requests.get("https://api.alternative.me/fng/")
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            value = int(data['data'][0]['value'])
            return value
    except Exception as e:
        log.error(f"Failed to fetch Fear & Greed index: {e}")
    return 50 # Neutral default

def fetch_news_sentiment():
    """Fetches news from RSS feeds and analyzes sentiment using VADER."""
    urls = [
        'https://cointelegraph.com/rss',
        'https://www.coindesk.com/arc/outboundfeeds/rss/'
    ]
    
    titles = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]: # Top 10 latest entries per feed
                titles.append(entry.title)
        except Exception as e:
            log.warning(f"Failed to parse RSS feed {url}: {e}")
            
    if not titles:
        return 0.0 # Neutral if no news found
        
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for title in titles:
            vs = analyzer.polarity_scores(title)
            scores.append(vs['compound'])
            
        avg_sentiment = sum(scores) / len(scores)
        return round(avg_sentiment, 3)
    except Exception as e:
        log.error(f"Sentiment analysis failed: {e}")
        return 0.0

def run_oracle():
    log.info("Oracle Agent V15.0 'Sentience' starting...")
    while True:
        try:
            fng = fetch_fear_greed()
            news = fetch_news_sentiment()
            timestamp = datetime.now(UTC).isoformat()
            
            try:
                with db._get_conn() as conn:
                    conn.execute(
                        "INSERT INTO market_context (timestamp, fear_index, news_score) VALUES (?, ?, ?)",
                        (timestamp, fng, news)
                    )
                    conn.commit()
                log.info(f"Market Context Updated: F&G={fng}, News={news}")
            except Exception as e:
                log.error(f"DB Insert Failed (market_context): {e}")
            
            # Update system state for dashboard visibility
            db.save_state("fear_index", str(fng))
            db.save_state("news_sentiment", str(news))
            
        except Exception as e:
            log.error(f"Oracle cycle failed: {e}")
            
        time.sleep(300) # 5 minutes

if __name__ == "__main__":
    run_oracle()

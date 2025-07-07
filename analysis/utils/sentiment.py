import requests       #simple Http, GET/POST CAll ke lie, yahoo finance se html pull krne ke lie
from bs4 import BeautifulSoup   # html/xml parser, raw html se headline nikalna
from textblob import TextBlob  #NLP toolkit, headlines ka sentiment

def fetch_news(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        news_tags = soup.find_all('h3', {'class': 'Mb(5px)'})
        headlines = [tag.text.strip() for tag in news_tags[:5]]  # Top 5 headlines
        return headlines
    except Exception as e:
        print("Error fetching news:", e)
        return []
    

def get_sentiment(text):
    """
    Returns polarity score between -1 to 1
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity  # >0 pos, <0 neg, 0 = neutral
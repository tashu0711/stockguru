# from django.shortcuts import render              #resnder function jo html template and data, combine krke bhejta hai
from django.http import HttpResponse

def analyze_stock(request):
    return HttpResponse("stockguru working!")
# Create your views here.

from django.shortcuts import render
from analysis.utils.data import get_stock_data, fetch_news, get_sentiment             #custome function import ho raha, jo stock data fetch kr rha hai. hai

def analyze_stock(request):                            #main view function, jb bhi koi '/' click krta to ye chlta hai
    if request.method == "POST":                        #form submit hone pr
        ticker = request.POST.get('ticker')             #form se stock symbol le raha hai
        df = get_stock_data(ticker)                     #yfinance se data fetch

        close_prices = df[['Date', 'Close']].to_dict(orient='records')  #relavent data

        news = fetch_news(ticker)
        sentiments = [{'text': text, 'score': get_sentiment(text)} for text in news]

        return render(request, "result.html", {             #html + data response brwoser ko.
            "ticker": ticker,
            "prices": close_prices,
            "sentiments": sentiments
        })

    return render(request, "analysis/home.html")              #else case

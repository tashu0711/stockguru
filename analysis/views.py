from django.shortcuts import render              #resnder function jo html template and data, combine krke bhejta hai
from django.http import HttpResponse

# def analyze_stock(request):
#     return HttpResponse("stockguru working!")
# # Create your views here.

# from django.shortcuts import render
from analysis.utils.data import get_stock_data
# from analysis.utils.sentiment import fetch_news, get_sentiment
#              #custome function import ho raha, jo stock data fetch kr rha hai. hai
from analysis.utils.ml_model import train_model
import plotly.graph_objs as go
import plotly.offline as opy



# def analyze_stock(request):                            #main view function, jb bhi koi '/' click krta to ye chlta hai
#     if request.method == "POST":                        #form submit hone pr
#         ticker = request.POST.get('ticker')             #form se stock symbol le raha hai
#         df = get_stock_data(ticker)                     #yfinance se data fetch
#                 # STEP 5: Create interactive price chart using Plotly
#         trace = go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price')
#         layout = go.Layout(title=f'{ticker} Closing Prices', xaxis_title='Date', yaxis_title='Price')
#         figure = go.Figure(data=[trace], layout=layout)
#         chart_div = opy.plot(figure, auto_open=False, output_type='div')

#         prediction = train_model(df, sentiments)
#         decision = "Buy" if prediction == 1 else ("Sell" if prediction == -1 else "Hold")

#         close_prices = df[['Date', 'Close']].to_dict(orient='records')  #relavent data

#         news = fetch_news(ticker)
#         sentiments = [{'text': text, 'score': get_sentiment(text)} for text in news]

#         return render(request, "result.html", {             #html + data response brwoser ko.
#             "ticker": ticker,
#             "prices": close_prices,
#             "sentiments": sentiments,
#             "decision": decision,
#             "chart_div": chart_div
#         })

#     return render(request, "analysis/home.html")              #else case



from analysis.utils.sentiment import fetch_news, get_sentiment
...
def analyze_stock(request):
    if request.method == "POST":
        ticker = request.POST.get("ticker")

        # 1️⃣  STOCK DATA
        df = get_stock_data(ticker)

        # 2️⃣  NEWS + SENTIMENT  ---->  always set a default value first
        sentiments = []                            #  ←  default empty list
        headlines  = fetch_news(ticker)            #  may return []
        if headlines:                              #  only if non‑empty
            sentiments = [
                {"text": h, "score": get_sentiment(h)}
                for h in headlines
            ]

        # 3️⃣  ML PREDICTION  (pass safe value)
        # prediction = train_model(df, sentiments)
        # prediction = train_model(df, sentiments)

# ---------------- Fallback using MA-cross directly --------------
        # if df['MA10'].iloc[-1] > df['MA30'].iloc[-1]:
        #     decision = "Buy"
        # elif df['MA10'].iloc[-1] < df['MA30'].iloc[-1]:
        #     decision = "Sell"
        # else:
        #     decision = "Hold"


        prediction = train_model(df, sentiments)
        decision   = "Buy" if prediction == 1 else ("Sell" if prediction == -1 else "Hold")

# ----------------------------------------------------------------


        # decision   = ("Buy" if prediction == 1 else
        #               "Sell" if prediction == -1 else
        #               "Hold")

        # 4️⃣  PLOTLY CHART
        trace   = go.Scatter(x=df["Date"], y=df["Close"],
                             mode="lines", name="Close Price")
        layout  = go.Layout(title=f"{ticker} Closing Prices",
                            xaxis_title="Date", yaxis_title="Price")
        figure  = go.Figure([trace], layout)
        chart_div = opy.plot(figure, auto_open=False, output_type="div")

        # 5️⃣  RENDER
        return render(request, "analysis/result.html", {
            "ticker": ticker,
            "prices": df[["Date","Close"]].tail(10).values.tolist(),
            "sentiments": sentiments,      #  always defined
            "decision": decision,
            "chart_div": chart_div,
        })

    # GET request → show form
    return render(request, "analysis/home.html")

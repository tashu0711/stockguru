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


# ═══════════════════════════════════════════════════════════════════════
#  SMART SCREENER + PAPER TRADING VIEWS
# ═══════════════════════════════════════════════════════════════════════

from django.shortcuts import redirect
from django.contrib import messages
from django.db.models import Sum, F, Q
from decimal import Decimal
import yfinance as yf
from analysis.utils.screener import screen_top_volume_negative, run_backtest
from analysis.models import PaperTrader, Trade, Holding, SuggestionLog


def get_or_create_trader(request):
    """Get or create PaperTrader for this session."""
    if not request.session.session_key:
        request.session.create()
    session_key = request.session.session_key

    trader, created = PaperTrader.objects.get_or_create(session_key=session_key)
    return trader


# ── SMART SCREENER ──────────────────────────────────────────────────────

def smart_screener(request):
    """Main screener page - finds top volume negative stocks."""
    results = None
    error = None

    if request.method == "POST":
        try:
            result_data = screen_top_volume_negative(top_n=4)

            # Check if time error
            if result_data and "error" in result_data:
                error = result_data["error"]
                results = None
            else:
                results = result_data

                # Log suggestions to track performance later
                if results and results.get("signals"):
                    for stock in results["signals"]:
                        SuggestionLog.objects.create(
                            ticker=stock["ticker"],
                            buy_price=stock["current"],
                            target_price=stock["target_price"],
                            stop_loss=stock.get("stop_loss"),
                        )
        except Exception as e:
            error = f"Screening error: {str(e)}"

    return render(request, "analysis/screener.html", {
        "results": results,
        "error": error,
    })


# ── BACKTEST REPORT ─────────────────────────────────────────────────────

def backtest_report(request):
    """Show historical backtest results (12 months with monthly breakdown)."""
    report = None
    error = None

    if request.method == "POST":
        try:
            report = run_backtest(months=12, top_n=4, target_pct=2.5, max_hold_days=7)
        except Exception as e:
            error = f"Backtest error: {str(e)}"

    return render(request, "analysis/backtest.html", {
        "report": report,
        "error": error,
    })


# ── PORTFOLIO ──────────────────────────────────────────────────────────

def portfolio(request):
    """Show user's paper trading portfolio with live P&L."""
    trader = get_or_create_trader(request)
    holdings = trader.holdings.all()

    # Fetch current prices for all holdings
    holdings_data = []
    total_investment = Decimal(0)
    total_current_value = Decimal(0)

    for holding in holdings:
        try:
            ticker_yf = f"{holding.ticker}.NS"
            stock = yf.Ticker(ticker_yf)
            hist = stock.history(period="1d")

            if not hist.empty:
                current_price = Decimal(str(hist["Close"].iloc[-1]))
            else:
                current_price = holding.avg_price  # fallback

            investment = holding.avg_price * holding.quantity
            current_value = current_price * holding.quantity
            pnl = current_value - investment
            pnl_pct = (pnl / investment) * 100 if investment > 0 else 0

            total_investment += investment
            total_current_value += current_value

            holdings_data.append({
                "ticker": holding.ticker,
                "quantity": holding.quantity,
                "avg_price": holding.avg_price,
                "current_price": current_price,
                "investment": investment,
                "current_value": current_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })
        except Exception as e:
            print(f"Error fetching {holding.ticker}: {e}")
            continue

    # Calculate performance stats
    trades = trader.trades.all()
    total_trades = trades.count()

    # Win rate calculation (simplified: profit = win)
    wins = 0
    losses = 0
    for trade in trades:
        if trade.action == "SELL":
            # Find corresponding buy
            buy_trades = Trade.objects.filter(
                trader=trader,
                ticker=trade.ticker,
                action="BUY",
                timestamp__lt=trade.timestamp
            ).order_by("-timestamp")

            if buy_trades.exists():
                buy_price = buy_trades.first().price
                if trade.price > buy_price:
                    wins += 1
                else:
                    losses += 1

    win_rate = round((wins / (wins + losses)) * 100, 1) if (wins + losses) > 0 else 0
    total_pnl = total_current_value - total_investment

    return render(request, "analysis/portfolio.html", {
        "trader": trader,
        "holdings": holdings_data,
        "total_investment": total_investment,
        "total_current_value": total_current_value,
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
    })


# ── BUY STOCK ──────────────────────────────────────────────────────────

def buy_stock(request):
    """Buy a stock with virtual money."""
    if request.method == "POST":
        trader = get_or_create_trader(request)

        ticker = request.POST.get("ticker")
        quantity = int(request.POST.get("quantity", 1))
        price = Decimal(request.POST.get("price"))

        total_cost = price * quantity

        if trader.balance >= total_cost:
            # Deduct balance
            trader.balance -= total_cost
            trader.save()

            # Create trade record
            Trade.objects.create(
                trader=trader,
                ticker=ticker,
                action="BUY",
                quantity=quantity,
                price=price,
                total=total_cost,
            )

            # Update or create holding (averaging)
            holding, created = Holding.objects.get_or_create(
                trader=trader,
                ticker=ticker,
                defaults={"quantity": quantity, "avg_price": price}
            )

            if not created:
                # Average down/up
                total_qty = holding.quantity + quantity
                total_investment = (holding.avg_price * holding.quantity) + (price * quantity)
                holding.avg_price = total_investment / total_qty
                holding.quantity = total_qty
                holding.save()

            messages.success(request, f"✓ Bought {quantity} {ticker} @ ₹{price}")
        else:
            messages.error(request, "❌ Insufficient balance!")

    return redirect("portfolio")


# ── SELL STOCK ─────────────────────────────────────────────────────────

def sell_stock(request):
    """Sell a stock and credit balance."""
    if request.method == "POST":
        trader = get_or_create_trader(request)

        ticker = request.POST.get("ticker")
        quantity = int(request.POST.get("quantity"))
        price = Decimal(request.POST.get("price"))

        try:
            holding = Holding.objects.get(trader=trader, ticker=ticker)

            if holding.quantity >= quantity:
                total_value = price * quantity

                # Credit balance
                trader.balance += total_value
                trader.save()

                # Create trade record
                Trade.objects.create(
                    trader=trader,
                    ticker=ticker,
                    action="SELL",
                    quantity=quantity,
                    price=price,
                    total=total_value,
                )

                # Update holding
                holding.quantity -= quantity
                if holding.quantity == 0:
                    holding.delete()
                else:
                    holding.save()

                messages.success(request, f"✓ Sold {quantity} {ticker} @ ₹{price}")
            else:
                messages.error(request, "❌ Not enough quantity to sell!")
        except Holding.DoesNotExist:
            messages.error(request, "❌ You don't own this stock!")

    return redirect("portfolio")


# ── TRADE HISTORY ──────────────────────────────────────────────────────

def trade_history(request):
    """Show all past trades."""
    trader = get_or_create_trader(request)
    trades = trader.trades.all()

    return render(request, "analysis/history.html", {
        "trades": trades,
        "trader": trader,
    })

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from analysis.utils.sentiment import fetch_news, get_sentiment

IST = pytz.timezone("Asia/Kolkata")

# ── Stock Universe: NIFTY 500 + Top BSE stocks ─────────────────────────
def get_ticker_list():
    """
    Returns NIFTY 500 + BSE 100 stocks.
    Try to import from niftystocks, fallback to hardcoded list.
    """
    # Try niftystocks package
    try:
        from niftystocks import get_nifty500_with_ns
        nifty500 = get_nifty500_with_ns()
    except:
        # Fallback: Expanded list (NIFTY 200 most liquid + some BSE)
        nifty500 = FALLBACK_TICKERS

    # Add top BSE stocks (these have good volume)
    bse_top = [
        "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO",
        "SBIN.BO", "BHARTIARTL.BO", "ITC.BO", "LT.BO", "AXISBANK.BO",
        "KOTAKBANK.BO", "MARUTI.BO", "SUNPHARMA.BO", "TATAMOTORS.BO", "ONGC.BO",
        "JSWSTEEL.BO", "TATASTEEL.BO", "WIPRO.BO", "HCLTECH.BO", "NTPC.BO",
    ]

    # Combine NSE + BSE (avoid duplicates by checking symbol)
    all_tickers = list(set(nifty500 + bse_top))
    return all_tickers


# Fallback list (NIFTY 200 + liquid mid/small caps)
FALLBACK_TICKERS = [
    # NIFTY 50
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "DMART.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "WIPRO.NS", "HCLTECH.NS", "POWERGRID.NS", "NTPC.NS", "TATAMOTORS.NS",
    "ONGC.NS", "JSWSTEEL.NS", "M&M.NS", "TATASTEEL.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "TECHM.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "DIVISLAB.NS",
    "GRASIM.NS", "DRREDDY.NS", "CIPLA.NS", "SBILIFE.NS", "BRITANNIA.NS",
    "COALINDIA.NS", "BPCL.NS", "EICHERMOT.NS", "APOLLOHOSP.NS", "INDUSINDBK.NS",
    "TATACONSUM.NS", "HEROMOTOCO.NS", "UPL.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS",

    # NIFTY NEXT 50
    "ADANIGREEN.NS", "AMBUJACEM.NS", "BANKBARODA.NS", "BERGEPAINT.NS",
    "BIOCON.NS", "BOSCHLTD.NS", "CHOLAFIN.NS", "COLPAL.NS", "CONCOR.NS",
    "DLF.NS", "DABUR.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS",
    "ICICIGI.NS", "ICICIPRULI.NS", "INDUSTOWER.NS", "IOC.NS", "IRCTC.NS",
    "JINDALSTEL.NS", "LICI.NS", "LUPIN.NS", "MARICO.NS", "MCDOWELL-N.NS",
    "MOTHERSON.NS", "NAUKRI.NS", "PEL.NS", "PIDILITIND.NS", "PFC.NS",
    "PIIND.NS", "PNB.NS", "RECLTD.NS", "SBICARD.NS", "SIEMENS.NS",
    "SRF.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS", "VEDL.NS",
    "ZOMATO.NS", "ZYDUSLIFE.NS", "HAL.NS", "BEL.NS", "CANBK.NS",
    "UNIONBANK.NS", "INDIANB.NS", "IDEA.NS", "SAIL.NS", "NHPC.NS",

    # NIFTY MIDCAP 50 (selected liquid)
    "GODREJPROP.NS", "MUTHOOTFIN.NS", "ABB.NS", "TVSMOTOR.NS", "ESCORTS.NS",
    "LICHSGFIN.NS", "AUBANK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "AUROPHARMA.NS", "GLENMARK.NS", "TORNTPOWER.NS", "ADANIPOWER.NS", "CUMMINSIND.NS",
    "MOTHERSUMI.NS", "ASTRAL.NS", "PETRONET.NS", "PAGEIND.NS", "COFORGE.NS",
    "L&TFH.NS", "M&MFIN.NS", "SHREECEM.NS", "MPHASIS.NS", "PERSISTENT.NS",
    "INDIAMART.NS", "LTTS.NS", "OFSS.NS", "GUJGASLTD.NS", "BALKRISIND.NS",
    "ATUL.NS", "ALKYLAMINE.NS", "DEEPAKNTR.NS", "CLEAN.NS", "DIXON.NS",
    "RELAXO.NS", "CROMPTON.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "BATAINDIA.NS",

    # Additional liquid stocks
    "VEDL.NS", "RBLBANK.NS", "YESBANK.NS", "COALINDIA.NS", "GAIL.NS",
    "NMDC.NS", "SAIL.NS", "JSWENERGY.NS", "ADANIGREEN.NS", "TATAPOWER.NS",
    "TORNTPOWER.NS", "RECLTD.NS", "PFIZER.NS", "ALKEM.NS", "BIOCON.NS",
    "IPCALAB.NS", "AFFLE.NS", "ROUTE.NS", "HAPPSTMNDS.NS", "MINDTREE.NS",
]


# ───────────────────────────────────────────────────────────────────────
#  LIVE SCREENER  –  with 2-4 PM time restriction
# ───────────────────────────────────────────────────────────────────────

def screen_top_volume_negative(top_n=4):
    """
    Core screening logic:
    1. Check if current time is 2-4 PM IST
    2. Download today's intraday data for all tickers (FAST: 1d period)
    3. Sort by volume  -> Top N
    4. Filter: current price < open price (negative)
    5. Add risk level, sell targets, news + sentiment
    Returns dict with results
    """
    # TIME CHECK: Only allow between 2-4 PM IST
    now_ist = datetime.now(IST)
    current_hour = now_ist.hour

    # Allow 14:00 (2 PM) to 15:59 (before 4 PM)
    if current_hour < 14 or current_hour >= 16:
        return {
            "error": f"⏰ Screener only runs between 2-4 PM IST. Current time: {now_ist.strftime('%I:%M %p IST')}",
            "allowed_time": "2:00 PM - 4:00 PM IST",
            "current_time": now_ist.strftime("%I:%M %p IST"),
        }

    tickers = get_ticker_list()
    print(f"Screening {len(tickers)} stocks...")

    # SPEED OPTIMIZATION: Use period='1d' instead of '5d' (only today's data)
    try:
        raw = yf.download(
            tickers,
            period="1d",         # Only today (FAST)
            interval="5m",       # 5-min candles
            group_by="ticker",
            progress=False,
            threads=True,        # Parallel download
        )
    except Exception as e:
        print("Screener download error:", e)
        return {"error": f"Data fetch error: {str(e)}"}

    if raw.empty:
        return {"error": "No data available from yfinance"}

    results = []

    for ticker in tickers:
        try:
            # Handle single vs multiple ticker dataframes
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()

            if df.empty:
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna(subset=["Close", "Open", "Volume"])
            if df.empty or len(df) < 2:
                continue

            # Extract metrics
            open_price = float(df["Open"].iloc[0])
            current_price = float(df["Close"].iloc[-1])
            total_volume = int(df["Volume"].sum())
            change_pct = ((current_price - open_price) / open_price) * 100 if open_price > 0 else 0

            results.append({
                "ticker": ticker.replace(".NS", "").replace(".BO", ""),
                "ticker_yf": ticker,
                "exchange": "NSE" if ".NS" in ticker else "BSE",
                "open": round(open_price, 2),
                "current": round(current_price, 2),
                "change_pct": round(change_pct, 2),
                "volume": total_volume,
            })
        except Exception as e:
            # Skip stocks with errors
            continue

    if not results:
        return {"error": "No valid stock data found"}

    # Sort by volume descending → Top N most bought
    results.sort(key=lambda x: x["volume"], reverse=True)
    top_stocks = results[:top_n]

    # Filter: only negative stocks (current < open)
    negative_stocks = [s for s in top_stocks if s["change_pct"] < 0]

    # Enrich each negative stock with targets, risk, news
    for stock in negative_stocks:
        change = stock["change_pct"]

        # Risk meter (based on how negative)
        if -2 <= change < -0.3:
            stock["risk_level"] = "STRONG BUY"
            stock["risk_color"] = "success"
        elif -3 <= change < -2:
            stock["risk_level"] = "BUY"
            stock["risk_color"] = "info"
        elif change < -3:
            stock["risk_level"] = "RISKY"
            stock["risk_color"] = "danger"
        else:
            stock["risk_level"] = "WEAK"
            stock["risk_color"] = "secondary"

        # Sell targets (swing trade: 1 week hold)
        stock["target_price"] = round(stock["current"] * 1.025, 2)  # +2.5%
        stock["stop_loss"] = round(stock["current"] * 0.97, 2)       # -3%
        stock["hold_period"] = "1 Week"
        stock["exit_note"] = "Hold if still negative next day. Don't panic sell."

        # News + Sentiment (optional - can be slow)
        try:
            headlines = fetch_news(stock["ticker_yf"])
            stock["sentiments"] = []
            stock["avg_sentiment"] = 0
            if headlines:
                sents = [{"text": h, "score": round(get_sentiment(h), 3)} for h in headlines[:5]]
                stock["sentiments"] = sents
                stock["avg_sentiment"] = round(np.mean([s["score"] for s in sents]), 3)
        except:
            stock["sentiments"] = []
            stock["avg_sentiment"] = 0

    return {
        "top_volume": top_stocks,
        "signals": negative_stocks,
        "screened_at": now_ist.strftime("%d-%b-%Y %I:%M %p IST"),
        "total_scanned": len(tickers),
    }


# ───────────────────────────────────────────────────────────────────────
#  BACKTEST  –  Past 2 months historical simulation
# ───────────────────────────────────────────────────────────────────────

def run_backtest(months=2, top_n=4, target_pct=2.5, max_hold_days=7):
    """
    Backtest the strategy over past `months` months.

    For each trading day at ~2 PM:
      1. Find top N volume stocks
      2. Filter negative ones
      3. Check if they hit +target_pct% within max_hold_days

    Returns backtest results dict with monthly breakdown.
    """
    tickers = get_ticker_list()
    end_date = datetime.now(IST).date()
    start_date = end_date - timedelta(days=months * 30 + 15)  # extra buffer for data

    # Download daily data for all tickers
    try:
        raw = yf.download(
            tickers,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as e:
        print("Backtest download error:", e)
        return None

    if raw.empty:
        return None

    # Build per-ticker daily DataFrames
    ticker_data = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna(subset=["Close", "Open", "Volume"])
            if len(df) < 10:
                continue
            df = df.reset_index()
            ticker_data[ticker] = df
        except Exception:
            continue

    if not ticker_data:
        return None

    # Get common trading dates
    all_dates = set()
    for df in ticker_data.values():
        all_dates.update(df["Date"].dt.date.tolist())
    trading_days = sorted(all_dates)

    # Only use dates within our backtest window
    bt_start = end_date - timedelta(days=months * 30)
    trading_days = [d for d in trading_days if d >= bt_start]

    trades = []
    wins = 0
    losses = 0
    total_return_pct = 0

    for day in trading_days:
        # For each day, find top N volume stocks
        day_stocks = []
        for ticker, df in ticker_data.items():
            day_rows = df[df["Date"].dt.date == day]
            if day_rows.empty:
                continue
            row = day_rows.iloc[0]
            open_p = float(row["Open"])
            close_p = float(row["Close"])
            vol = int(row["Volume"])
            change = ((close_p - open_p) / open_p) * 100 if open_p > 0 else 0

            day_stocks.append({
                "ticker": ticker.replace(".NS", "").replace(".BO", ""),
                "ticker_yf": ticker,
                "open": open_p,
                "close": close_p,
                "volume": vol,
                "change_pct": change,
                "date": day,
            })

        if not day_stocks:
            continue

        # Top N by volume
        day_stocks.sort(key=lambda x: x["volume"], reverse=True)
        top = day_stocks[:top_n]

        # Filter negative
        negative = [s for s in top if s["change_pct"] < 0]

        for stock in negative:
            # Look ahead: did it hit target in next max_hold_days?
            ticker_df = ticker_data.get(stock["ticker_yf"])
            if ticker_df is None:
                continue

            buy_price = stock["close"]  # bought at close on signal day
            future = ticker_df[ticker_df["Date"].dt.date > day].head(max_hold_days)

            if future.empty:
                continue

            # Check if high in any future day hit target
            max_price = float(future["High"].max()) if "High" in future.columns else float(future["Close"].max())
            exit_price = float(future["Close"].iloc[-1])  # price at end of hold
            ret = ((max_price - buy_price) / buy_price) * 100
            actual_ret = ((exit_price - buy_price) / buy_price) * 100

            hit_target = ret >= target_pct
            if hit_target:
                wins += 1
                trade_ret = target_pct  # exited at target
            else:
                trade_ret = actual_ret  # held full period
                if trade_ret >= 0:
                    wins += 1
                else:
                    losses += 1

            total_return_pct += trade_ret

            trades.append({
                "date": day.strftime("%d-%b-%Y"),
                "ticker": stock["ticker"],
                "buy_price": round(buy_price, 2),
                "max_price": round(max_price, 2),
                "exit_price": round(exit_price, 2),
                "return_pct": round(trade_ret, 2),
                "hit_target": hit_target,
            })

    total_trades = wins + losses
    win_rate = round((wins / total_trades) * 100, 1) if total_trades > 0 else 0
    avg_return = round(total_return_pct / total_trades, 2) if total_trades > 0 else 0

    # Virtual P&L: starting with 1 Lakh
    virtual_capital = 100000
    cumulative = []
    running = virtual_capital
    for t in trades:
        profit = running * (t["return_pct"] / 100) * 0.25  # 25% capital per trade
        running += profit
        cumulative.append({
            "date": t["date"],
            "capital": round(running, 2),
        })

    # ── MONTHLY BREAKDOWN ──
    from collections import defaultdict
    monthly_stats = defaultdict(lambda: {
        "trades": [],
        "wins": 0,
        "losses": 0,
        "total_return": 0,
        "capital_start": 0,
        "capital_end": 0,
    })

    # Group trades by month
    for i, t in enumerate(trades):
        # Parse date (format: "dd-MMM-yyyy")
        try:
            trade_date = datetime.strptime(t["date"], "%d-%b-%Y")
            month_key = trade_date.strftime("%b %Y")  # e.g., "Jan 2025"

            monthly_stats[month_key]["trades"].append(t)
            monthly_stats[month_key]["total_return"] += t["return_pct"]

            if t["hit_target"] or t["return_pct"] >= 0:
                monthly_stats[month_key]["wins"] += 1
            else:
                monthly_stats[month_key]["losses"] += 1

            # Track capital evolution
            if i < len(cumulative):
                monthly_stats[month_key]["capital_end"] = cumulative[i]["capital"]
        except:
            continue

    # Calculate monthly metrics
    monthly_breakdown = []
    running_capital = virtual_capital

    for month in sorted(monthly_stats.keys(), key=lambda x: datetime.strptime(x, "%b %Y")):
        data = monthly_stats[month]
        month_trades = len(data["trades"])
        month_wins = data["wins"]
        month_losses = data["losses"]
        month_win_rate = round((month_wins / month_trades) * 100, 1) if month_trades > 0 else 0
        month_avg_return = round(data["total_return"] / month_trades, 2) if month_trades > 0 else 0

        # Calculate profit for this month
        month_profit = 0
        for trade in data["trades"]:
            month_profit += running_capital * (trade["return_pct"] / 100) * 0.25

        capital_before = running_capital
        running_capital += month_profit

        monthly_breakdown.append({
            "month": month,
            "trades": month_trades,
            "wins": month_wins,
            "losses": month_losses,
            "win_rate": month_win_rate,
            "avg_return": month_avg_return,
            "total_return": round(data["total_return"], 2),
            "capital_start": round(capital_before, 2),
            "capital_end": round(running_capital, 2),
            "profit": round(month_profit, 2),
        })

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "total_return_pct": round(total_return_pct, 2),
        "final_capital": round(running, 2) if cumulative else virtual_capital,
        "profit": round(running - virtual_capital, 2) if cumulative else 0,
        "trades": trades,
        "cumulative": cumulative,
        "monthly_breakdown": monthly_breakdown,
        "months": months,
        "target_pct": target_pct,
        "max_hold_days": max_hold_days,
    }

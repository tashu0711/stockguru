# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# import numpy as np

# def train_model(df, sentiments):
#     df = df.copy()

#     # Step 1: Calculate average sentiment
#     avg_sentiment = np.mean([s['score'] for s in sentiments]) if sentiments else 0

#     # Step 2: Feature engineering
#     df['Sentiment'] = avg_sentiment
#     df['Price_Change'] = df['Close'].pct_change()
#     df.dropna(inplace=True)

#     # Step 3: Create target labels (1 = Buy, 0 = Hold, -1 = Sell)
#     df['Signal'] = df['Price_Change'].apply(lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0))

#     # Step 4: Prepare features (X) and labels (y)
#     X = df[['Price_Change', 'Sentiment']]
#     y = df['Signal']

#     # Step 5: Scale data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Step 6: Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#     # Step 7: Train model
#     model = RandomForestClassifier()
#     model.fit(X_train, y_train)

#     # Step 8: Predict latest row (recent trend)
#     latest_input = scaler.transform([[df['Price_Change'].iloc[-1], avg_sentiment]])
#     prediction = model.predict(latest_input)[0]

#     return prediction

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from django.shortcuts import render

# --- stock data fetch ---
# df = get_stock_data(ticker, period='1y')

# ⬇️  अगर Yahoo ने data नहीं दिया, graceful fallback
def handle_empty_stock_data(df, ticker, request):
    if df.empty or 'Close' not in df.columns:
        return render(request, "analysis/result.html", {
            "ticker": ticker,
            "error": "⚠ Stock data unavailable (rate‑limit). Please try again in a few minutes."
        })


def _debug_info(df):
    print("─" * 40)
    print("Rows after dropna:", len(df))
    if not df.empty:
        print("Label counts:\n", df['Signal'].value_counts(dropna=False))
        print("Latest Price_Change:", df['Price_Change'].iloc[-1])
    print("─" * 40)



def train_model(df, sentiments):
    df = df.copy()

    # 1️⃣  Sentiment average (safe fallback)
    avg_sent = np.mean([s['score'] for s in sentiments]) if sentiments else 0
    df['Sentiment'] = avg_sent

    # 2️⃣  Price change
    df['Price_Change'] = df['Close'].pct_change()

    # 3️⃣  Drop missing values
    df.dropna(inplace=True)

        # ---- MOVING‑AVERAGE LABEL LOGIC ----
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA30'] = df['Close'].rolling(30).mean()
    df.dropna(inplace=True)

    df['Signal'] = np.where(df['MA10'] > df['MA30'], 1,
                            np.where(df['MA10'] < df['MA30'], -1, 0))

    _debug_info(df)          # ←  यही जगह debug‑print की

    # 🧠 DEBUGGING: Check data size
    print("Rows after dropna:", len(df))

    # 4️⃣  Adaptive threshold for Buy/Sell labels (e.g., > 0.3%)
    df['Signal'] = df['Price_Change'].apply(
        lambda x: 1 if x > 0.003 else (-1 if x < -0.003 else 0)
    )

    # 🔍 Optional: Print label distribution
    print("Label counts:", df['Signal'].value_counts())

    # 5️⃣  Edge case check
    if df.empty or len(df) < 5:
        return 0  # Not enough data, return HOLD

    if df.empty or 'Close' not in df.columns:
        print("⚠️  Empty DF inside train_model – returning Hold")
        return 0


    # 6️⃣  Prepare features & target
    X = df[['Price_Change', 'Sentiment']]
    y = df['Signal']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 7️⃣  Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 8️⃣  Predict latest decision
    latest_input = scaler.transform([[df['Price_Change'].iloc[-1], avg_sent]])
    prediction = model.predict(latest_input)[0]

    return prediction


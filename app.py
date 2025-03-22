from flask import Flask, render_template_string, request, jsonify
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import base64
import os

app = Flask(__name__)

# Load background image safely
background_base64 = ""
if os.path.exists("background.jpg"):
    with open("background.jpg", "rb") as image_file:
        background_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# Fetch real-time stock data
def get_real_time_stock_data(symbol):
    stock = yf.Ticker(f"{symbol}.NS")
    try:
        df = stock.history(period="1d", interval="1m")
        if df.empty or df.isna().all().all():
            return None
        latest = df.iloc[-1]
        return {"time": str(latest.name), "close": round(latest["Close"], 2), "volume": int(latest["Volume"])}
    except Exception:
        return None

# Fetch historical stock data
def get_historical_stock_data(symbol):
    stock = yf.Ticker(f"{symbol}.NS")
    try:
        df = stock.history(period="3mo")
        if df.empty or df.isna().all().all():
            return None
        return df
    except Exception:
        return None

# Prepare data for prediction
def prepare_data(df):
    if df is None or df.empty:
        return None, None, None
    df["Prev_Close"] = df["Close"].shift(1)
    df["Price_Change"] = df["Close"] - df["Prev_Close"]
    df["Target"] = (df["Price_Change"] > 0).astype(int)
    df = df.dropna()
    if df.empty:
        return None, None, None
    return df[["Prev_Close", "Volume"]], df["Target"], df

# Train model and predict
def train_and_predict(features, target, current_data):
    if features is None or target is None or current_data is None or features.empty or target.empty:
        return None
    if len(features) < 10:
        return None
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    current_features = scaler.transform([[current_data["close"], current_data["volume"]]])
    prediction = model.predict(current_features)[0]
    return "Up" if prediction == 1 else "Down"

# Web Interface
HTML_PAGE = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Finnova Stock Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            width: 100vw;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: url('data:image/jpeg;base64,{background_base64}') no-repeat center center fixed;
            background-size: cover;
            color: white;
        }}
        .container {{
            width: 90%;
            max-width: 800px;
            background: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        input, button {{
            padding: 10px;
            font-size: 16px;
            margin: 10px;
            border-radius: 5px;
            border: none;
        }}
        button {{
            background-color: #28a745;
            color: white;
            cursor: pointer;
        }}
        button:hover {{
            background-color: #218838;
        }}
        canvas {{
            width: 100%;
            max-height: 300px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>

    <div class="container">
        <h1>WELCOME TO FINNOVA PREDICTOR</h1>
        <p>Enter the Indian stock symbol </p>
        <input type="text" id="stockSymbol" placeholder="Enter symbol">
        <button onclick="fetchStockPrediction()">Predict</button>

        <h3 id="realTimeData"></h3>
        <h3 id="prediction"></h3>

        <canvas id="stockChart"></canvas>
    </div>

    <script>
        async function fetchStockPrediction() {{
            const symbol = document.getElementById("stockSymbol").value.trim().toUpperCase();
            if (!symbol) {{
                alert("Please enter a stock symbol.");
                return;
            }}

            const response = await fetch("/predict", {{
                method: "POST",
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify({{ symbol: symbol }}),
            }});

            const data = await response.json();
            if (!response.ok) {{
                alert(data.error);
                return;
            }}

            document.getElementById("realTimeData").innerText = 
                `Real-time Data: ${{symbol}}.NS - Close: ${{data.real_time_data.close}}, Volume: ${{data.real_time_data.volume}}`;
            
            document.getElementById("prediction").innerText = 
                `Prediction: ${{symbol}}.NS will move ${{data.prediction}}`;

            renderChart(data.dates, data.historical_prices, symbol);
        }}

        function renderChart(labels, data, symbol) {{
            const ctx = document.getElementById('stockChart').getContext('2d');
            if (window.stockChartInstance) {{
                window.stockChartInstance.destroy();
            }}
            window.stockChartInstance = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: `${{symbol}}.NS Closing Prices`,
                        data: data,
                        borderColor: "blue",
                        fill: false
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false
                }}
            }});
        }}
    </script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    symbol = request.json.get("symbol", "").upper()
    if not symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    real_time_data = get_real_time_stock_data(symbol)
    if real_time_data is None:
        return jsonify({"error": "Failed to fetch real-time data"}), 400

    historical_data = get_historical_stock_data(symbol)
    if historical_data is None or historical_data.empty:
        return jsonify({"error": "Failed to fetch historical data"}), 400

    features, target, df = prepare_data(historical_data)
    if features is None or target is None or df is None or df.empty:
        return jsonify({"error": "Insufficient data for prediction"}), 400

    prediction = train_and_predict(features, target, real_time_data)
    if prediction is None:
        return jsonify({"error": "Prediction failed"}), 400

    return jsonify({
        "real_time_data": real_time_data,
        "prediction": prediction,
        "historical_prices": df["Close"].tolist(),
        "dates": df.index.strftime("%Y-%m-%d").tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

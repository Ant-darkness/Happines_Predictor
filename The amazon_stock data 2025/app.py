from flask import Flask, render_template,request, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)

# Load Model

model = pickle.load(open("Amazon stock data 2000-2025.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():

    try:
        if request.method == "POST":
            # 'open', 'high', 'low', 'close', 'adj_close'

            opener = float(request.form['open'])
            high = float(request.form['high'])
            low = float(request.form['low'])
            close = float(request.form['close'])
            adj_close = float(request.form['adj_close'])

            features = [[opener, high, low, close, adj_close]]

            prediction = model.predict(features)[0]

            return render_template("result.html", prediction=round(prediction, 2))
        return redirect(url_for("home"))
    except Exception as e:
        return f"error! {e}"


if __name__ == "__main__":
    app.run(debug=True)
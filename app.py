from flask import Flask, render_template
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("lstm_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("last_sequence.pkl", "rb") as f:
    last_sequence = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    next_pred_scaled = model.predict(last_sequence)[0][0]
    next_pred = scaler.inverse_transform([[next_pred_scaled]])[0][0]

    return f"Next predicted stock price: ${next_pred:.2f}"

if __name__ == "__main__":
    app.run(debug=True)
    
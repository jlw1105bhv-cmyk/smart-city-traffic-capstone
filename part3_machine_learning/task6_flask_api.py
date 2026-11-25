from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

# Simple rule-based prediction
# Simulates deployed model

@app.route("/predict")

def predict():

    hour = int(
        request.args.get(
            "hour",
            12
        )
    )

    temp = float(
        request.args.get(
            "temp",
            290
        )
    )

    weather = request.args.get(
        "weather",
        "Clear"
    )

    prediction = "Medium"

    if hour in [7, 8, 16, 17, 18]:

        prediction = "High"

    elif weather in [
        "Rain",
        "Snow",
        "Thunderstorm"
    ]:

        prediction = "High"

    return jsonify({

        "hour": hour,

        "temp": temp,

        "weather": weather,

        "predicted_congestion": prediction

    })


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )
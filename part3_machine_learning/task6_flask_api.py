from flask import Flask
from flask import request
from flask import jsonify

import logging

from utils.logger_config import (
    setup_logging
)

setup_logging(
    "logs/task6_flask_api.log"
)

logger = logging.getLogger(__name__)

app = Flask(__name__)


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

    logger.info(
        f"Prediction request: "
        f"hour={hour}, "
        f"temp={temp}, "
        f"weather={weather}, "
        f"prediction={prediction}"
    )

    return jsonify({

        "hour": hour,

        "temp": temp,

        "weather": weather,

        "predicted_congestion": prediction

    })


if __name__ == "__main__":

    logger.info(
        "Starting Flask API on port 5001"
    )

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )
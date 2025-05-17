from flask import Flask, request, jsonify
from flask_cors import CORS
from machine import Machine

app = Flask(__name__)
CORS(app)


@app.get("/api/rms")
def get_rms_historic():
    Machine.data_return()


@app.post("/api/train")
def post_training():
    if request.is_json():
        data = request.get_json()

        required_fields = [
            "niu",
            "alfa",
            "rms",
            "epochs",
            "upper_limit",
            "lower_limit",
            "hidden_neurons",
            "training_patterns",
            "input_neurons",
        ]

        if not all(field in data for field in required_fields):
            return jsonify({"error": "Datos faltantes!"}), 400

    Machine.learn(
        training_patterns=data["training_patterns"],
        alfa=data["alfa"],
        hidden_neurons=data["hidden_neurons"],
        input_neurons=data["input_neurons"],
        lower_limit=data["lower_limit"],
        upper_limit=data["upper_limit"],
        max_epochs=data["epochs"],
        niu=data["niu"],
        output_neurons=3,
        rms=data["rms"],
    )


@app.post("/api/predict")
def post_predict():
    if request.is_json():
        data = request.get_json()

        if "input_data" not in data:
            return jsonify({"error": "Falta ingresar su entrada!"}), 400

        Machine.predict(
            hidden_neurons=data["hidden_neurons"],
            input_neurons=data["input_neurons"],
            inputs=data["inputs"],
            output_neurons=data["output_neurons"],
        )


if __name__ == "__main__":
    app.run(debug=True)

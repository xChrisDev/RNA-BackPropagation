import pandas as pd
import numpy as np
from propagation_values import hidden_propagation, output_propagation
from weights import get_weights
from normalize_patterns import normalize
from back_propagation import adapt_weights
from save_values import save, load
from flask import jsonify


class Machine:
    rms_history = []

    @staticmethod
    def learn(
        training_patterns,
        input_neurons,
        hidden_neurons,
        output_neurons,
        max_epochs,
        rms,
        alfa,
        niu,
        lower_limit,
        upper_limit,
    ):
        pd.DataFrame(training_patterns).to_csv(
            "data/training_patterns.csv", index=False, header=False, mode="w"
        )
        normalized_patterns, num_patterns = normalize("data/training_patterns.csv")
        tk = pd.read_csv("data/target.csv", header=None, dtype=float).values
        epoch = 1
        is_training = True
        Machine.rms_history = []

        wij, wjk, theta_j, theta_k = get_weights(
            input_neurons, hidden_neurons, output_neurons, lower_limit, upper_limit
        )

        while is_training and epoch < max_epochs:
            rms_obtained = 0.0

            sj = hidden_propagation(
                normalized_patterns,
                wij,
                theta_j,
                input_neurons,
                hidden_neurons,
                num_patterns,
            )

            sk = output_propagation(
                sj, wjk, theta_k, hidden_neurons, output_neurons, num_patterns
            )

            for i in range(num_patterns):
                for k in range(output_neurons):
                    rms_obtained += (tk[i][k] - sk[i][k]) ** 2

            rms_obtained = rms_obtained / (num_patterns * output_neurons)
            Machine.rms_history.append(rms_obtained)

            if rms_obtained <= rms:
                save(wij, wjk, theta_j, theta_k)
                is_training = False
            else:
                wij, wjk, theta_j, theta_k = adapt_weights(
                    input_neurons,
                    hidden_neurons,
                    output_neurons,
                    num_patterns,
                    sj,
                    sk,
                    normalized_patterns,
                    alfa,
                    niu,
                    wjk,
                    theta_k,
                    wij,
                    theta_j,
                )
                epoch += 1
        return jsonify({"message": f"Red entrenada con {epoch} epocas"}), 200

    @staticmethod
    def predict(input_neurons, hidden_neurons, output_neurons, inputs):
        try:
            pd.DataFrame(inputs).to_csv(
                "data/input_values.csv", index=False, header=False, mode="w"
            )

            wij, wjk, theta_j, theta_k = load()
            normalized_patterns_input, num_patterns_input = normalize(
                "data/input_values.csv"
            )
            tk = pd.read_csv("data/target.csv", header=None, dtype=float).values

            sj = hidden_propagation(
                normalized_patterns_input,
                wij,
                theta_j,
                input_neurons,
                hidden_neurons,
                num_patterns_input,
            )
            sk = output_propagation(
                sj, wjk, theta_k, hidden_neurons, output_neurons, num_patterns_input
            )

            predictions = np.zeros((num_patterns_input, output_neurons))
            for j in range(num_patterns_input):
                for i in range(output_neurons):
                    predictions[j][i] = 1 if sk[j][i] >= 0.5 else 0

            acum = 0
            result = ""
            for i in range(num_patterns_input):
                acum = 0
                for j in range(output_neurons):
                    if predictions[i][j] == tk[i][j]:
                        acum += 1
                if acum == output_neurons:
                    match i:
                        case 0:
                            result = "A"
                        case 1:
                            result = "E"
                        case 2:
                            result = "I"
                        case 3:
                            result = "O"
                        case 4:
                            result = "U"

            return jsonify({"prediction": result}), 200

        except Exception as e:
            return jsonify(
                {
                    "error": "Debes entrenar la red primero o verificar que existan los archivos de pesos."
                }
            ), 400

    @staticmethod
    def data_return():
        return jsonify({"rms": Machine.rms_history}), 200

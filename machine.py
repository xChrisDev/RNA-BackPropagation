import pandas as pd
import numpy as np
from propagation_values import hidden_propagation, output_propagation
from weights import get_weights
from normalize_patterns import normalize
from back_propagation import adapt_weights
from save_values import save, load


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
        normalized_patterns, num_patterns = normalize("data/training_patterns.csv", is_training=True)
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
                print(f"Is training in epoch: {epoch} & RMS: {rms_obtained}")
        return {"message": f"Red entrenada con {epoch} epocas", "rms_history": Machine.rms_history}

    @staticmethod
    def predict(input_neurons, hidden_neurons, output_neurons, inputs):
        try:
            print("Cargar inputs...")
            pd.DataFrame(inputs).to_csv(
                "data/input_values.csv", index=False, header=False, mode="w"
            )

            print("Normalizando...")
            wij, wjk, theta_j, theta_k = load()
            normalized_patterns_input, num_patterns_input = normalize("data/input_values.csv", is_training=False)

            tk = pd.read_csv("data/target.csv", header=None, dtype=float).values

            print("Valores de propagación...")
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

            print("Predicciones...")
            predictions = np.zeros((num_patterns_input, output_neurons))
            for j in range(num_patterns_input):
                for i in range(output_neurons):
                    predictions[j][i] = 1 if sk[j][i] >= 0.5 else 0

            pred = predictions[0].tolist()

            if pred == [0, 0, 1]:
                result = "Vocal A"
            elif pred == [0, 1, 0]:
                result = "Vocal E"
            elif pred == [1, 0, 0]:
                result = "Vocal I"
            elif pred == [1, 1, 0]:
                result = "Vocal O"
            elif pred == [1, 1, 1]:
                result = "Vocal U"
            else:
                result = "Desconocido"

            print(f"Prediction: {result} & values {pred}")
            return {"prediction": result}

        except Exception as e:
            print(e)
            return {"error": e}

    @staticmethod
    def data_return():
        return {"rms": Machine.rms_history}

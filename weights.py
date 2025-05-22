import pandas as pd
import numpy as np

def get_weights(input_layers, hidden_layers, output_layers, lower_limit, upper_limit):
    def generate_random_weights():
        wij = np.random.uniform(low=lower_limit, high=upper_limit, size=(input_layers, hidden_layers))
        wjk = np.random.uniform(low=lower_limit, high=upper_limit, size=(hidden_layers, output_layers))
        theta_j = np.random.uniform(low=lower_limit, high=upper_limit, size=hidden_layers)
        theta_k = np.random.uniform(low=lower_limit, high=upper_limit, size=output_layers)
        return wij, wjk, theta_j, theta_k

    try:
        wij = pd.read_csv("data/weights_ij.csv", header=None, dtype=float).values
        wjk = pd.read_csv("data/weights_jk.csv", header=None, dtype=float).values
        theta_j = pd.read_csv("data/theta_j.csv", header=None, dtype=float).values.flatten()
        theta_k = pd.read_csv("data/theta_k.csv", header=None, dtype=float).values.flatten()

        # Validar formas
        if wij.shape != (input_layers, hidden_layers) or \
           wjk.shape != (hidden_layers, output_layers) or \
           theta_j.shape[0] != hidden_layers or \
           theta_k.shape[0] != output_layers:
            print("⚠️ Pesos anteriores no coinciden con la configuracion actual. Regenerando...")
            return generate_random_weights()

    except Exception:
        print("⚠️ No se pudieron cargar los pesos anteriores. Generando nuevos...")
        return generate_random_weights()

    return wij, wjk, theta_j, theta_k

import numpy as np


def hidden_propagation(normalized_patterns, wij, theta_j, input_neurons, hidden_neurons, num_patterns):
    rj = np.zeros((num_patterns, hidden_neurons))
    sj = np.zeros((num_patterns, hidden_neurons))

    for p in range(num_patterns):  # Para cada patrón
        for j in range(hidden_neurons):  # Para cada neurona oculta
            suma = 0
            for i in range(input_neurons):  # Para cada entrada
                suma += normalized_patterns[p][i] * wij[i][j]
            suma += theta_j[j]
            rj[p][j] = suma
            sj[p][j] = 1 / (1 + np.exp(-suma))
    return sj



def output_propagation(sj, wjk, theta_k, hidden_neurons, output_neurons, num_patterns):
    rk = np.zeros((num_patterns, output_neurons))
    sk = np.zeros((num_patterns, output_neurons))

    for p in range(num_patterns):  # Para cada patrón
        for k in range(output_neurons):  # Para cada neurona de salida
            suma = 0
            for j in range(hidden_neurons):  # Para cada neurona oculta
                suma += sj[p][j] * wjk[j][k]
            suma += theta_k[k]
            rk[p][k] = suma
            sk[p][k] = 1 / (1 + np.exp(-suma)) 
    return sk


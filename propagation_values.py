from messages import message_color
from colorama import Fore
import numpy as np

def hidden_propagation(normalized_patterns, wij, theta_j, input_layers, hidden_layers, num_patterns):
    rj = np.zeros((num_patterns, hidden_layers))
    sj = np.zeros((num_patterns, hidden_layers))
    pattern_index = 0
    value_index = 0

    for j in range(hidden_layers):
        for i in range(input_layers):
            for pattern in normalized_patterns:
                w_sum = 0
                for value in pattern:
                    w_sum += wij[value_index][j] * value
                    value_index += 1
                value_index = 0
                w_sum += theta_j[j]
                rj[pattern_index][j] = w_sum
                sj[pattern_index][j] = 1 / (1 + np.exp(-rj[pattern_index][j]))
                pattern_index += 1
            pattern_index = 0

    message_color("PROPAGACIÓN - CAPA OCULTA (Rj)", Fore.CYAN)
    print(rj)
    message_color("PROPAGACIÓN - CAPA OCULTA (Sj)", Fore.CYAN)
    print(sj)

    return rj, sj

def output_propagation(sj, wjk, theta_k, hidden_layers, output_layers, num_patterns):
    rk = np.zeros((num_patterns, output_layers))
    sk = np.zeros((num_patterns, output_layers))
    pattern_index = 0
    value_index = 0

    for j in range(output_layers):
        for i in range(hidden_layers):
            for pattern in sj:
                w_sum = 0
                for value in pattern:
                    w_sum += wjk[value_index][j] * value
                    value_index += 1
                value_index = 0
                w_sum += theta_k[j]
                rk[pattern_index][j] = w_sum
                sk[pattern_index][j] = 1 / (1 + np.exp(-rk[pattern_index][j]))
                pattern_index += 1
            pattern_index = 0

    message_color("PROPAGACIÓN - CAPA SALIDA (Rk)", Fore.CYAN)
    print(rk)
    message_color("PROPAGACIÓN - CAPA SALIDA (Sk)", Fore.CYAN)
    print(sk)

    return rk, sk

import numpy as np
import pandas as pd
from colorama import Fore
from messages import message_color


def adapt_weights(
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
):
    tk = pd.read_csv("data/target.csv", header=None, dtype=float).values

    delta_wij = np.zeros((input_neurons, hidden_neurons))
    delta_wjk = np.zeros((hidden_neurons, output_neurons))
    delta_theta_j = np.zeros(hidden_neurons)
    delta_theta_k = np.zeros(output_neurons)
    temp_delta_wij = np.zeros((input_neurons, hidden_neurons))
    temp_delta_wjk = np.zeros((hidden_neurons, output_neurons))
    temp_delta_theta_j = np.zeros(hidden_neurons)
    temp_delta_theta_k = np.zeros(output_neurons)
    error_k = np.zeros((num_patterns, output_neurons))
    error_j = np.zeros((num_patterns, hidden_neurons))

    # Errores en K
    for i in range(num_patterns):
        for k in range(output_neurons):
            error_k[i][k] = (tk[i][k] - sk[i][k]) * sk[i][k] * (1 - sk[i][k])

    # Errores en J
    for i in range(num_patterns):
        for j in range(hidden_neurons):
            sum_k = 0.0
            for k in range(output_neurons):
                sum_k += error_k[i][k] * wjk[j][k]
            error_j[i][j] = sj[i][j] * (1 - sj[i][j]) * sum_k

    # Ajuste de pesos capa de salida
    for i in range(num_patterns):
        for j in range(hidden_neurons):
            for k in range(output_neurons):
                delta_wjk[j][k] = (niu * error_k[i][k] * sj[i][j]) + (alfa * temp_delta_wjk[j][k])
                temp_delta_wjk[j][k] = delta_wjk[j][k]
                wjk[j][k] += delta_wjk[j][k]

        for k in range(output_neurons):
            delta_theta_k[k] = (niu * error_k[i][k]) + (alfa * temp_delta_theta_k[k])
            temp_delta_theta_k[k] = delta_theta_k[k]
            theta_k[k] += delta_theta_k[k]

    # Ajuste de pesos capa oculta 
    for i in range(num_patterns):
        for n in range(input_neurons):
            for j in range(hidden_neurons):
                delta_wij[n][j] = (niu * error_j[i][j] * normalized_patterns[i][n]) + (alfa * temp_delta_wij[n][j])
                temp_delta_wij[n][j] = delta_wij[n][j]
                wij[n][j] += delta_wij[n][j]

        for j in range(hidden_neurons):
            delta_theta_j[j] = (niu * error_j[i][j]) + (alfa * temp_delta_theta_j[j])
            temp_delta_theta_j[j] = delta_theta_j[j]
            theta_j[j] += delta_theta_j[j]

    # message_color("ERROR K", Fore.BLUE)
    # print(error_k)
    # message_color("WJK", Fore.BLUE)
    # print(wjk)
    # message_color("Δ WJK", Fore.BLUE)
    # print(delta_wjk)
    # message_color("θK", Fore.BLUE)
    # print(theta_k)
    # message_color("Δ θK", Fore.BLUE)
    # print(delta_theta_k)
    # message_color("ERROR J", Fore.GREEN)
    # print(error_j)
    # message_color("WIJ", Fore.GREEN)
    # print(wij)
    # message_color("Δ WIJ", Fore.GREEN)
    # print(delta_wij)
    # message_color("θJ", Fore.GREEN)
    # print(theta_j)
    # message_color("Δ θJ", Fore.GREEN)
    # print(delta_theta_j)

    return wij, wjk, theta_j, theta_k

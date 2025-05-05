import numpy as np
import pandas as pd

def adapt_weights(input_layers, hidden_layers, output_layers, num_patterns, rj, sj, rk, sk, normalized_patterns, alfa, niu):
    delta_wij = np.zeros((input_layers, hidden_layers))
    delta_wjk = np.zeros((hidden_layers, output_layers))
    delta_theta_j = np.zeros(hidden_layers)
    delta_theta_k = np.zeros(output_layers)
    wij = 0
    wjk = 0
    theta_j = 0
    theta_k = 0
    return wij, wjk, theta_j, theta_k
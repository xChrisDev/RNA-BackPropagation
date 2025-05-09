from messages import message_color
from colorama import Fore
import pandas as pd
import numpy as np

def get_weights(input_layers, hidden_layers, output_layers):
    try:
        wij = pd.read_csv("data/weights_ij.csv", header=None, dtype=float).values
        wjk = pd.read_csv("data/weights_jk.csv", header=None, dtype=float).values
        theta_j = pd.read_csv("data/theta_j.csv", header=None, dtype=float).values.flatten()
        theta_k = pd.read_csv("data/theta_k.csv", header=None, dtype=float).values.flatten()
    except Exception:
        wij = np.random.uniform(low=-0.3, high=0.3, size=(input_layers, hidden_layers))
        wjk = np.random.uniform(low=-0.3, high=0.3, size=(hidden_layers, output_layers))
        theta_j = np.random.uniform(low=-0.3, high=0.3, size=hidden_layers)
        theta_k = np.random.uniform(low=-0.3, high=0.3, size=output_layers)

        # wij = pd.read_csv("data/weights_ij.csv", header=None, dtype=float).values
        # wjk = pd.read_csv("data/weights_jk.csv", header=None, dtype=float).values
        # theta_j = pd.read_csv("data/theta_j.csv", header=None, dtype=float).values.flatten()
        # theta_k = pd.read_csv("data/theta_k.csv", header=None, dtype=float).values.flatten()

    # message_color("PESOS INICIALES WIJ", Fore.MAGENTA)
    # print(wij)
    # message_color("PESOS INICIALES WJK", Fore.MAGENTA)
    # print(wjk)

    return wij, wjk, theta_j, theta_k

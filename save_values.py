import pandas as pd
from colorama import Fore
from messages import message_color

def save(wij, wjk, theta_j, theta_k):
    pd.DataFrame(wij).to_csv("data/weights_ij.csv", index=False, header=False, mode='w')
    pd.DataFrame(wjk).to_csv("data/weights_jk.csv", index=False, header=False, mode='w')
    pd.DataFrame(theta_j).to_csv("data/theta_j.csv", index=False, header=False, mode='w')
    pd.DataFrame(theta_k).to_csv("data/theta_k.csv", index=False, header=False, mode='w')
    message_color("PESOS GUARDADOS", Fore.MAGENTA)

def load():
    try:
        wij = pd.read_csv("data/weights_ij.csv", header=None).values
        wjk = pd.read_csv("data/weights_jk.csv", header=None).values
        theta_j = pd.read_csv("data/theta_j.csv", header=None).values.flatten()
        theta_k = pd.read_csv("data/theta_k.csv", header=None).values.flatten()
        message_color("PESOS CARGADOS", Fore.MAGENTA)
        return wij, wjk, theta_j, theta_k
    except FileNotFoundError:
        message_color("Error: No se encontraron archivos de pesos guardados.", Fore.RED)
        raise

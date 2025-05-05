from colorama import Fore
from messages import message_color
import pandas as pd


def save(wij, wjk, theta_j, theta_k):
    pd.DataFrame(wij).to_csv("data/weights_ij.csv", index=False, header=False)
    pd.DataFrame(wjk).to_csv("data/weights_jk.csv", index=False, header=False)
    pd.DataFrame(theta_j).to_csv("data/theta_j.csv", index=False, header=False)
    pd.DataFrame(theta_k).to_csv("data/theta_k.csv", index=False, header=False)
    message_color("PESOS GUARDADOS", Fore.MAGENTA)

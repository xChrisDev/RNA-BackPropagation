import pandas as pd

def save(wij, wjk, theta_j, theta_k):
    pd.DataFrame(wij).to_csv("data/weights_ij.csv", index=False, header=False, mode='w')
    pd.DataFrame(wjk).to_csv("data/weights_jk.csv", index=False, header=False, mode='w')
    pd.DataFrame(theta_j).to_csv("data/theta_j.csv", index=False, header=False, mode='w')
    pd.DataFrame(theta_k).to_csv("data/theta_k.csv", index=False, header=False, mode='w')

def load():
    try:
        wij = pd.read_csv("data/weights_ij.csv", header=None).values
        wjk = pd.read_csv("data/weights_jk.csv", header=None).values
        theta_j = pd.read_csv("data/theta_j.csv", header=None).values.flatten()
        theta_k = pd.read_csv("data/theta_k.csv", header=None).values.flatten()
        return wij, wjk, theta_j, theta_k
    except FileNotFoundError:
        raise

import pandas as pd
import numpy as np
import os

def normalize(filename, is_training, max_file="data/max_values.csv"):
    raw_patterns = pd.read_csv(filename, header=None, dtype=float).values
    num_patterns = raw_patterns.shape[0]

    if is_training:
        max_values = np.max(raw_patterns, axis=0)
        max_values[max_values == 0] = 1
        pd.DataFrame([max_values]).to_csv(max_file, index=False, header=False)
    else:
        if not os.path.exists(max_file):
            raise FileNotFoundError("No se encontró el archivo con valores máximos para normalización.")
        max_values = pd.read_csv(max_file, header=None).values[0]

    normalized_patterns = raw_patterns / max_values
    return normalized_patterns, num_patterns

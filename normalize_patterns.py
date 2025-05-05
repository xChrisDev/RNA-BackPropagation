from messages import message_color
from colorama import Fore
import pandas as pd
import numpy as np

def normalize(filename):
    raw_patterns = pd.read_csv(filename, header=None, dtype=float).values
    num_patterns = raw_patterns.shape[0]

    flag = False
    for pattern in raw_patterns:
        for value in pattern:
            if value > 1:
                flag = True
                break
        if flag:
            break

    if flag:
        max_pattern_values = np.max(raw_patterns, axis=0)
        max_pattern_values[max_pattern_values == 0] = 1 
        normalized_patterns = raw_patterns / max_pattern_values
    else:
        normalized_patterns = raw_patterns.copy()

    return normalized_patterns, num_patterns

from messages import message_color
from colorama import Fore
import pandas as pd
import numpy as np

def normalize(filename):
    raw_patterns = pd.read_csv(filename, header=None, dtype=float).values
    normalized_patterns = np.zeros((raw_patterns.shape[0], raw_patterns.shape[1]))
    max_pattern_values = [0 for _ in range(len(raw_patterns[0]))]
    index = 0
    num_patterns = 0

    message_color("PATRONES SIN NORMALIZAR", Fore.YELLOW)
    print(raw_patterns)

    for pattern in raw_patterns:
        for value in pattern:
            if value >= max_pattern_values[index]:
                max_pattern_values[index] = value
            index += 1
        index = 0
        num_patterns += 1

    for i in range(raw_patterns.shape[0]):
        for j in range(raw_patterns.shape[1]):
            normalized_patterns[i][j] = raw_patterns[i][j] / max_pattern_values[j]

    message_color("PATRONES NORMALIZADOS", Fore.GREEN)
    print(normalized_patterns)

    return normalized_patterns, num_patterns
import pandas as pd
import matplotlib.pyplot as plt
from propagation_values import hidden_propagation, output_propagation
from normalize_patterns import normalize
from weights import get_weights
from back_propagation import adapt_weights
from colorama import init, Fore
from messages import message_color
from save_values import save

init(autoreset=True)


def main():
    normalized_patterns, num_patterns = normalize("data/training_patterns.csv")
    tk = pd.read_csv("data/target.csv", header=None, dtype=float).values
    input_neurons = 3
    hidden_neurons = 2
    output_neurons = 2
    rms = 0.01
    alfa = 0.3
    niu = 0.7
    epoch = 1
    max_epochs = 10000
    is_training = True
    rms_obtained = 0.0
    rms_history = []
    
    wij, wjk, theta_j, theta_k = get_weights(
        input_neurons, hidden_neurons, output_neurons
    )

    while is_training and epoch < max_epochs:
        rms_obtained = 0.0

        rj, sj = hidden_propagation(
            normalized_patterns, wij, theta_j, input_neurons, hidden_neurons, num_patterns
        )

        rk, sk = output_propagation(
            sj, wjk, theta_k, hidden_neurons, output_neurons, num_patterns
        )

        for i in range(num_patterns):
            for k in range(output_neurons):
                rms_obtained += (tk[i][k] - sk[i][k]) ** 2 

        rms_obtained = rms_obtained / (num_patterns * output_neurons)
        rms_history.append(rms_obtained)

        message_color(f"EPOCH {epoch}", Fore.BLUE)
        print(Fore.CYAN + f"RMS: {rms_obtained:.4f} | Target: {rms:.4f} | Epoch: {epoch}")
        df_sk = pd.DataFrame(sk, columns=[f'Output_{i+1}' for i in range(sk.shape[1])]).round(4)
        df_tk = pd.DataFrame(tk, columns=[f'Output_{i+1}' for i in range(tk.shape[1])]).round(4)

        print(Fore.CYAN + "\nSk:")
        print(df_sk.to_string(index=False))

        print(Fore.CYAN + "\nTk:")
        print(df_tk.to_string(index=False))


        if rms_obtained <= rms:
            save(wij, wjk, theta_j, theta_k)
            get_histogram_epochs(rms_history)
            message_color("Red entrenada.", Fore.GREEN)
            is_training = False
        else:
            wij, wjk, theta_j, theta_k = adapt_weights(
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
            )
            epoch += 1

def get_histogram_epochs(rms_history):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rms_history) + 1), rms_history, color='blue', linewidth=1)
    plt.title("RMS durante el entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("RMS")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

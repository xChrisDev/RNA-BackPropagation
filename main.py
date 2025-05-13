import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from propagation_values import hidden_propagation, output_propagation
from normalize_patterns import normalize
from weights import get_weights
from back_propagation import adapt_weights
from colorama import init, Fore
from messages import message_color
from save_values import save, load

init(autoreset=True)


def main():
    normalized_patterns, num_patterns = normalize("data/training_patterns.csv")
    tk = pd.read_csv("data/target.csv", header=None, dtype=float).values
    input_neurons = 3
    hidden_neurons = 2
    output_neurons = 2
    rms = 0.01
    alfa = 0.15
    niu = 0.5
    epoch = 1
    max_epochs = 10000
    is_training = True
    rms_obtained = 0.0
    rms_history = []
    
    option = 0
    while option != 3:
        message_color("SELECCIONE LA OPCIÓN\n1. Entrenamiento\n2. Reconocimiento\n3. Salir", Fore.BLUE)
        option = int(input())
        
        if option == 1:
            epoch = 1
            is_training = True

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
        elif option == 2:
            try:
                wij, wjk, theta_j, theta_k = load()
                normalized_patterns_input, num_patterns_input = normalize("data/input_values.csv")

                rj, sj = hidden_propagation(normalized_patterns_input, wij, theta_j, input_neurons, hidden_neurons, num_patterns_input)
                rk, sk = output_propagation(sj, wjk, theta_k, hidden_neurons, output_neurons, num_patterns_input)
                
                predictions = np.zeros((num_patterns_input, output_neurons))
                for j in range(num_patterns_input):
                    for i in range(output_neurons):
                        predictions[j][i] = 1 if sk[j][i] >= 0.5 else 0

                df_pred = pd.DataFrame(predictions, columns=[f'Output_{i+1}' for i in range(predictions.shape[1])]).round(0)
                print(Fore.CYAN + "\nPredicciones:")
                print(df_pred.to_string(index=False))
                
                sk_pred = pd.DataFrame(sk, columns=[f'Output_{i+1}' for i in range(sk.shape[1])]).round(0)
                print(Fore.CYAN + "\nSKs:")
                print(sk_pred.to_string(index=False))

            except Exception as e:
                message_color(f"Error: {str(e)}", Fore.RED)
                message_color("Debes entrenar la red primero o verificar que existan los archivos de pesos.", Fore.RED)  
            
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

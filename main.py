from propagation_values import hidden_propagation, output_propagation
from normalize_patterns import normalize
from weights import get_weights
from back_propagation import adapt_weights
from colorama import init


init(autoreset=True)


def main():
    input_layers = 3
    hidden_layers = 2
    output_layers = 2
    rms = 0.1
    alfa = 0.5
    niu = 0.75
    epochs = 10

    normalized_patterns, num_patterns = normalize("data/training_patterns.csv")
    wij, wjk, theta_j, theta_k = get_weights(input_layers, hidden_layers, output_layers)

    rj, sj = hidden_propagation(
        normalized_patterns, wij, theta_j, input_layers, hidden_layers, num_patterns
    )

    rk, sk = output_propagation(
        sj, wjk, theta_k, hidden_layers, output_layers, num_patterns
    )
    
    adapt_weights(
        input_layers,
        hidden_layers,
        output_layers,
        rj,
        sj,
        rk,
        sk,
        normalized_patterns,
        num_patterns,
        alfa,
        niu,
    )


if __name__ == "__main__":
    main()

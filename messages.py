from colorama import Fore, Style

def message_color(texto, color=Fore.CYAN):
    print("\n" + color + "="*40)
    print(color + texto)
    print(color + "="*40 + Style.RESET_ALL)
from colorama import Fore, Style

def message_color(texto, color=Fore.CYAN):
    print("\n" + color + "="*60)
    print(color + texto)
    print(color + "="*60 + Style.RESET_ALL)
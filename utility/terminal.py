from os import system, name

DEFAULT_SLEEP_SECS = 5


def clear_screen():
    if name == 'nt': # windows
        _ = system('cls')

    else: # mac and linux
        _ = system('clear')

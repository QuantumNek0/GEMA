from config import *


def clear_screen():
    """
    Clears the screen of the terminal
    """

    if os.name == 'nt': # windows
        _ = os.system('cls')

    else: # mac and linux
        _ = os.system('clear')


def confirmation(prompt: str) -> bool:
    """
    asks the user for a yes/no answer based on the ''prompt''

    :param prompt: the prompt to show the user, generally a question
    :return: returns ''True'' if the user answer is affirmative, else ''False''
    """
    prompt += " [Y/n] "

    user_input = input(prompt).casefold()
    clear_screen()

    return True if user_input == "y" else False


def select_option(options: [], header: str = "Options") -> int:
    """
    from a list of options, the user selects the index of the desired option

    :param options: array of ''string'' to show to the user
    :param header: the header of the options
    :return: returns the index of the desired option
    """

    valid_option = False
    options = list(options)

    while not valid_option:

        print(header)
        for i, option in enumerate(options):
            print(f"{i + 1}. {option}")

        selected_index = int(input("\n>> ")) - 1

        if selected_index < 0 or selected_index >= len(options):
            print("\nInvalid option!")
            time.sleep(DEFAULT_SLEEP_SECS)

        else:
            valid_option = True

        clear_screen()

    return selected_index


def are_all_positives(numbers: [], can_be_zero: bool = True) -> bool:
    """
    checks whether a given array is made of only positive numbers

    :param numbers: an array of numbers
    :param can_be_zero: if set to ''True'', zero will be a valid value for the array
    :return: returns ''True'' if the array ''numbers'' contains only valid values
    """

    for number in numbers:
        if can_be_zero:
            if number < 0: return False

        else:
            if number <= 0: return False

    return True


def alphanumeric_split(alphanum: str) -> (str, str):
    import re

    numeric_part = ''.join(re.findall(r'\d+', alphanum))
    non_numeric_part = ''.join(re.findall(r'\D+', alphanum))

    return numeric_part, non_numeric_part

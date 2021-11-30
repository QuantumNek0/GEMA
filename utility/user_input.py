import time
from utility.terminal import DEFAULT_SLEEP_SECS, clear_screen


def confirmation(prompt: str) -> bool:
    prompt += " [Y/n] "

    user_input = input(prompt).casefold()
    clear_screen()

    return True if user_input == "y" else False


def select_option(options: [], header: str = "Options") -> int:
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

    for number in numbers:
        if can_be_zero:
            if number < 0: return False

        else:
            if number <= 0: return False

    return True

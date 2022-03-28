from typing import List
from random import randint


def chord_type_from_degree(degree: int, scale_type: str) -> str:
    """
    returns whether the passed degree is major or minor depending on the scale type

    :param degree: degree of the harmonic progression
    :param scale_type: whether the scale is major or minor
    """
    if degree == 1:
        return "maj" if scale_type == "maj" else "min"

    elif degree == 2:
        return "min" if scale_type == "maj" else "dim"

    elif degree == 3:
        return "min" if scale_type == "maj" else "maj"

    elif degree == 4:
        return "maj" if scale_type == "maj" else "min"

    elif degree == 5:
        return "maj" if scale_type == "maj" else "min"

    elif degree == 6:
        return "min" if scale_type == "maj" else "maj"

    elif degree == 7:
        return "dim" if scale_type == "maj" else "maj"

    else:
        return "error"


def rand_harmonic_progression() -> List[int]:
    """
    Handcrafted set of harmonic progressions to be chosen at random

    :return: only returns progression of exactly 4 notes
    """
    r = randint(0, 10)

    if r == 0:                  # Happy in maj | Sad in min
        return [1, 4, 5, 1]
    elif r == 1:                # Uplifting in maj | Melancholy in min
        return [1, 5, 6, 4]
    elif r == 2:                # Happy in maj | Sad  in min
        return [1, 6, 4, 5]
    elif r == 3:                # Happy in maj | Sad in min
        return [1, 4, 5, 4]
    elif r == 4:                # Uplifting in maj | Sad in min
        return [6, 4, 1, 5]
    elif r == 5:                # Happy in maj | Sad in min
        return [1, 4, 2, 5]
    elif r == 6:                # Happy in maj | Sad in min
        return [1, 4, 1, 5]
    elif r == 7:                # Happy  in maj | Sad in min
        return [1, 5, 1, 4]
    elif r == 8:                # Sad in maj | Sad in min
        return [6, 5, 4, 3]
    elif r == 9:                # Uplifting in maj | Melancholy in min
        return [6, 4, 1, 5]
    elif r == 10:               # Uplifting in maj | Melancholy in min
        return [1, 4, 6, 5]


def mood_progression(scale_type: str, mood: str) -> List[int]:
    """
    Kind of classifies each progression depending on the feeling it gives

    :var scale_type: if the scale is either major or minor
    :var mood: the selected mood
    :return: only returns progression of exactly 4 notes
    """
    if mood == "happy":

        if scale_type == "maj":

            r = randint(0, 5)

            if r == 0:  # Happy in maj
                return [1, 4, 5, 1]
            elif r == 1:  # Happy in maj
                return [1, 6, 4, 5]
            elif r == 2:  # Happy in maj
                return [1, 4, 5, 4]
            elif r == 3:  # Happy in maj
                return [1, 4, 2, 5]
            elif r == 4:  # Happy in maj
                return [1, 4, 1, 5]
            elif r == 5:  # Happy  in maj
                return [1, 4, 2, 5]

    elif mood == "uplifting":

        if scale_type == "maj":

            r = randint(0, 3)

            if r == 0:  # Uplifting in maj
                return [1, 5, 6, 4]
            elif r == 1:  # Uplifting in maj
                return [6, 4, 1, 5]
            elif r == 2:  # Uplifting in maj
                return [6, 4, 1, 5]
            elif r == 3:  # Uplifting in maj
                return [1, 4, 6, 5]

    elif mood == "sad":

        if scale_type == "maj":
            return [6, 5, 4, 3] # Sad in maj

        if scale_type == "min":

            r = randint(0, 7)

            if r == 0:  # Sad in min
                return [1, 4, 5, 1]
            elif r == 1:  # Sad  in min
                return [1, 6, 4, 5]
            elif r == 2:  # Sad in min
                return [1, 4, 5, 4]
            elif r == 3:  # Sad in min
                return [6, 4, 1, 5]
            elif r == 4:  # Sad in min
                return [1, 4, 2, 5]
            elif r == 5:  # Sad in min
                return [1, 4, 1, 5]
            elif r == 6:  # Sad in min
                return [1, 5, 1, 4]
            elif r == 7:  # Sad in min
                return [6, 5, 4, 3]

    elif mood == "melancholy":

        if scale_type == "min":

            r = randint(0, 2)

            if r == 0:  # Melancholy in min
                return [1, 5, 6, 4]
            elif r == 1:  # Melancholy in min
                return [6, 4, 1, 5]
            elif r == 2:  # Melancholy in min
                return [1, 4, 6, 5]

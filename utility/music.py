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

    if r == 0:
        return [1, 4, 5, 1]
    elif r == 1:
        return [1, 5, 6, 4]
    elif r == 2:
        return [1, 6, 4, 5]
    elif r == 3:
        return [1, 4, 5, 4]
    elif r == 4:
        return [6, 4, 1, 5]
    elif r == 5:
        return [1, 4, 2, 5]
    elif r == 6:
        return [1, 4, 1, 5]
    elif r == 7:
        return [1, 5, 1, 4]
    elif r == 8:
        return [6, 5, 4, 3]
    elif r == 9:
        return [6, 4, 1, 5]
    elif r == 10:
        return [1, 4, 6, 5]
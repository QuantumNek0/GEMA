from config import *

alpha_maj_keys = {
    "1B": 12,
    "2B": 7,
    "3B": 2,
    "4B": 9,
    "5B": 4,
    "6B": 11,
    "7B": 6,
    "8B": 1,
    "9B": 8,
    "10B": 3,
    "11B": 10,
    "12B": 5,
}

short_maj_keys = {
    "B": 12,
    "F#": 7,
    "Db": 2,
    "Ab": 9,
    "Eb": 4,
    "Bb": 11,
    "F": 6,
    "C": 1,
    "G": 8,
    "D": 3,
    "A": 10,
    "E": 5,
}

alpha_to_short_key = {
    "1A": "Abm",
    "1B": "B",
    "2A": "Ebm",
    "2B": "F#",
    "3A": "Bbm",
    "3B": "Db",
    "4A": "Fm",
    "4B": "Ab",
    "5A": "Cm",
    "5B": "Eb",
    "6A": "Gm",
    "6B": "Bb",
    "7A": "Dm",
    "7B": "F",
    "8A": "Am",
    "8B": "C",
    "9A": "Em",
    "9B": "G",
    "10A": "Bm",
    "10B": "D",
    "11A": "F#m",
    "11B": "A",
    "12A": "C#m",
    "12B": "E",
}

relative_min_key = {
    "B": "Abm",
    "F#": "Ebm",
    "Db": "Bbm",
    "Ab": "Fm",
    "Eb": "Cm",
    "Bb": "Gm",
    "F": "Dm",
    "C": "Am",
    "G": "Em",
    "D": "Bm",
    "A": "F#m",
    "E": "C#m",
}

short_key_to_accidentals = {
    "B": (5, SHARPS, MAJOR),
    "Abm": (7, FLATS, MINOR),

    "F#": (6, SHARPS, MAJOR),
    "Ebm": (6, FLATS, MINOR),

    "Db": (5, FLATS, MAJOR),
    "Bbm": (5, FLATS, MINOR),

    "Ab": (4, FLATS, MAJOR),
    "Fm": (4, FLATS, MINOR),

    "Eb": (3, FLATS, MAJOR),
    "Cm": (3, FLATS, MINOR),

    "Bb": (2, FLATS, MAJOR),
    "Gm": (2, FLATS, MINOR),

    "F": (1, FLATS, MAJOR),
    "Dm": (1, FLATS, MINOR),

    "C": (0, SHARPS, MAJOR),
    "Am": (0, SHARPS, MINOR),

    "G": (1, SHARPS, MAJOR),
    "Em": (1, SHARPS, MINOR),

    "D": (2, SHARPS, MAJOR),
    "Bm": (2, SHARPS, MINOR),

    "A": (3, SHARPS, MAJOR),
    "F#m": (3, SHARPS, MINOR),

    "E": (4, SHARPS, MAJOR),
    "C#m": (4, SHARPS, MINOR),
}


def find_in_dict(dictionary, value):
    for key in dictionary:
        if dictionary[key] == value:
            return key
    return None

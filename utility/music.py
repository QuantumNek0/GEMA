from config import *

alpha_keys = {
    "1B": 11,
    "1A": 11,
    "2B": 6,
    "2A": 6,
    "3B": 1,
    "3A": 1,
    "4B": 8,
    "4A": 8,
    "5B": 3,
    "5A": 3,
    "6B": 10,
    "6A": 10,
    "7B": 5,
    "7A": 5,
    "8B": 0,
    "8A": 0,
    "9B": 7,
    "9A": 7,
    "10B": 2,
    "10A": 2,
    "11B": 9,
    "11A": 9,
    "12B": 4,
    "12A": 4,
}

maj_alpha_keys = {
    "1B": 11,
    "2B": 6,
    "3B": 1,
    "4B": 8,
    "5B": 3,
    "6B": 10,
    "7B": 5,
    "8B": 0,
    "9B": 7,
    "10B": 2,
    "11B": 9,
    "12B": 4,
}

min_alpha_keys = {
    "1A": 11,
    "2A": 6,
    "3A": 1,
    "4A": 8,
    "5A": 3,
    "6A": 10,
    "7A": 5,
    "8A": 0,
    "9A": 7,
    "10A": 2,
    "11A": 9,
    "12A": 4,
}

short_maj_keys = {
    "B": 11,
    "F#": 6,
    "Db": 1,
    "Ab": 8,
    "Eb": 3,
    "Bb": 10,
    "F": 5,
    "C": 0,
    "G": 7,
    "D": 2,
    "A": 9,
    "E": 4,
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
    "E": "C#m"
}

relative_maj_key = {
    "Abm": "B",
    "Ebm": "F#",
    "Bbm": "Db",
    "Fm": "Ab",
    "Cm": "Eb",
    "Gm": "Bb",
    "Dm": "F",
    "Am": "C",
    "Em": "G",
    "Bm": "D",
    "F#m": "A",
    "C#m": "E"
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

from config import *

maj_key_encoding = {
    "B": 5,
    "F#": 6,
    "Db": 7,
    "Ab": 8,
    "Eb": 9,
    "Bb": 10,
    "F": 11,
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
}

min_key_encoding = {
    "Abm": 5,
    "Ebm": 6,
    "Bbm": 7,
    "Fm": 8,
    "Cm": 9,
    "Gm": 10,
    "Dm": 11,
    "Am": 0,
    "Em": 1,
    "Bm": 2,
    "F#m": 3,
    "C#m": 4,
}


key_encoding_midi = {
    5: 71,
    6: 66,
    7: 61,
    8: 68,
    9: 63,
    10: 70,
    11: 65,
    0: 60,
    1: 67,
    2: 62,
    3: 69,
    4: 64,
}

key_encoding_key = {
    5: "B",
    6: "F#",
    7: "Db",
    8: "Ab",
    9: "Eb",
    10: "Bb",
    11: "F",
    0: "C",
    1: "G",
    2: "D",
    3: "A",
    4: "E",
}

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


# Midi notes
class MidiValues:
    """

    a collection of midi values to be easily referenced based on the note/scale they represent
    i.e c[4] = 60 = C4

    """

    # Natural
    C = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
    D = [14, 26, 38, 50, 62, 74, 86, 98, 110, 122]
    E = [16, 28, 40, 52, 64, 76, 88, 100, 112, 124]
    F = [17, 29, 41, 53, 65, 77, 89, 101, 113, 125]
    G = [19, 31, 43, 55, 67, 79, 91, 103, 115, 127]
    A = [21, 33, 45, 57, 69, 81, 93, 105, 117]
    B = [23, 35, 47, 59, 71, 83, 95, 107, 119]

    # Sharp
    Cs = [13, 25, 37, 49, 61, 73, 85, 97, 109, 121]
    Ds = [15, 27, 39, 51, 63, 75, 87, 99, 111, 123]
    Es = F
    Fs = [18, 30, 42, 54, 66, 78, 90, 102, 114, 126]
    Gs = [20, 32, 44, 56, 68, 80, 92, 104, 116]
    As = [22, 34, 46, 58, 70, 82, 94, 106, 118]
    Bs = [24, 36, 48, 60, 72, 84, 96, 108, 120]

    # Flat
    Cb = [11, 23, 35, 47, 59, 71, 83, 95, 107, 119]
    Db = Cs
    Eb = Ds
    Fb = E
    Gb = Fs
    Ab = Gs
    Bb = As

    # Rest
    rest = 0

    # Music Keys
    Key = namedtuple("Key", ["name", "values"])

    key = { # dictionary of scales
        "C": {
            "natural": {
                "maj": [C, D, E, F, G, A, B],
                "min": [C, D, Eb, F, G, Ab, Bb]
            },
            "#": {
                "maj": [Cs, Ds, Es, Fs, Gs, As, Bs],
                "min": [Cs, Ds, E, Fs, Gs, A, B]
            },
            "b": {
                "maj": [Cb, Db, Eb, Fb, Gb, Ab, Bb],
                "min": [Cb, Db, D, Fb, Gb, G, A]
            }
        },

        "D": {
            "natural": {
                "maj": [D, E, Fs, G, A, B, Cs],
                "min": [D, E, F, G, A, Bb, C]
            },
            "#": {
                "maj": [Ds, Es, G, Gs, As, Bs, D],
                "min": [Ds, Es, Fs, Gs, As, B, Cs]
            },
            "b": {
                "maj": [Db, Eb, F, Gb, Ab, Bb, C],
                "min": [Db, Eb, Fb, Gb, Ab, A, Cb]
            }
        },

        "E": {
            "natural": {
                "maj": [E, Fs, Gs, A, B, Cs, Ds],
                "min": [E, Fs, G, A, B, C, D]
            },
            "#": {
                "maj": [Es, G, A, As, Bs, D, E],
                "min": [Es, G, Gs, As, Bs, Cs, Ds]
            },
            "b": {
                "maj": [Eb, F, G, Ab, Bb, C, D],
                "min": [Eb, F, Gb, Ab, Bb, Cb, Db]
            }
        },

        "F": {
            "natural": {
                "maj": [F, G, A, Bb, C, D, E],
                "min": [F, G, Ab, Bb, C, Db, Eb]
            },
            "#": {
                "maj": [Fs, Gs, As, B, Cs, Ds, Es],
                "min": [Fs, Gs, A, B, Cs, D, E]
            },
            "b": {
                "maj": [Fb, Gb, Ab, A, Cb, Db, Eb],
                "min": [Fb, Gb, G, A, Cb, C, D]
            }
        },

        "G": {
            "natural": {
                "maj": [G, A, B, C, D, E, Fs],
                "min": [G, A, Bb, C, D, Eb, F]
            },
            "#": {
                "maj": [Gs, As, Bs, Cs, Ds, Es, G],
                "min": [Gs, As, B, Cs, Ds, E, Fs]
            },
            "b": {
                "maj": [Gb, Ab, Bb, Cb, Db, Eb, F],
                "min": [Gb, Ab, A, Cb, Db, D, Fb]
            }
        },

        "A": {
            "natural": {
                "maj": [A, B, Cs, D, E, Fs, Gs],
                "min": [A, B, C, D, E, F, G]
            },
            "#": {
                "maj": [As, Bs, D, Ds, Es, G, A],
                "min": [As, Bs, Cs, Ds, Es, Fs, Gs]
            },
            "b": {
                "maj": [Ab, Bb, C, Db, Eb, F, G],
                "min": [Ab, Bb, Cb, Db, Eb, Fb, Gb]
            }
        },

        "B": {
            "natural": {
                "maj": [B, Cs, Ds, E, Fs, Gs, As],
                "min": [B, Cs, D, E, Fs, G, A]
            },
            "#": {
                "maj": [Bs, D, E, Es, G, A, B],
                "min": [Bs, D, Ds, Es, G, Gs, As]
            },
            "b": {
                "maj": [Bb, C, D, Eb, F, G, A],
                "min": [Bb, C, Db, Eb, F, Gb, Ab]
            }
        }
    }

    alpha_key = {  # dictionary of alphanumeric scales
        "1": {
            "A": key["A"]["b"]["min"],
            "B": key["B"]["natural"]["maj"]
        },

        "2": {
            "A": key["E"]["b"]["min"],
            "B": key["F"]["#"]["maj"]
        },

        "3": {
            "A": key["B"]["b"]["min"],
            "B": key["D"]["b"]["maj"]
        },

        "4": {
            "A": key["F"]["natural"]["min"],
            "B": key["A"]["b"]["maj"]
        },

        "5": {
            "A": key["C"]["natural"]["min"],
            "B": key["E"]["b"]["maj"]
        },

        "6": {
            "A": key["G"]["natural"]["min"],
            "B": key["B"]["b"]["maj"]
        },

        "7": {
            "A": key["D"]["natural"]["min"],
            "B": key["F"]["natural"]["maj"]
        },

        "8": {
            "A": key["A"]["natural"]["min"],
            "B": key["C"]["natural"]["maj"]
        },

        "9": {
            "A": key["E"]["natural"]["min"],
            "B": key["G"]["natural"]["maj"]
        },

        "10": {
            "A": key["B"]["natural"]["min"],
            "B": key["D"]["natural"]["maj"]
        },

        "11": {
            "A": key["F"]["#"]["min"],
            "B": key["A"]["natural"]["maj"]
        },

        "12": {
            "A": key["D"]["b"]["min"],
            "B": key["E"]["natural"]["maj"]
        },
    }

    key_names = [ # storing scales into an array along with a name for an easy way to show them to the user
        Key(
            "C",
            [
                Key("C major", key["C"]["natural"]["maj"]),
                Key("C minor", key["C"]["natural"]["min"]),
                Key("C# major", key["C"]["#"]["maj"]),
                Key("C# minor", key["C"]["#"]["min"]),
                Key("Cb major", key["C"]["b"]["maj"]),
                Key("Cb minor", key["C"]["b"]["min"])
            ]
        ),
        Key(
            "D",
            [
                Key("D major", key["D"]["natural"]["maj"]),
                Key("D minor", key["D"]["natural"]["min"]),
                Key("D# major", key["D"]["#"]["maj"]),
                Key("D# minor", key["D"]["#"]["min"]),
                Key("Db major", key["D"]["b"]["maj"]),
                Key("Db minor", key["D"]["b"]["min"])
            ]
        ),
        Key(
            "E",
            [
                Key("E major", key["E"]["natural"]["maj"]),
                Key("E minor", key["E"]["natural"]["min"]),
                Key("E# major", key["E"]["#"]["maj"]),
                Key("E# minor", key["E"]["#"]["min"]),
                Key("Eb major", key["E"]["b"]["maj"]),
                Key("Eb minor", key["E"]["b"]["min"])
            ]
        ),
        Key(
            "F",
            [
                Key("F major", key["F"]["natural"]["maj"]),
                Key("F minor", key["F"]["natural"]["min"]),
                Key("F# major", key["F"]["#"]["maj"]),
                Key("F# minor", key["F"]["#"]["min"]),
                Key("Fb major", key["F"]["b"]["maj"]),
                Key("Fb minor", key["F"]["b"]["min"])
            ]
        ),
        Key(
            "G",
            [
                Key("G major", key["G"]["natural"]["maj"]),
                Key("G minor", key["G"]["natural"]["min"]),
                Key("G# major", key["G"]["#"]["maj"]),
                Key("G# minor", key["G"]["#"]["min"]),
                Key("Gb major", key["G"]["b"]["maj"]),
                Key("Gb minor", key["G"]["b"]["min"])
            ]
        ),
        Key(
            "A",
            [
                Key("A major", key["A"]["natural"]["maj"]),
                Key("A minor", key["A"]["natural"]["min"]),
                Key("A# major", key["A"]["#"]["maj"]),
                Key("A# minor", key["A"]["#"]["min"]),
                Key("Ab major", key["A"]["b"]["maj"]),
                Key("Ab minor", key["A"]["b"]["min"])
            ]
        ),
        Key(
            "B",
            [
                Key("B major", key["B"]["natural"]["maj"]),
                Key("B minor", key["B"]["natural"]["min"]),
                Key("B# major", key["B"]["#"]["maj"]),
                Key("B# minor", key["B"]["#"]["min"]),
                Key("Bb major", key["B"]["b"]["maj"]),
                Key("Bb minor", key["B"]["b"]["min"])
            ]
        )
    ]


long_key_to_target = {
    "C major": maj_key_encoding["C"],
    "C minor": min_key_encoding["Cm"],
    "C# major": maj_key_encoding["Db"],
    "C# minor": min_key_encoding["C#m"],
    "Cb major": maj_key_encoding["B"],
    "Cb minor": min_key_encoding["Bm"],
    "D major": maj_key_encoding["D"],
    "D minor": min_key_encoding["Dm"],
    "D# major": maj_key_encoding["Eb"],
    "D# minor": min_key_encoding["Ebm"],
    "Db major": maj_key_encoding["Db"],
    "Db minor": min_key_encoding["C#m"],
    "E major": maj_key_encoding["E"],
    "E minor": min_key_encoding["Em"],
    "E# major": maj_key_encoding["F"],
    "E# minor": min_key_encoding["Fm"],
    "Eb major": maj_key_encoding["Eb"],
    "Eb minor": min_key_encoding["Ebm"],
    "F major": maj_key_encoding["F"],
    "F minor": min_key_encoding["Fm"],
    "F# major": maj_key_encoding["F#"],
    "F# minor": min_key_encoding["F#m"],
    "Fb major": maj_key_encoding["E"],
    "Fb minor": min_key_encoding["Em"],
    "G major": maj_key_encoding["G"],
    "G minor": min_key_encoding["Gm"],
    "G# major": maj_key_encoding["Ab"],
    "G# minor": min_key_encoding["Abm"],
    "Gb major": maj_key_encoding["F#"],
    "Gb minor": min_key_encoding["F#m"],
    "A major": maj_key_encoding["A"],
    "A minor": min_key_encoding["Am"],
    "A# major": maj_key_encoding["Bb"],
    "A# minor": min_key_encoding["Bbm"],
    "Ab major": maj_key_encoding["Ab"],
    "Ab minor": min_key_encoding["Abm"],
    "B major": maj_key_encoding["B"],
    "B minor": min_key_encoding["Bm"],
    "B# major": maj_key_encoding["C"],
    "B# minor": min_key_encoding["Cm"],
    "Bb major": maj_key_encoding["Bb"],
    "Bb minor": min_key_encoding["Bbm"],
}


key_name_to_key_values = {
    "Abm": MidiValues.key['A']['b']['min'],
    "B": MidiValues.key['B']['natural']['maj'],
    "Ebm": MidiValues.key['E']['b']['min'],
    "F#": MidiValues.key['F']['#']['maj'],
    "Bbm": MidiValues.key['B']['b']['min'],
    "Db": MidiValues.key['D']['b']['maj'],
    "Fm": MidiValues.key['F']['natural']['min'],
    "Ab": MidiValues.key['A']['b']['maj'],
    "Cm": MidiValues.key['C']['natural']['min'],
    "Eb": MidiValues.key['E']['b']['maj'],
    "Gm": MidiValues.key['G']['natural']['min'],
    "Bb": MidiValues.key['B']['b']['maj'],
    "Dm": MidiValues.key['D']['natural']['min'],
    "F": MidiValues.key['F']['natural']['maj'],
    "Am": MidiValues.key['A']['natural']['min'],
    "C": MidiValues.key['C']['natural']['maj'],
    "Em": MidiValues.key['E']['natural']['min'],
    "G": MidiValues.key['G']['natural']['maj'],
    "Bm": MidiValues.key['B']['natural']['min'],
    "D": MidiValues.key['D']['natural']['maj'],
    "F#m": MidiValues.key['F']['#']['min'],
    "A": MidiValues.key['A']['natural']['maj'],
    "C#m": MidiValues.key['C']['#']['min'],
    "E": MidiValues.key['E']['natural']['maj'],
}


def find_in_dict(dictionary, value):
    for key in dictionary:
        if dictionary[key] == value:
            return key
    return None

alpha_key = {
    "1A": 12,
    "1B": 12,
    "2A": 7,
    "2B": 7,
    "3A": 2,
    "3B": 2,
    "4A": 9,
    "4B": 9,
    "5A": 4,
    "5B": 4,
    "6A": 11,
    "6B": 11,
    "7A": 6,
    "7B": 6,
    "8A": 1,
    "8B": 1,
    "9A": 8,
    "9B": 8,
    "10A": 3,
    "10B": 3,
    "11A": 10,
    "11B": 10,
    "12A": 5,
    "12B": 5,
}


def find_in_dict(dictionary, value):
    for key in dictionary:
        if dictionary[key] == value:
            return key
    return None

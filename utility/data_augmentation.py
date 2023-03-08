from config import *

from classes.midi import *
from utility.music import *
from utility.user_input import alphanumeric_split
from algorithm.genetic import continuous

DEFAULT_NOTE_DURATION = 0.25
DEFAULT_N_VARIATIONS = 15
DEFAULT_N_NOISE_MELODIES = 10
DEFAULT_VARIATION_PROBABILITY = 0.5


class MidiData(NamedTuple):
    artist: str
    song: str
    key: str
    bpm: int


def extract_midi_data(path: str) -> MidiData:
    mid = path[path.rfind('/')+1:path.rfind('.')]

    mid_info = mid[:mid.find('-')]
    mid_meta = mid[mid.find('-') + 1:]

    artist = mid_info[:mid_info.find('_')]
    song = mid_info[mid_info.find('_') + 1:]

    key = mid_meta[:mid_meta.find('_')]
    bpm = mid_meta[mid_meta.find('_') + 1:]

    return MidiData(artist, song, key, int(bpm))


class AugmentedMidi(MIDI):

    def normalize(self, max_note_duration: float = DEFAULT_NOTE_DURATION):

        if not self.single_notes:
            raise Exception("Currently there's no support for multi-note normalization.")

        aux_mid = type(self)(self.name, self.n_tracks, self.single_notes, self.playable)
        aux_mid.add_tempo(self.bpm)
        aux_mid.add_key(self.key)

        for t, track in enumerate(self.melody):

            note = track["notes"]
            duration = track["durations"]

            n_notes = round(sum(track["durations"]) / max_note_duration)

            i = 0
            while len(note) < n_notes:

                if duration[i] != max_note_duration:

                    note_dup = note[i]

                    duplicates = round(duration[i] / max_note_duration)

                    note.pop(i)
                    duration.pop(i)

                    for j in range(duplicates):
                        note.insert(i + j, note_dup)
                        duration.insert(i + j, max_note_duration)

                i = i + 1

            for n, dur in zip(note, duration):
                aux_mid.add_note(n, dur, track=t)

        return aux_mid

    def smooth(self):

        if not self.single_notes:
            raise Exception("Currently there's no support for multi-note smoothing.")

        aux_mid = type(self)(self.name, self.n_tracks, self.single_notes, self.playable)
        aux_mid.add_tempo(self.bpm)
        aux_mid.add_key(self.key)

        for t, track in enumerate(self.melody):

            note = track["notes"]
            duration = track["durations"]
            no_notes = len(note)

            i = 0
            while i < no_notes:

                j = 1
                note_duration = duration[i]
                if i != no_notes - 1:

                    while note[i] == note[i + j]:
                        note_duration += duration[i + j]
                        j += 1

                        if i + j == no_notes:
                            break

                aux_mid.add_note(note[i], note_duration, track=t)
                i += j

        return aux_mid

    def add_padding(self, n: int, padding: int = -1, padding_size: float = DEFAULT_NOTE_DURATION, track: int = 0):
        aux_mid = self

        for _ in range(n):
            aux_mid.add_note(padding, padding_size, track=track)

        return aux_mid

    def variation(self, scale: [], no_variations: int = DEFAULT_N_VARIATIONS, variation_probability: float = DEFAULT_VARIATION_PROBABILITY,
                  note_probability: float = DEFAULT_NOTE_PROBABILITY, midi_range: int = DEFAULT_BITS_PER_NOTE):

        from random import randrange
        from statistics import median

        aux_mid = type(self)(self.name, self.n_tracks, self.single_notes, self.playable)
        aux_mid.add_tempo(self.bpm)
        aux_mid.add_key(self.key)

        scale = continuous(scale, return_sorted=True)  # gets rid of the separations between notes

        for t, track in enumerate(self.melody):
            note = track["notes"]
            duration = track["durations"]

            root = round(median(note))
            root_pos = scale.index(root)  # finds the root in this new array

            for _ in range(no_variations):
                index = randrange(len(note))
                rand_note = randrange(scale[root_pos - midi_range], scale[root_pos + midi_range])

                while random() <= variation_probability and index <= len(note) - 1:
                    note[index] = rand_note if random() <= note_probability else -1
                    index = index + 1

            for n, dur in zip(note, duration):
                aux_mid.add_note(n, dur, track=t)

        return aux_mid

    def transpose(self, semitones: int):
        aux_mid = type(self)(self.name, self.n_tracks, self.single_notes, self.playable)
        aux_mid.add_tempo(self.bpm)

        for t, track in enumerate(self.melody):

            notes = track["notes"]
            durations = track["durations"]

            for note, duration in zip(notes, durations):

                if note != -1:
                    aux_mid.add_note(note + semitones, duration, track=t)
                else:
                    aux_mid.add_note(note, duration, track=t)

        return aux_mid

    def flip(self):
        aux_mid = type(self)(self.name, self.n_tracks, self.single_notes, self.playable)
        aux_mid.add_tempo(self.bpm)
        aux_mid.add_key(self.key)

        for t, track in enumerate(self.melody):

            notes = track["notes"]
            durations = track["durations"]

            notes.reverse()
            durations.reverse()

            for note, duration in zip(notes, durations):
                aux_mid.add_note(note, duration, track=t)

        return aux_mid


def write_transpositions(path: str):
    midi_data = extract_midi_data(path)
    k = alpha_maj_keys[midi_data.key]  # Original key
    original_key = k

    is_descending = False

    for i in range(NO_MAJ_KEYS):  # Ignoring minor relatives
        mid = AugmentedMidi(single_notes=True, playable=False)
        mid.read_midi(path, ignore_meta=True)
        mid.add_tempo(midi_data.bpm)

        if i <= NO_MAJ_KEYS / 2:  # Transposing the melody up to +6 semitones up
            mid = mid.transpose(+i)
            if i != 0:
                k += 1

        else:  # Transposing the melody down to -5 semitones down
            if not is_descending:
                k = original_key
                is_descending = True

            mid = mid.transpose(-(i % int(NO_MAJ_KEYS / 2)))
            k -= 1

        if k > NO_MAJ_KEYS:
            k = 1
        elif k < 1:
            k = 12

        key_root, key_type = alphanumeric_split(find_in_dict(alpha_maj_keys, k))
        key_signature = find_in_dict(short_maj_keys, k)

        relative_mid = copy.deepcopy(mid)

        mid.add_key(key_signature)
        mid.write_midi(f"{midi_data.artist}_{midi_data.song}-{key_root}B_{midi_data.bpm}",
                       path=f"../classes/data/augmented_data/transpositions/B/{key_root}")

        relative_mid.add_key(relative_min_key[key_signature])
        relative_mid.write_midi(f"{midi_data.artist}_{midi_data.song}-{key_root}A_{midi_data.bpm}",
                       path=f"../classes/data/augmented_data/transpositions/A/{key_root}")


def write_variations(path: str, n: int = DEFAULT_N_NOISE_MELODIES):
    midi_data = extract_midi_data(path)
    key_root, key_type = alphanumeric_split(midi_data.key)

    for i in range(n):

        m = AugmentedMidi(single_notes=True, playable=False)
        m.read_midi(path, ignore_first_track=True)
        m.add_tempo(midi_data.bpm)
        m.add_key(alpha_to_short_key[midi_data.key])

        m.add_name(f"{midi_data.artist}_{midi_data.song}-{key_root}{key_type}_{midi_data.bpm}-{i}")
        m = m.normalize()

        while m.total_duration < 16:
            m = m.add_padding(n=1, padding_size=DEFAULT_NOTE_DURATION)

        # print(f"writing variation n.{n} of: {mid}")
        m = m.variation(MidiValues.alpha_key[key_root][key_type])
        m = m.smooth()

        m.write_midi(path="../classes/data/augmented_data/noisy_data")


def main():
    trans = True
    variations = True
    flip = True

    directory = "../classes/data/original_data"
    n_originals = len(os.listdir(directory))

    if trans:
        directory = "../classes/data/original_data"

        for f in tqdm(os.listdir(directory), desc="transposing", unit="melodies"):
            if f.endswith(".mid"):
                path = os.path.join(directory, f)
                write_transpositions(path)

    if variations:
        directory = "../classes/data/augmented_data/transpositions"

        for root, dirs, files in tqdm(os.walk(directory), total=DEFAULT_N_VARIATIONS*NO_KEYS*n_originals, desc="adding noise", unit="directories"):
            for file in files:
                if file.endswith(".mid"):

                    path = os.path.join(root, file)
                    write_variations(path)

    if flip:
        directory = "../classes/data/augmented_data/noisy_data"

        for f in tqdm(os.listdir(directory), desc="flipping", unit="melodies"):
            if f.endswith(".mid"):
                path = os.path.join(directory, f)
                i = path[path.rfind('-')+1:path.rfind('.')]

                mid = path[:path.rfind('-')] + ".mid"
                midi_data = extract_midi_data(mid)

                key_root, key_type = alphanumeric_split(midi_data.key)

                m = AugmentedMidi(single_notes=True, playable=False)
                m.read_midi(path, ignore_first_track=True)
                m.add_tempo(midi_data.bpm)
                m.add_key(alpha_to_short_key[midi_data.key])

                m.add_name(f"{midi_data.artist}_{midi_data.song}-{key_root}{key_type}_{midi_data.bpm}-{i}_flipped")

                m_flipped = m.flip()
                m_flipped.write_midi(path="../classes/data/augmented_data/flipped_data")

        directory = "../classes/data/augmented_data/transpositions"

        for root, dirs, files in tqdm(os.walk(directory), total=NO_KEYS*n_originals, desc="flipping", unit="directories"):
            for file in files:
                if file.endswith(".mid"):
                    path = os.path.join(root, file)
                    midi_data = extract_midi_data(path)

                    key_root, key_type = alphanumeric_split(midi_data.key)

                    m = AugmentedMidi(single_notes=True, playable=False)
                    m.read_midi(path, ignore_first_track=True)
                    m.add_tempo(midi_data.bpm)
                    m.add_key(alpha_to_short_key[midi_data.key])

                    m.add_name(f"{midi_data.artist}_{midi_data.song}-{key_root}{key_type}_{midi_data.bpm}_flipped")

                    m_flipped = m.flip()
                    m_flipped.write_midi(path="../classes/data/augmented_data/flipped_data")


if __name__ == '__main__':
    main()

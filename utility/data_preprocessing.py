from config import *
from classes.midi import MIDI
from utility.data_augmentation import AugmentedMidi, DEFAULT_NOTE_DURATION
from utility.music import *


class MidiDataset(torch_data.Dataset):
    def __init__(self, data_dir, are_tensors: bool = True):
        self.data_dir = data_dir
        self.are_tensors = are_tensors
        self.classes = sorted(os.listdir(data_dir))
        self.midi = []
        for i, cls in enumerate(self.classes):
            if cls == ".DS_Store":
                continue
            cls_dir = os.path.join(data_dir, cls)

            for f in tqdm(os.listdir(cls_dir), desc="loading midis", unit="midi"):
                if f.endswith(".mid"):
                    midi_path = os.path.join(cls_dir, f)

                    m = MIDI(single_notes=True, playable=False)
                    m.read_midi(midi_path)

                    notes = [note for note in m.melody[1]["notes"]]
                    note_lengths = [note_len for note_len in m.melody[1]["durations"]]

                    if self.are_tensors:
                        notes = torch.tensor(notes, dtype=torch.float32)
                        note_lengths = torch.tensor(note_lengths, dtype=torch.float32)

                    self.midi.append((notes, note_lengths, m.key, m.bpm))

    def __getitem__(self, index):
        return self.midi[index]

    def __setitem__(self, key, value):
        self.midi[key] = value

    def __len__(self):
        return len(self.midi)


def preprocess_midi(midi_data: MidiDataset, between_zero_one: bool = True, encoded_labels: bool = True):

    i = 0
    bias = DEFAULT_INDUCTIVE_BIAS_SIZE
    for notes, note_lengths, key, bpm in tqdm(midi_data, desc="preprocessing midis", unit="midi"):

        m = AugmentedMidi(single_notes=True, playable=False)
        m.add_key(key)
        m.add_tempo(bpm)

        if key.find('m') == -1:
            encoded_maj_key = maj_key_encoding[key]
            # encoded_min_key = min_key_encoding[relative_min_key[m.key]]
        else:
            # encoded_min_key = min_key_encoding[m.key]
            encoded_maj_key = maj_key_encoding[relative_maj_key[key]]

        for note, note_length in zip(notes, note_lengths):

            if midi_data.are_tensors:
                m.add_note(int(note.numpy()), note_length.numpy())
            else:
                m.add_note(int(note), note_length)

        m.normalize()
        while sum(m.melody[0]["durations"]) < 16: # Duration of 4 bars
            m.add_padding(n=1)

        for _ in range(bias):
            m.add_note(int(key_encoding_midi[encoded_maj_key]), DEFAULT_NOTE_DURATION)  # Sesgo

        if between_zero_one:
            normalized_notes = [raw_note / N_MIDI_VALUES for raw_note in m.melody[0]["notes"]]
        else:
            normalized_notes = [raw_note for raw_note in m.melody[0]["notes"]]

        if midi_data.are_tensors:
            normalized_notes = torch.tensor(normalized_notes, dtype=torch.float32)

        midi_data[i] = (normalized_notes, encoded_maj_key if encoded_labels else key)
        i += 1

    return midi_data


def data_report(data: torch_data.Dataset, columns: [], title: str = "data"):
    from dataprep.eda import create_report
    from scipy.stats import entropy

    print("creating report...")
    df = pd.DataFrame(columns=columns)

    for i, sample_point in enumerate(data):
        df.loc[i] = [*sample_point]

    melodies = df['notes'].to_numpy()

    h = 0.0
    for melody in melodies:
        h_i = entropy(melody)
        h += h_i

    h /= len(melodies)

    # print(f"entropy: {h}")

    report = create_report(df, title=title)
    report.save("../classes/data/preprocessed_data/" + title)
    df.to_csv("../classes/data/preprocessed_data/" + title + ".csv")


def main():
    dataset = MidiDataset("../classes/data/augmented_data", are_tensors=False)
    dataset = preprocess_midi(dataset, between_zero_one=True, encoded_labels=True)
    data_report(dataset, columns=['notes', 'key'], title="preprocessed_data")


if __name__ == '__main__':
    main()

from config import *


class MIDI:
    """

    MIDI merges MIDIFile and pyo into a single class for ease of use, and the ability to indirectly play the MIDIFile

    :var single_notes: if set to ''True'', notes will have no overlap (does not affect chords)
    :var MIDI.bpm: beats per minute (how fast the midi file will play)
    :var MIDI.melody:
        notes: stores the sequence of midi notes into an array i.e [60, 61] = [C4, C#4]
        durations: stores the length of the note in beats
                  i.e [..., 0.5, ...], ''0.5'' is the length of the note at index i (where the ''0.5'' is)
        beat: stores the beat at which the note would be played
              i.e [..., 1, ...], ''1'' is the beat where the note at index i (where the ''1' is) would be played

              Normally the values in beat are auto calculated given the length of the note if ''single_notes'' is
              set to ''True'', but can be manually set too (note that this can cause overlap between notes)

    """

    def __init__(self, name: str = "", tracks: int = 1, bpm: int = 120, single_notes: bool = False, playable: bool = True):
        self.m = MIDIFile(tracks)
        self.single_notes = single_notes
        self.playable = playable

        self.melody = [{"notes": [], "durations": [], "beat": []} for _ in range(tracks)]

        for t in range(tracks):
            self.m.addTempo(t, 0, bpm)  # Adds the message at time = 0
        self.bpm = bpm

        self.total_duration = 0
        self.n_tracks = tracks

        self.name = name if name != "" else "untitled"

        if self.playable:
            self.s = Server().boot()

    def __del__(self):
        self.m.close()

        if self.playable:
            self.s.shutdown()

    def close(self):
        self.__del__()

    def add_name(self, name: str):
        self.name = name

    def add_track_name(self, name: str, track: int = 0):
        self.m.addTrackName(track, 0, name) # Adds the message at time = 0

    def add_tempo(self, bpm: int):
        for t in range(self.n_tracks):
            self.m.addTempo(t, 0, bpm)  # Adds the message at time = 0
        self.bpm = bpm

    def add_track(self):
        new_m = MIDIFile(self.n_tracks + 1)
        self.melody += [{"notes": [], "durations": [], "beat": []}]

        for t in range(self.n_tracks):
            for event in self.m.tracks[t].MIDIEventList:
                new_m.tracks[t].MIDIEventList.append(event)

        self.m = new_m
        self.n_tracks += 1

    def add_note(self,
                 note: int, duration: float = 1,
                 on_beat: float = 0, track: int = 0, channel: int = 0, volume: int = 100
                 ):
        """

        adds a note to the MIDIFile object and stores its values into ''self.melody''

        :param note: midi note i.e 60 = middle C
        :param duration: length of the note in beats
        :param on_beat: the beat where the note would be played
        :param track: the track where the note will be added
        :param channel: the channel where the note will be added
        :param volume: volume of the given note

        """
        self.melody[track]["beat"] += [sum(self.melody[track]["durations"])] if self.single_notes else [on_beat]
        # the beat is at where the previous note/chord ended

        self.melody[track]["durations"] += [duration]
        self.total_duration += duration

        if note != MidiValues.rest:
            self.melody[track]["notes"] += [note]
            self.m.addNote(track, channel, note, self.melody[track]["beat"][-1], duration, volume)
        else:
            self.melody[track]["notes"] += [MidiValues.rest]

    def add_chord(self,
                  root: int, chord_type: str, duration: float = 1,
                  on_beat: float = 0, track: int = 0, channel: int = 0, volume: int = 100
                  ):
        """

        adds a chord to the MIDIFile object and stores its values into ''self.melody''

        :param root: root of the chord in midi value
        :param chord_type: chord type i.e "maj", "min". defined in ''chord_from_root''
        :param duration: length of the chord in beats
        :param on_beat: the beat where the chord would be played
        :param track: the track where the chord will be added
        :param channel: the channel where the chord will be added
        :param volume: volume of the given chord

        """

        self.melody[track]["beat"] += [sum(self.melody[track]["durations"])] if self.single_notes else [on_beat]
        # the beat is at where the previous note/chord ended

        chord = MIDI.chord_from_root(root, chord_type)

        self.melody[track]["notes"] += [chord]
        self.melody[track]["durations"] += [duration]
        self.total_duration += duration

        for note in chord:
            self.m.addNote(track, channel, note, self.melody[track]["beat"][-1], duration, volume)

    def read_midi(self, path: str, ignore_first_track: bool = False):
        """

        reads a midi file and extracts it's values to fit the class and stores them inside the MIDI object

        :param path: the path where the midi file is located
        :param ignore_first_track: if the midi has a first track with just meta messages, this can be set to True
                                    and ignore those messages

        Note: this only works when 'single_notes' is set to True

        """

        if not self.single_notes:
            raise Exception("Currently there's no support for multi-note reading.")

        m = MidiFile(path)

        if ignore_first_track:
            del m.tracks[0]

        # time_unit represents how many ticks equal a '1' in duration
        time_unit = m.ticks_per_beat

        for _ in range(len(m.tracks) - 1):
            self.add_track()

        for i, t in enumerate(m.tracks):
            duration_in_ticks = 0
            is_playing = False

            for msg in t:

                if msg.type == "key_signature":
                    print(msg)

                if msg.type == "time_signature":
                    print(msg)

                if msg.type == "set_tempo" and i == 0:
                    self.add_tempo(tempo2bpm(msg.tempo))

                # This way we can avoid all the meta messages at the beginning
                if is_playing:

                    # in case any other messages get in the way while the track is sending notes, i.e tempo/key changes
                    duration_in_ticks += msg.time

                if msg.type == "note_on":
                    is_playing = True

                    # this means there was a rest between this note and the last note_off event
                    if duration_in_ticks != 0:

                        self.add_note(-1, duration_in_ticks / time_unit, track=i)
                        duration_in_ticks = 0

                elif msg.type == "note_off":

                    self.add_note(msg.note, duration_in_ticks / time_unit, track=i)
                    duration_in_ticks = 0

    def write_midi(self, name: str = "", path: str = "", mode: str = "wb"):
        """

        writes a midi file with the MIDIFile object values

        :param name: name of the file (must include extension .mid)
        :param path: the path to output the midi
        :param mode: how the file is opened

        """
        if name == "":
            name = self.name

        name += ".mid"
        path += "/" if path != "" else ""

        with open(path + name, mode) as out:
            self.m.writeFile(out)

    def play(self):
        if not self.playable:
            raise Exception("Midi object is not playable!")

        self.write_midi("play", path="classes/temp")

        self.s.start()

        mid = Notein()
        amp = MidiAdsr(mid["velocity"])
        pit = MToF(mid["pitch"])
        osc = Osc(SquareTable(), freq=pit, mul=amp).mix(1)
        rev = STRev(osc, revtime=1, cutoff=4000, bal=0.2).out()

        mid = MidiFile("classes/temp/play.mid")

        time.sleep(1.5)
        for message in mid.play():
            self.s.addMidiEvent(*message.bytes())

        self.s.stop()

    # @staticmethod
    # def metronome(bpm: int):
    #     """
    #
    #     plays a metronome
    #
    #     :param bpm: beats per minute (how fast the metronome plays)
    #
    #     """
    #
    #     met = Metro(time=1 / (bpm / 60.0)).play()
    #     t = CosTable([(0, 0), (50, 1), (200, .3), (500, 0)])
    #     amp = TrigEnv(met, table=t, dur=.25, mul=1)
    #     freq = Iter(met, choice=[660, 440, 440, 440])
    #
    #     # time.sleep(1) # gives time to load
    #
    #     return Sine(freq=freq, mul=amp).mix(2).out()

    @staticmethod
    def chord_from_root(root: int, chord_type: str = "major") -> []:
        """

        returns a 3 note chord based on the root (no inversions [yet])

        :param root: root of the chord
        :param chord_type: chord type i.e "maj", "min"

        """

        if chord_type == "major" or chord_type == "maj":
            return [root, root + 4, root + 7]

        elif chord_type == "minor" or chord_type == "min":
            return [root, root + 3, root + 7]

        elif chord_type == "augmented" or chord_type == "aug":
            return [root, root + 4, root + 8]

        elif chord_type == "diminished" or chord_type == "dim":
            return [root, root + 3, root + 6]

    @staticmethod
    def beats_to_secs(no_beats: float, bpm: int) -> float:
        """

        returns the the length of the beats in seconds

        :param no_beats: number of beats
        :param bpm: the bpm at which the beats are in

        """

        return (60/bpm) * no_beats


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
    rest = -1

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
            "8",
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

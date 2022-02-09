from midiutil import MIDIFile
from mido import MidiFile
from pyo import *

from collections import namedtuple
import time


class MIDI:
    """

    MIDI merges MIDIFile and pyo into a single class for ease of use, and the ability to indirectly play the MIDIFile

    :var single_notes: if set to ''True'', notes will have no overlap (does not affect chords)
    :var MIDI.bpm: beats per minute (how fast the midi file will play)
    :var MIDI.melody:
        notes: stores the sequence of midi notes into an array i.e [60, 61] = [C4, C#4]
        duration: stores the length of the note in beats
                  i.e [..., 0.5, ...], ''0.5'' is the length of the note at index i (where the ''0.5'' is)
        beat: stores the beat at which the note would be played
              i.e [..., 1, ...], ''1'' is the beat where the note at index i (where the ''1' is) would be played

              Normally the values in beat are auto calculated given the length of the note if ''single_notes'' is
              set to ''True'', but can be manually set too (note that this can cause overlap between notes)

    """

    single_notes: bool
    bpm: int

    def __init__(self, tracks: int = 1, single_notes: bool = False):
        self.m = MIDIFile(tracks)
        self.single_notes = single_notes

        self.melody = [{"notes": [], "duration": [], "beat": []} for _ in range(tracks)]

        self.s = Server().boot()

    def __del__(self):
        self.m.close()
        self.s.shutdown()

    def add_name(self, name: str, track: int = 0):
        self.m.addTrackName(track, 0, name)

    def add_tempo(self, bpm: int, track: int = 0):
        self.bpm = bpm
        self.m.addTempo(track, 0, bpm)

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

        self.melody[track]["beat"] += [sum(self.melody[track]["duration"])] if self.single_notes else [on_beat]
        # the beat is at where the previous note/chord ended

        self.melody[track]["duration"] += [duration]

        if note != rest:
            self.melody[track]["notes"] += [note]
            self.m.addNote(track, channel, note, self.melody[track]["beat"][-1], duration, volume)
        else:
            self.melody[track]["notes"] += [rest]

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

        self.melody[track]["beat"] += [sum(self.melody[track]["duration"])] if self.single_notes else [on_beat]
        # the beat is at where the previous note/chord ended

        chord = MIDI.chord_from_root(root, chord_type)

        self.melody[track]["notes"] += [chord]
        self.melody[track]["duration"] += [duration]

        for note in chord:
            self.m.addNote(track, channel, note, self.melody[track]["beat"][-1], duration, volume)

    def write_midi(self, name: str = "output", path: str = "", mode: str = "wb"):
        """

        writes a midi file with the MIDIFile object values

        :param name: name of the file (must include extension .mid)
        :param path: the path to output the midi
        :param mode: how the file is opened

        """

        name += ".mid"
        path += "/" if path != "" else ""

        with open(path + name, mode) as out:
            self.m.writeFile(out)

    def play(self):
        self.write_midi("play", path="classes/temp")

        self.s.start()

        mid = Notein()
        amp = MidiAdsr(mid["velocity"])
        pit = MToF(mid["pitch"])
        osc = Osc(SquareTable(), freq=pit, mul=amp).mix(1)
        rev = STRev(osc, revtime=1, cutoff=4000, bal=0.2).out()

        mid = MidiFile("classes/temp/play.mid")

        for message in mid.play():
            self.s.addMidiEvent(*message.bytes())

        self.s.stop()

    @staticmethod
    def metronome(bpm: int):
        """

        plays a metronome

        :param bpm: beats per minute (how fast the metronome plays)

        """

        met = Metro(time=1 / (bpm / 60.0)).play()
        t = CosTable([(0, 0), (50, 1), (200, .3), (500, 0)])
        amp = TrigEnv(met, table=t, dur=.25, mul=1)
        freq = Iter(met, choice=[660, 440, 440, 440])

        # time.sleep(1) # gives time to load

        return Sine(freq=freq, mul=amp).mix(2).out()

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

"""

a collection of midi values to be easily referenced based on the note/scale they represent
i.e c[4] = 60 = C4

"""

# Natural

c = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
d = [14, 26, 38, 50, 62, 74, 86, 98, 110, 122]
e = [16, 28, 40, 52, 64, 76, 88, 100, 112, 124]
f = [17, 29, 41, 53, 65, 77, 89, 101, 113, 125]
g = [19, 31, 43, 55, 67, 79, 91, 103, 115, 127]
a = [21, 33, 45, 57, 69, 81, 93, 105, 117]
b = [23, 35, 47, 59, 71, 83, 95, 107, 119]

# Sharp

c_s = [13, 25, 37, 49, 61, 73, 85, 97, 109, 121]
d_s = [15, 27, 39, 51, 63, 75, 87, 99, 111, 123]
e_s = f
f_s = [18, 30, 42, 54, 66, 78, 90, 102, 114, 126]
g_s = [20, 32, 44, 56, 68, 80, 92, 104, 116]
a_s = [22, 34, 46, 58, 70, 82, 94, 106, 118]
b_s = [24, 36, 48, 60, 72, 84, 96, 108, 120]

# Flat

c_b = [11, 23, 35, 47, 59, 71, 83, 95, 107, 119]
d_b = c_s
e_b = d_s
f_b = e
g_b = f_s
a_b = g_s
b_b = a_s

# Rest

rest = -1

# Scales

Scale = namedtuple("Scale", ["name", "values"])

scales = { # dictionary of scales
    "C": {
        "natural": {
            "maj": [c, d, e, f, g, a, b],
            "min": [c, d, e_b, f, g, a_b, b_b]
        },
        "sharp": {
            "maj": [c_s, d_s, e_s, f_s, g_s, a_s, b_s],
            "min": [c_s, d_s, e, f_s, g_s, a, b]
        },
        "flat": {
            "maj": [c_b, d_b, e_b, f_b, g_b, a_b, b_b],
            "min": [c_b, d_b, d, f_b, g_b, g, a]
        }
    },

    "D": {
        "natural": {
            "maj": [d, e, f_s, g, a, b, c_s],
            "min": [d, e, f, g, a, b_b, c]
        },
        "sharp": {
            "maj": [d_s, e_s, g, g_s, a_s, b_s, d],
            "min": [d_s, e_s, f_s, g_s, a_s, b, c_s]
        },
        "flat": {
            "maj": [d_b, e_b, f, g_b, a_b, b_b, c],
            "min": [d_b, e_b, f_b, g_b, a_b, a, c_b]
        }
    },

    "E": {
        "natural": {
            "maj": [e, f_s, g_s, a, b, c_s, d_s],
            "min": [e, f_s, g, a, b, c, d]
        },
        "sharp": {
            "maj": [e_s, g, a, a_s, b_s, d, e],
            "min": [e_s, g, g_s, a_s, b_s, c_s, d_s]
        },
        "flat": {
            "maj": [e_b, f, g, a_b, b_b, c, d],
            "min": [e_b, f, g_b, a_b, b_b, c_b, d_b]
        }
    },

    "F": {
        "natural": {
            "maj": [f, g, a, b_b, c, d, e],
            "min": [f, g, a_b, b_b, c, d_b, e_b]
        },
        "sharp": {
            "maj": [f_s, g_s, a_s, b, c_s, d_s, e_s],
            "min": [f_s, g_s, a, b, c_s, d, e]
        },
        "flat": {
            "maj": [f_b, g_b, a_b, a, c_b, d_b, e_b],
            "min": [f_b, g_b, g, a, c_b, c, d]
        }
    },

    "G": {
        "natural": {
            "maj": [g, a, b, c, d, e, f_s],
            "min": [g, a, b_b, c, d, e_b, f]
        },
        "sharp": {
            "maj": [g_s, a_s, b_s, c_s, d_s, e_s, g],
            "min": [g_s, a_s, b, c_s, d_s, e, f_s]
        },
        "flat": {
            "maj": [g_b, a_b, b_b, c_b, d_b, e_b, f],
            "min": [g_b, a_b, a, c_b, d_b, d, f_b]
        }
    },

    "A": {
        "natural": {
            "maj": [a, b, c_s, d, e, f_s, g_s],
            "min": [a, b, c, d, e, f, g]
        },
        "sharp": {
            "maj": [a_s, b_s, d, d_s, e_s, g, a],
            "min": [a_s, b_s, c_s, d_s, e_s, f_s, g_s]
        },
        "flat": {
            "maj": [a_b, b_b, c, d_b, e_b, f, g],
            "min": [a_b, b_b, c_b, d_b, e_b, f_b, g_b]
        }
    },

    "B": {
        "natural": {
            "maj": [b, c_s, d_s, e, f_s, g_s, a_s],
            "min": [b, c_s, d, e, f_s, g, a]
        },
        "sharp": {
            "maj": [b_s, d, e, e_s, g, a, b],
            "min": [b_s, d, d_s, e_s, g, g_s, a_s]
        },
        "flat": {
            "maj": [b_b, c, d, e_b, f, g, a],
            "min": [b_b, c, d_b, e_b, f, g_b, a_b]
        }
    }
}

Scales = [ # storing scales into an array along with a name for an easy way to show them to the user
    Scale(
        "C",
        [
            Scale("C major", scales["C"]["natural"]["maj"]),
            Scale("C minor", scales["C"]["natural"]["min"]),
            Scale("C# major", scales["C"]["sharp"]["maj"]),
            Scale("C# minor", scales["C"]["sharp"]["min"]),
            Scale("Cb major", scales["C"]["flat"]["maj"]),
            Scale("Cb minor", scales["C"]["flat"]["min"])
        ]
    ),
    Scale(
        "D",
        [
            Scale("D major", scales["D"]["natural"]["maj"]),
            Scale("D minor", scales["D"]["natural"]["min"]),
            Scale("D# major", scales["D"]["sharp"]["maj"]),
            Scale("D# minor", scales["D"]["sharp"]["min"]),
            Scale("Db major", scales["D"]["flat"]["maj"]),
            Scale("Db minor", scales["D"]["flat"]["min"])
        ]
    ),
    Scale(
        "E",
        [
            Scale("E major", scales["E"]["natural"]["maj"]),
            Scale("E minor", scales["E"]["natural"]["min"]),
            Scale("E# major", scales["E"]["sharp"]["maj"]),
            Scale("E# minor", scales["E"]["sharp"]["min"]),
            Scale("Eb major", scales["E"]["flat"]["maj"]),
            Scale("Eb minor", scales["E"]["flat"]["min"])
        ]
    ),
    Scale(
        "F",
        [
            Scale("F major", scales["F"]["natural"]["maj"]),
            Scale("F minor", scales["F"]["natural"]["min"]),
            Scale("F# major", scales["F"]["sharp"]["maj"]),
            Scale("F# minor", scales["F"]["sharp"]["min"]),
            Scale("Fb major", scales["F"]["flat"]["maj"]),
            Scale("Fb minor", scales["F"]["flat"]["min"])
        ]
    ),
    Scale(
        "G",
        [
            Scale("G major", scales["G"]["natural"]["maj"]),
            Scale("G minor", scales["G"]["natural"]["min"]),
            Scale("G# major", scales["G"]["sharp"]["maj"]),
            Scale("G# minor", scales["G"]["sharp"]["min"]),
            Scale("Gb major", scales["G"]["flat"]["maj"]),
            Scale("Gb minor", scales["G"]["flat"]["min"])
        ]
    ),
    Scale(
        "A",
        [
            Scale("A major", scales["A"]["natural"]["maj"]),
            Scale("A minor", scales["A"]["natural"]["min"]),
            Scale("A# major", scales["A"]["sharp"]["maj"]),
            Scale("A# minor", scales["A"]["sharp"]["min"]),
            Scale("Ab major", scales["A"]["flat"]["maj"]),
            Scale("Ab minor", scales["A"]["flat"]["min"])
        ]
    ),
    Scale(
        "B",
        [
            Scale("B major", scales["B"]["natural"]["maj"]),
            Scale("B minor", scales["B"]["natural"]["min"]),
            Scale("B# major", scales["B"]["sharp"]["maj"]),
            Scale("B# minor", scales["B"]["sharp"]["min"]),
            Scale("Bb major", scales["B"]["flat"]["maj"]),
            Scale("Bb minor", scales["B"]["flat"]["min"])
        ]
    )
]

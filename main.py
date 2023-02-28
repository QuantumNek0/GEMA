from config import *

from algorithm.genetic import *
from utility.user_input import *


# Global variables
population_size = int
bits_per_note = int
output_size = int

time_signature = TIME_SIGNATURE
notes_per_bar = int
unit_note = int
no_bars = int
bpm = int


def set_defaults():
    global bits_per_note, population_size, output_size
    global time_signature, unit_note, no_bars, bpm, notes_per_bar

    bits_per_note = DEFAULT_BITS_PER_NOTE  # Deviation
    population_size = DEFAULT_POPULATION_SIZE
    output_size = DEFAULT_OUTPUT_SIZE

    time_signature = DEFAULT_TIME_SIGNATURE
    unit_note = DEFAULT_UNIT_NOTE
    no_bars = DEFAULT_NO_BARS
    bpm = DEFAULT_BPM

    notes_per_bar = ((1 / time_signature.beat_note) / (1 / unit_note)) * time_signature.beats_per_bar


def main():
    global bits_per_note, population_size, output_size
    global time_signature, unit_note, no_bars, bpm, notes_per_bar

    continue_evolution = True
    clear_screen() # clears the pyo prompt

    while continue_evolution:
        valid_parameters = False

        key_root, key_type = (0, 0)
        set_defaults()

        while not valid_parameters:
            # default values

            if not confirmation("Use default values?"):

                if confirmation("Show advanced options?"):
                    bits_per_note = int(input("Bits per note: "))
                    population_size = int(input("Population size: "))
                    output_size = int(input("Output size: "))

                notes_per_bar = int(input("Notes per bar: "))
                no_bars = int(input("Number of bars: "))
                bpm = int(input("Beats per minute: "))

            clear_screen()

            if not confirmation("randomize key?"):
                key_root = select_option((s.name for s in MidiValues.key_names), "Root of scale")
                key_type = select_option((s.name for s in MidiValues.key_names[key_root].values), "Scale type")
            else:
                key_root = randrange(0, len(MidiValues.key_names))
                key_type = randrange(0, len(MidiValues.key_names[key_root].values))

            if not (are_all_positives([bits_per_note, population_size, output_size, notes_per_bar, no_bars, bpm], can_be_zero=False) \
                    and output_size <= population_size):

                print("\nInvalid options!")
                time.sleep(DEFAULT_SLEEP_SECS)

            else:
                valid_parameters = True

            clear_screen()

        note_length = time_signature.beats_per_bar / notes_per_bar
        key = MidiValues.key_names[key_root].values[key_type].values

        population, number_generations = run_evolution(
            populate_func=partial(
                gen_rand_population,
                population_size=population_size,
                bits_per_note=bits_per_note,
                note_length=note_length,
                no_bars=no_bars
            ),
            fitness_func=partial(
                fitness,
                key=key,
                note_length=note_length,
                bpm=bpm
            )
        )
        # retrieving best genomes

        for i in range(output_size):
            melody = population[i]

            mid = genome_to_midi(melody, key, note_length)
            mid.add_tempo(bpm)
            mid.write_midi("melody" + str((i + 1)), path="out")

        clear_screen()  # clears pyo prompt

        print("\nhighest rated melodies stored in midi files!")
        time.sleep(DEFAULT_SLEEP_SECS)

        clear_screen()
        continue_evolution = False if not confirmation("generate another melody?") else True


if __name__ == '__main__':
    main()

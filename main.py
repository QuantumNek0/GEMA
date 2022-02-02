from functools import partial

from algorithm.genetic import *
from utility.terminal import *
from utility.user_input import *
from utility.music import rand_harmonic_progression


def main():

    # default values
    bits_per_note = 4  # Deviation
    population_size = 5

    notes_per_bar = 8
    no_bars = 4
    bpm = 128

    harmonic_progression = []

    continue_evolution = True

    clear_screen() # clears the pyo prompt
    while continue_evolution:
        valid_parameters = False

        while not valid_parameters:

            if not confirmation("Use default values?"):

                if confirmation("Show advanced options?"):
                    bits_per_note = int(input("Bits per note: "))
                    population_size = int(input("Population size: "))

                notes_per_bar = int(input("Notes per bar: "))
                no_bars = int(input("Number of bars: "))
                bpm = int(input("Beats per minute: "))

            clear_screen()

            if not confirmation("randomize scale?"):
                scale_root = select_option((s.name for s in Scales), "Root of scale")
                scale_type = select_option((s.name for s in Scales[scale_root].values), "Scale type")

            else:
                scale_root = randrange(0, len(Scales))
                scale_type = randrange(0, len(Scales[scale_root].values))

            if not confirmation("randomize harmonic progression?"):
                for _ in range(4):
                    harmonic_progression += [int(input(">> "))]
            else:
                for _ in range(4):
                    # harmonic_progression += [randint(1, 7)]
                    harmonic_progression = rand_harmonic_progression()

            if not are_all_positives([bits_per_note, population_size, notes_per_bar, no_bars, bpm], can_be_zero=False) \
                    and not are_all_positives(harmonic_progression, can_be_zero=False):

                print("\nInvalid options!")
                time.sleep(DEFAULT_SLEEP_SECS)

            else:
                valid_parameters = True

            clear_screen()

        note_length = 4 / notes_per_bar  # the 4 represents the 4 beats in a single bar (time signature = 4/4)

        scale = Scales[scale_root].values[scale_type].values
        str_scale_type = "maj" if Scales[scale_root].values[scale_type].name.find("major") != -1 else "min"

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
                scale=scale,
                scale_type=str_scale_type,
                harmonic_progression=harmonic_progression,
                note_length=note_length,
                bpm=bpm
            )
        )
        # best_genome = population[0]
        #
        # mid = genome_to_midi(best_genome, scale, str_scale_type, harmonic_progression, note_length)
        # mid.add_tempo(bpm)
        # mid.write_midi("out/best_melody")

        for i in range(3):
            melody = population[i]

            mid = genome_to_midi(melody, scale, str_scale_type, harmonic_progression, note_length)
            mid.add_tempo(bpm)
            mid.write_midi("melody" + str((i + 1)), path="out")

        print("\nhighest rated melodies stored in midi file!")
        time.sleep(DEFAULT_SLEEP_SECS)

        clear_screen()
        continue_evolution = False if not confirmation("generate another melody?") else True


if __name__ == '__main__':
    main()

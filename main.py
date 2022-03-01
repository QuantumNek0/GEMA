from functools import partial

from algorithm.genetic import *
from utility.user_input import *
from utility.terminal import clear_screen, DEFAULT_SLEEP_SECS
from utility.music import rand_harmonic_progression


def main():

    # default values

    bits_per_note = 4
    population_size = 6
    notes_per_bar = 8
    no_bars = 4
    bpm = 128
    harmonic_progression = [] # 1 chord per bar

    generate_melody = True

    clear_screen() # clears the pyo prompt

    # main loop

    while generate_melody:
        valid_parameters = False

        # validating parameters

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

            if not confirmation("randomize harmonic progression?",
                                note="if the number of bars is not exactly 4, you'll be asked for an input regardless") or no_bars != 4:
                for _ in range(no_bars):
                    harmonic_progression += [int(input(">> "))]
            else:
                harmonic_progression = rand_harmonic_progression() # does not support progressions for more than 4 bars

            if not are_all_positives([bits_per_note, population_size, notes_per_bar, no_bars, bpm] + harmonic_progression, can_be_zero=False):

                print("\nInvalid options!")
                time.sleep(DEFAULT_SLEEP_SECS)

            else:
                valid_parameters = True

            clear_screen()

        # calculating special parameters

        note_length = 4 / notes_per_bar  # the 4 represents the 4 beats in a single bar (time signature = 4/4)

        scale = Scales[scale_root].values[scale_type].values
        str_scale_type = "maj" if Scales[scale_root].values[scale_type].name.find("major") != -1 else "min"

        # calling main evolution loop

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

        # retrieving best genomes

        for i in range(3):
            melody = population[i]

            mid = genome_to_midi(melody, scale, str_scale_type, harmonic_progression, note_length)
            mid.add_tempo(bpm)
            mid.write_midi("melody" + str((i + 1)), path="out")

        clear_screen() # clears pyo prompt

        print("\nhighest rated melodies stored in midi file!")
        time.sleep(DEFAULT_SLEEP_SECS)

        # main loop stop condition

        clear_screen()
        generate_melody = False if not confirmation("generate another melody?", note="Melodies will be overwritten") else True


if __name__ == '__main__':
    main()

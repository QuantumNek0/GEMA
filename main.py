from config import *

from algorithm.genetic import *
from utility.user_input import *
from utility.data_augmentation import DEFAULT_NOTE_DURATION


def main(key_root: int = 0, key_type: int = 0, bits_per_note: int = DEFAULT_BITS_PER_NOTE, population_size: int = DEFAULT_POPULATION_SIZE,
         output_size: int = DEFAULT_OUTPUT_SIZE, bpm: int = DEFAULT_BPM, use_vae: bool = True):

    note_length = DEFAULT_NOTE_DURATION
    no_bars = DEFAULT_NO_BARS
    clear_screen() # clears the pyo prompt

    continue_evolution = True
    while continue_evolution:

        key = MidiValues.key_names[key_root].values[key_type].values
        target = long_key_to_target[MidiValues.key_names[key_root].values[key_type].name]

        if use_vae:
            latent_dims = DEFAULT_LATENT_SIZE
            step_size = DEFAULT_STEP_SIZE

            vae = VariationalAutoencoder(latent_dims)
            vae.load_state_dict(torch.load("classes/files/music_vae.pt"))
            vae.eval()

            populate_func = partial(
                gen_latent_population,
                population_size=population_size,
                bits_per_note=bits_per_note,
                step_size=step_size,
                note_length=note_length,
                bpm=bpm,
                target=target,
                autoencoder=vae
            )
        else:
            populate_func = partial(
                gen_rand_population,
                population_size=population_size,
                bits_per_note=bits_per_note,
                note_length=note_length,
                no_bars=no_bars
            )

        population, number_generations = run_evolution(
            populate_func=populate_func,
            fitness_func=partial(
                fitness,
                key=key,
                note_length=note_length,
                bpm=bpm
            )
        )

        # writing best genomes
        for i in range(output_size):
            melody = population[i]

            mid = genome_to_midi(melody, key, note_length)
            mid.add_tempo(bpm)
            mid.write_midi("melody" + str((i + 1)), path="out")

        clear_screen()  # clears pyo prompt

        print("\nhighest rated melodies stored in midi files!")
        time.sleep(DEFAULT_SLEEP_SECS)

        clear_screen()
        continue_evolution = False if not ask("generate another melody?") else True


if __name__ == '__main__':
    main()

from config import *

from algorithm.genetic import *
from utility.user_input import *
from utility.data_augmentation import DEFAULT_NOTE_DURATION
import streamlit as st


def main(key_root: str = 'C', key_type: str = 'natural', key_mode: str = 'maj', bits_per_note: int = DEFAULT_BITS_PER_NOTE,
         population_size: int = DEFAULT_POPULATION_SIZE, output_size: int = DEFAULT_OUTPUT_SIZE, bpm: int = DEFAULT_BPM,
         first_gen_method: str = 'Random'):

    note_length = DEFAULT_NOTE_DURATION
    no_bars = DEFAULT_NO_BARS
    clear_screen() # clears the pyo prompt

    if key_type == 'natural':
        target = long_key_to_target[key_root + ' ' + key_mode + 'or']
    else:
        target = long_key_to_target[key_root + key_type + ' ' + key_mode + 'or']

    key = MidiValues.key[key_root][key_type][key_mode]

    if first_gen_method == 'VAE':
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
    elif first_gen_method == 'Random':
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

    st.write("highest rated melodies stored in 'out' directory!")


if __name__ == '__main__':
    main()

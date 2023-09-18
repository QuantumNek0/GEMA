# YouTube: Kie Codes

from config import *

from classes.autoencoder import VariationalAutoencoder, randn_latent_walk
from classes.midi import MIDI
from utility.user_input import ask
from utility.music import *
from utility.data_augmentation import AugmentedMidi, continuous

Note = List[int]
Genome = List[Note]
Population = List[Genome]

# functions to be easily called without problem specific arguments
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, List[int]], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]


def gen_rand_note(no_bits: int, note_probability: float = DEFAULT_NOTE_PROBABILITY) -> Note:
    """
    generates a random note represented by an array [], each index in the note represents the deviation from
    the scale root i.e. [1, 0, 0, 0] represents a 4 bit note representing the root note, [0, 1, 0, 0] is the
    adjacent note from the root acceptable for the scale (the note could go up or down, that is decided separately),
    finally [0, 0, 0, 0] is a rest

    Note: each note cannot have more than one bit being a ''1''

    :param no_bits: number of bits in the note, could be seen as deviation from the root
    :param note_probability: chance of returning a note instead of a rest
    :return: returns a random note
    """

    note = [0] * no_bits

    if random.random() <= note_probability:
        note[random.randint(0, no_bits - 1)] = 1

    return note


def gen_rand_genome(
            bits_per_note: int,
            note_length: float,
            no_bars: int,
            note_probability: float = DEFAULT_NOTE_PROBABILITY,
            blocks: bool = True
        ) -> Genome:
    """
    generates a random genome, each genome has an array of random notes

    :param bits_per_note: number of bits in each note
    :param note_length: the length of the note (in beats), each note has the same length
    :param no_bars: number of bars created
    :param note_probability: chance of returning a note instead of a rest
    :param blocks: if ''True'', returns each note into a different array [[note], [note], ...]
           instead of returning one big array [note + note + ...]
    :return: returns a random genome
    """

    no_notes = int(4 / note_length) * no_bars # the 4 represents the 4 beats in a single bar (time signature = 4/4)

    genome = [gen_rand_note(bits_per_note, note_probability) for _ in range(no_notes)]

    return genome if blocks else continuous(genome)


def gen_rand_population(
            population_size: int,
            bits_per_note: int,
            note_length: float,
            no_bars: int,
            note_probability: float = DEFAULT_NOTE_PROBABILITY,
        ) -> Population:
    """
    Generates a random population based on random notes

    :param population_size: size of population
    :param bits_per_note: bits per note
    :param note_length: length of the notes
    :param no_bars: total number of bars
    :param note_probability: probability of creating a note instead of a rest
    :return: returns a random population
    """

    return [gen_rand_genome(bits_per_note, note_length, no_bars, note_probability, blocks=True)
            for _ in range(population_size)]


def gen_latent_population(
            population_size: int,
            bits_per_note: int,
            step_size: float,
            note_length: float,
            bpm: int,
            target: int,
            autoencoder: VariationalAutoencoder,
        ) -> Population:
    """
    Generates a random population based on a vae latent space

    :param population_size: size of population
    :param step_size: the step size for the random walk in the latent space
    :param bits_per_note: bits per note
    :param note_length: length of the notes
    :param bpm: beats per minute
    :param target: key of the melody
    :param autoencoder: the autoencoder where the latent space is going to be sampled from
    :return: returns a latent population
    """

    from utility.music import key_encoding_key
    from classes.autoencoder import plot2d_latent, plot2d_walk, CsvDataset
    import streamlit as st

    num_steps = population_size - 1

    autoencoder.eval()
    autoencoder.vector_target = torch.load("./classes/files/vector_targets.pt")

    v0 = autoencoder.vector_target[target]
    latent_walk = randn_latent_walk(autoencoder.latent_dims, num_steps, step_size, v0)

    dataset = CsvDataset("classes/data/preprocessed_data/preprocessed_data.csv")
    fig_latent, ax_latent = plot2d_latent(autoencoder, dataset, show_plot=False)
    ax_latent = plot2d_walk(latent_walk, ax_latent)
    fig_latent.savefig("classes/temp/rand_walk.png")
    st.image("classes/temp/rand_walk.png", caption="Random Walk in Latent Space", use_column_width=True)

    autoencoder.decoder.eval()
    melodies = autoencoder.decoder(latent_walk)
    melodies = melodies.detach().numpy()

    population = []

    for melody in melodies:
        mid = MIDI(single_notes=True)

        for note in melody:
            mid.add_note(round(note * N_MIDI_VALUES), note_length)

        mid.add_key(key_encoding_key[target])
        mid.add_tempo(bpm)

        genome = midi_to_genome(mid, bits_per_note)
        population += [genome]

    return population


def fitness(genome: Genome, key: [], note_length: float, bpm: int) -> int:
    """
    rates the performance of a genome

    :param genome: genome
    :param key: the key is needed to play the midi file
    :param note_length: length of the notes
    :param bpm: beats per minute
    :return: returns a fitness level for a given genome, based on the rating of the user
    """
    from utility.user_input import clear_screen

    mid = genome_to_midi(genome, key, note_length)
    mid.add_tempo(bpm)

    mid = mid.smooth()
    mid.reboot_server()

    rating = 'r'
    while rating == 'r':
        clear_screen()  # clears pyo prompt

        print("playing melody...")
        mid.play()

        clear_screen()
        print("input 'r' for replay")

        rating = input("\nrating >> ")

    return abs(int(rating))


def single_point_crossover(genome_a: Genome, genome_b: Genome) -> Tuple[Genome, Genome]:
    """
    does a single point crossover between two genomes

    :param genome_a: first genome
    :param genome_b: second genome
    :return: returns a pair a genomes generated by a single point crossover on ''genome_a'' and ''genome_b''
    """

    point = random.randint(1, len(genome_a) - 1)

    return genome_a[0:point] + genome_b[point:], genome_b[0:point] + genome_a[point:]


def selection_pair(population: Population, fitness_values: List[int]) -> Population:
    """
    selects the best two performing genomes

    :param population: population of genomes
    :param fitness_values: array of fitness values to avoid calling the fitness function multiple times
    :return: returns a pair of genomes based on their fitness value, the two genomes with the highest value
             get returned
    """

    return random.sample(
        population=gen_weighted_dist(population, fitness_values),
        k=2
    )


def gen_weighted_dist(population: Population, fitness_values: List[int]) -> Population:
    """
    generates a weighted distribution of genomes based on their fitness value

    :param population: population of genomes
    :param fitness_values: array of fitness values to avoid calling the fitness function multiple times
    :return: returns the same population but each genome is duplicated several times based on their
             fitness value. The genomes with the highest value get duplicated several times, while the lowest ones
             just a few or none
    """

    weighted_population = []

    for i, genome in enumerate(population):
        weighted_population += [genome] * int(fitness_values[i] + 1)

    return weighted_population


def mutation(genome: Genome, no_mutations: int = 1,
             mutation_probability: float = 0.5, note_probability: float = 0.8) -> Genome:
    """
    mutates a given genome

    :param genome: original genome
    :param no_mutations: number of mutations that may occur
    :param mutation_probability: probability of a mutation actually occurring
    :param note_probability: probability of creating a note instead of a rest
    :return: returns a given genome with mutations
    """

    for _ in range(no_mutations):
        index = random.randrange(len(genome))

        if random.random() <= mutation_probability:
            genome[index] = gen_rand_note(len(genome[index]), note_probability)

    return genome


def genome_to_midi(genome: Genome, key: [], note_length: float,
                   high_note_prob: float = 0.6, octave_origin: int = 4) -> AugmentedMidi:
    """
    converts a given genome to a midi file

    :param genome: binary array
    :param key: collection of allowed values
    :param note_length: length of the notes in beats
    :param high_note_prob: chance of the deviation being above the root
    :param octave_origin: the pitch of the root
    :return: returns a MIDIFile object with the specified values from a genome
    """
    mid = AugmentedMidi(single_notes=True)

    root = key[0][octave_origin] # root is always at index 0
    key = continuous(key, return_sorted=True) # gets rid of the separations between notes
    root_pos = key.index(root) # finds the root in this new array

    for note in genome:
        rest_note = True
        for pitch, bit in enumerate(note):

            if bit == 1:
                mid.add_note(
                    key[root_pos + pitch] if random.random() <= high_note_prob else key[root_pos - pitch],
                    note_length
                )
                rest_note = False
                break

            if rest_note and pitch == len(note) - 1:
                mid.add_note(MidiValues.rest, note_length)

    return mid


def midi_to_genome(mid: MIDI, no_bits: int, track: int = 0, octave_origin: int = 4) -> Genome:
    """
    Converts given midi object to a genome

    :param mid: MIDI object
    :param no_bits: number of bits per note
    :param track: the track which the notes will be extracted from
    :param octave_origin: the octave at which the whole melody will center around
    :return: returns a Genome
    """
    key = key_name_to_key_values[mid.key]

    root = key[0][octave_origin]  # root is always at index 0
    key = continuous(key, return_sorted=True)  # gets rid of the separations between notes
    root_pos = key.index(root)  # finds the root in this new array

    track_notes = mid.melody[track]['notes']
    track_note_durations = mid.melody[track]['durations']

    genome = []

    for note, duration in zip(track_notes, track_note_durations):
        genome_note = [0] * no_bits

        try:
            note_pos = key.index(note)
        except ValueError:
            note_pos = root_pos

        pitch = abs(root_pos - note_pos)

        try:
            genome_note[pitch] = 1
        except IndexError:
            genome_note[-1] = 1

        genome += [genome_note]

    return genome


def sort_population(population: [], fitness_func: FitnessFunc) -> Tuple[Population, List[int]]:
    """
    Sorts the population from max to min according to their fitness value

    :param population: population of genomes
    :param fitness_func: criteria to assign fitness values to each genome
    :return: returns the sorted population along with their fitness values stored in a different array
    """

    fitness_values = []

    for genome in population:
        fitness_values += [fitness_func(genome)]

    # descending selection sort
    for i in range(len(population)):

        max_idx = i
        for j in range(i + 1, len(population)):
            if fitness_values[max_idx] < fitness_values[j]:
                max_idx = j

        population[i], population[max_idx] = population[max_idx], population[i]
        fitness_values[i], fitness_values[max_idx] = fitness_values[max_idx], fitness_values[i]

    return population, fitness_values


def run_evolution(
            populate_func: PopulateFunc,
            fitness_func: FitnessFunc,
            selection_func: SelectionFunc = selection_pair,
            crossover_func: CrossoverFunc = single_point_crossover,
            mutation_func: MutationFunc = mutation,
            generation_limit: int = DEFAULT_GENERATION_LIMIT
        ) -> Tuple[Population, int]:
    """
    runs the evolution with the given parameters

    :param populate_func: function to generate the initial population
    :param fitness_func: criteria to assign fitness values to each genome
    :param selection_func: criteria to select the best performing genomes
    :param crossover_func: method to crossover the best performing genomes
    :param mutation_func: method to mutate genomes
    :param generation_limit: limit of generations before the process ends itself
    :return: returns the final sorted population along with the total number of generations
             that the algorithm ran through
    """

    population = populate_func()

    generation = 0
    for generation in range(generation_limit):

        population, fitness_values = sort_population(population, fitness_func)

        if not ask("continue to next generation?"):
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):

            parents = selection_func(population, fitness_values)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)

            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population, generation

# YouTube: Kie Codes

from random import *
from typing import List, Callable, Tuple

from classes.midi import *
from utility.user_input import confirmation

DEFAULT_GENERATION_LIMIT = 100

Note = List[int]
Genome = List[Note]
Population = List[Genome]

# functions to be easily called without problem specific arguments

FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, List[int]], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]


def gen_rand_note(no_bits: int, note_probability: float = 0.7) -> Note:
    """

    generates a random note represented by an array [], each index in the note represents the deviation from
    the scale root i.e. [1, 0, 0, 0] represents a 4 bit note representing the root note, [0, 1, 0, 0] is the
    adjacent note from the root acceptable for the scale (the note could go up or down, that is decided separately),
    finally [0, 0, 0, 0] is a rest

    Note: each note cannot have more than one bit being a ''1''

    :param no_bits: number of bits in the note, could be seen as deviation from the root
    :param note_probability: chance of returning a note instead of a rest

    """

    note = [0] * no_bits

    if random.random() <= note_probability:
        note[randint(0, no_bits - 1)] = 1

    return note


def gen_rand_genome(
            bits_per_note: int,
            note_length: float,
            no_bars: int,
            note_probability: float = 0.7,
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

    """

    no_notes = int(4 / note_length) * no_bars # the 4 represents the 4 beats in a single bar (time signature = 4/4)

    genome = [gen_rand_note(bits_per_note, note_probability) for _ in range(no_notes)]

    return genome if blocks else continuous(genome)


def gen_rand_population(
            population_size: int,
            bits_per_note: int,
            note_length: float,
            no_bars: int,
            note_probability: float = 0.7,
        ) -> Population:
    """

    Generates a random population based on random notes

    :param population_size: size of population
    :param bits_per_note: bits per note
    :param note_length: length of the notes
    :param no_bars: total number of bars
    :param note_probability: probability of creating a note instead of a rest

    """

    return [gen_rand_genome(bits_per_note, note_length, no_bars, note_probability, blocks=True)
            for _ in range(population_size)]


def fitness(genome: Genome, scale: [], note_length: float, bpm: int) -> int:
    """

    returns a fitness level for a given genome, based on the rating of the user

    :param genome: genome
    :param scale: the scale is needed to play the midi file
    :param note_length: length of the notes
    :param bpm: beats per minute

    """

    mid = genome_to_midi(genome, scale, note_length)
    mid.add_tempo(bpm)

    mid.play()
    rating = int(input("\nrating >> "))

    return abs(rating)


def single_point_crossover(genome_a: Genome, genome_b: Genome) -> Tuple[Genome, Genome]:
    """

    does a single point crossover between two genomes

    :param genome_a: first genome
    :param genome_b: second genome

    """

    point = randint(1, len(a) - 1)

    return genome_a[0:point] + genome_b[point:], genome_b[0:point] + genome_a[point:]


def selection_pair(population: Population, fitness_values: List[int]) -> Population:
    """

    returns a pair of genomes based on their fitness value, the two genomes with the highest value
    get returned

    :param population: population of genomes
    :param fitness_values: array of fitness values to avoid calling the fitness function multiple times

    """

    return sample(
        population=gen_weighted_dist(population, fitness_values),
        k=2
    )


def gen_weighted_dist(population: Population, fitness_values: List[int]) -> Population:
    """

    returns the same population but each genome is duplicated several times based on their
    fitness value. The genomes with the highest value get duplicated several times, while the lowest ones
    just a few or none

    :param population: population of genomes
    :param fitness_values: array of fitness values to avoid calling the fitness function multiple times

    """

    weighted_population = []

    for i, genome in enumerate(population):
        weighted_population += [genome] * int(fitness_values[i] + 1)

    return weighted_population


def mutation(genome: Genome, no_mutations: int = 1,
             mutation_probability: float = 0.5, note_probability: float = 0.7) -> Genome:
    """

    returns a given genome with some mutations based on the number of mutations (''no_mutations'')
    and the probability of the mutation (''mutation_probability'')

    :param genome: original genome
    :param no_mutations: number of mutations that may ocurr
    :param mutation_probability: probability of a mutation ocurring
    :param note_probability: probability of creating a note instead of a rest

    """

    for _ in range(no_mutations):
        index = randrange(len(genome))

        if random.random() <= mutation_probability:
            genome[index] = gen_rand_note(len(genome[index]), note_probability)

    return genome


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

    """

    population = populate_func()

    generation = 0
    for generation in range(generation_limit):

        population, fitness_values = sort_population(population, fitness_func)

        if not confirmation("continue to next generation?"):
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


def genome_to_midi(genome: Genome, scale: [], note_length: float,
                   high_note_prob: float = 0.6, octave_origin: int = 4) -> MIDI:
    """

    returns a MIDIFile object with the specified values from a genome

    :param genome: binary array
    :param scale: collection of allowed values
    :param note_length: length of the notes in beats
    :param high_note_prob: chance of the deviation being above the root
    :param octave_origin: the pitch of the root

    """

    mid = MIDI(single_notes=True)

    root = scale[0][octave_origin] # root is always at index 0
    scale = continuous(scale, return_sorted=True) # gets rid of the separations between notes
    root_pos = scale.index(root) # finds the root in this new array

    for note in genome:
        rest_note = True
        for pitch, bit in enumerate(note):

            if bit == 1:
                mid.add_note(
                    scale[root_pos + pitch] if random.random() <= high_note_prob else scale[root_pos - pitch],
                    note_length
                )
                rest_note = False
                break

            if rest_note and pitch == len(note) - 1:
                mid.add_note(rest, note_length)

    return mid


def continuous(array: [], return_sorted: bool = False) -> []:
    """

    returns a single array [] from another array made of other arrays [[], [], ...]

    :param array: [[], [], ...]
    :param return_sorted: returns the continuous_array either sorted or not

    """

    continuous_array = []
    for arr in array:
        continuous_array += arr
    return sorted(continuous_array) if return_sorted else continuous_array


def sort_population(population: [], fitness_func: FitnessFunc) -> Tuple[Population, List[int]]:
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

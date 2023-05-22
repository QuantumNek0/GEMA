# Imports and global variables
import os
import time
from random import *
from tqdm import tqdm

from midiutil import *
from mido import MidiFile, tempo2bpm
from pyo import *

from typing import List, Callable, Tuple, NamedTuple
from collections import namedtuple
from functools import partial
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as torch_data
from torch.utils.data import random_split

# Music constants
TIME_SIGNATURE = namedtuple("TIME_SIGNATURE", ["beats_per_bar", "beat_note"])
NO_MAJ_KEYS = 12
NO_MIN_KEYS = 12
NO_KEYS = 24
N_MIDI_VALUES = 127

# Defaults
DEFAULT_BITS_PER_NOTE = 4
DEFAULT_OUTPUT_SIZE = 3

# Music defaults
DEFAULT_TIME_SIGNATURE = TIME_SIGNATURE(4, 4)
DEFAULT_UNIT_NOTE = 16 # (Semiquaver) the smallest note that will appear on a given melody
DEFAULT_NO_BARS = 4
DEFAULT_BPM = 128

# Defaults for genetic algorithm
DEFAULT_GENERATION_LIMIT = 100
DEFAULT_POPULATION_SIZE = 5
DEFAULT_NOTE_PROBABILITY = 0.45

# Defaults for autoencoder
DEFAULT_INPUT_SIZE = 64  # Size of midi melodies
DEFAULT_HIDDEN1_SIZE = 32  # Number of hidden units in VAE
DEFAULT_HIDDEN2_SIZE = 16  # Number of hidden units in VAE
DEFAULT_HIDDEN3_SIZE = 8  # Number of hidden units in VAE
DEFAULT_LATENT_SIZE = 2 # Dimensions of the latent space
DEFAULT_ACTIVATION = nn.ReLU()
# DEFAULT_CRITERION = nn.MSELoss()
DEFAULT_CLASSIFIER_CRITERION = nn.CrossEntropyLoss()
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_N_EPOCHS = 3
DEFAULT_BATCH_SIZE = 32

# Misc defaults
DEFAULT_SLEEP_SECS = 5

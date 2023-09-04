# Imports and global variables
from midiutil import *
from mido import MidiFile, tempo2bpm
from pyo import *

from typing import List, Callable, Tuple, NamedTuple
from collections import namedtuple
from functools import partial
import numpy as np
import pandas as pd
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as torch_data
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from torch.utils.data import random_split
from sklearn.metrics import precision_score, accuracy_score

import os
import time
# from random import *
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Torch Seed
torch.manual_seed(0)
plt.rcParams['figure.dpi'] = 150

# Music constants
TIME_SIGNATURE = namedtuple("TIME_SIGNATURE", ["beats_per_bar", "beat_note"])
NO_MAJ_KEYS = 12
NO_MIN_KEYS = NO_MAJ_KEYS
NO_KEYS = NO_MAJ_KEYS + NO_MIN_KEYS
N_MIDI_VALUES = 127

# Defaults
DEFAULT_BITS_PER_NOTE = 8
DEFAULT_OUTPUT_SIZE = 3
DEFAULT_STEP_SIZE = 0.1

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
DEFAULT_N_EPOCHS = 80
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_LATENT_SIZE = 2 # Dimensions of the latent space
DEFAULT_INDUCTIVE_BIAS_SIZE = 5

DEFAULT_KL_WEIGHT = 0.0010
DEFAULT_RECONSTRUCTION_WEIGHT = 0.0020
DEFAULT_CLASSIFIER_WEIGHT = 1
DEFAULT_ACTIVATION = nn.PReLU()
DEFAULT_CLASSIFIER_CRITERION = nn.CrossEntropyLoss()

# Encoder
DEFAULT_VAE_INPUT_SIZE = 64 + DEFAULT_INDUCTIVE_BIAS_SIZE  # Size of midi melodies
DEFAULT_VAE_HIDDEN1_SIZE = 32  # Number of hidden units in VAE
DEFAULT_VAE_HIDDEN2_SIZE = 16  # Number of hidden units in VAE
DEFAULT_VAE_HIDDEN3_SIZE = 8  # Number of hidden units in VAE

# Decoder
DEFAULT_CLASSIFIER_HIDDEN1_SIZE = 6
DEFAULT_CLASSIFIER_HIDDEN2_SIZE = 8
DEFAULT_CLASSIFIER_HIDDEN3_SIZE = 10
DEFAULT_CLASSIFIER_OUTPUT_SIZE = 12 # Number of classes (12 major keys)

# Misc defaults
DEFAULT_SLEEP_SECS = 5

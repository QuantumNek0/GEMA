# Imports and global variables
from midiutil import *
from mido import MidiFile, tempo2bpm
from pyo import *

import torch
import torch.nn as nn

from typing import List, Callable, Tuple, NamedTuple
from collections import namedtuple
from functools import partial
import copy

import os
from random import *
from tqdm import tqdm
import time

# Music constants
TIME_SIGNATURE = namedtuple("TIME_SIGNATURE", ["beats_per_bar", "beat_note"])
NO_MAJ_KEYS = 12
NO_MIN_KEYS = 12
NO_KEYS = 24

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

# Misc defaults
DEFAULT_SLEEP_SECS = 5

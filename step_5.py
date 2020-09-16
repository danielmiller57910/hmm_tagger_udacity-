import os
import pandas as pd
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from helpers import show_model, Dataset
import pdb
from pandas import json_normalize
from word_by_tag import word_by_tag
from emmission_state_list import emission_state_list

TRAINING_ALL_WORD_PATH = os.path.join(os.getcwd(), "training_all_words.csv")
TRAINING_WORD_PROBABILITY_MATRIX = os.path.join(os.getcwd(), "training_word_probability_matrix.csv")

emission_hash = emission_state_list(TRAINING_WORD_PROBABILITY_MATRIX, TRAINING_ALL_WORD_PATH)

example = emission_hash['NOUN']
model = HiddenMarkovModel('example')
model.add_states(example)
model.add_transition(model.start, example, 1.0)
model.add_transition(example, example, 1.0)
model.add_transition(example, model.end, 1.0)
model.bake()

res = model.viterbi(["time", "time", "time"])

for i, r in res[1]:
    print(i, r.name)
import os
import pandas as pd
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from helpers import show_model, Dataset
import pdb
from pandas import json_normalize
from word_by_tag import word_by_tag
from emmission_state_list import emission_state_list
from tag_aggregate import tag_aggregate_start_end

TRAINING_ALL_WORD_PATH = os.path.join(os.getcwd(), "training_all_words.csv")
TRAINING_WORD_PROBABILITY_MATRIX = os.path.join(os.getcwd(), "training_word_probability_matrix.csv")
TAG_TRAINING_PATH = os.path.join(os.getcwd(), "tag_training.csv")

emission_hash = emission_state_list(TRAINING_WORD_PROBABILITY_MATRIX, TRAINING_ALL_WORD_PATH)
start_prob, end_prob = tag_aggregate_start_end(TAG_TRAINING_PATH)
start_sum = sum(start_prob.values())
end_sum = sum(end_prob.values())
start_state_dict = {}
end_state_dict = {}

for k, v in start_prob.items():
    start_state_dict[k] = (v / start_sum)

for k, v in end_prob.items():
    end_state_dict[k] = (v / start_sum)

example = emission_hash['NOUN']
model = HiddenMarkovModel('example')

# add emmission states P(W|T)
model.add_states([val for val in emission_hash.values()])

# add start states P(T|S)
for k in start_state_dict:
    model.add_transition(model.start, emission_hash[k], start_state_dict[k])

# add end states P(T|E)
for k in end_state_dict:
    model.add_transition(emission_hash[k], model.end, end_state_dict[k])


model.add_transition(example, example, 1.0)
model.bake()

res = model.viterbi(["time", "time", "time", "can", "can"])

for i, r in res[1]:
    print(i, r.name)
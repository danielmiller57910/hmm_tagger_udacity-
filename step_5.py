import os
import pandas as pd
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from helpers import show_model, Dataset
import pdb
from pandas import json_normalize
from word_by_tag import word_by_tag
from utils import emission_state_list, tag_aggregate_start_end, bigram_sequence_probability

TRAINING_ALL_WORD_PATH = os.path.join(os.getcwd(), "training_all_words.csv")
TRAINING_WORD_PROBABILITY_MATRIX = os.path.join(os.getcwd(), "training_word_probability_matrix.csv")
TAG_TRAINING_PATH = os.path.join(os.getcwd(), "tag_training.csv")
BIGRAM_SEQUENCE_PATH = os.path.join(os.getcwd(), "bigram_sequence_training_path.csv")

emission_hash = emission_state_list(TRAINING_WORD_PROBABILITY_MATRIX, TRAINING_ALL_WORD_PATH)
bigram_hash = bigram_sequence_probability(BIGRAM_SEQUENCE_PATH)
for k in bigram_hash:
    evidence, prior = k.split()[0], k.split()[1]
    print(evidence, prior, bigram_hash[k])
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

# finally add P(T|T^-1)
for k in bigram_hash:
    evidence, prior = k.split()[0], k.split()[1]
    model.add_transition(emission_hash[evidence], emission_hash[prior], bigram_hash[k])

model.bake()

res = model.viterbi(["The", "big", "brown", "dog", "bark", "at", "my", "fox"])

for i, r in res[1]:
    print(i, r.name)
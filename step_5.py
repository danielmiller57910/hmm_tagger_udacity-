import os
import pandas as pd
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from helpers import show_model, Dataset
import pdb
from pandas import json_normalize
from word_by_tag import word_by_tag

TRAINING_ALL_WORD_PATH = os.path.join(os.getcwd(), "training_all_words.csv")
TRAINING_WORD_PROBABILITY_MATRIX = os.path.join(os.getcwd(), "training_word_probability_matrix.csv")

if not os.path.exists(TRAINING_WORD_PROBABILITY_MATRIX):
    word_by_tag(TRAINING_ALL_WORD_PATH, TRAINING_WORD_PROBABILITY_MATRIX)

df = pd.read_csv(TRAINING_WORD_PROBABILITY_MATRIX)
df.drop(columns=['Unnamed: 0', 'SUM'], inplace=True)
distinct_types = [col for col in df.columns if col not in ['Word', 'SUM']]

unigram_word_hash = {}

"""
Split df based on distinct tags
"""
for d in distinct_types:
    sample_df = df[df[d] > 0]
    unigram_word_hash[d] = sample_df[['Word', d]]

"""
P W|T V word in tag
"""
for k in unigram_word_hash:
    subset = unigram_word_hash[k]
    subset[k] = subset[k] / subset[k].sum()
    unigram_word_hash[k] = subset

"""
Assert probability distribution == 1
"""
for k in unigram_word_hash:
    subset = unigram_word_hash[k]
    print(f'{k} distribution sum => {subset[k].sum()}')

"""
Create discrete distribution objects
"""
state_list = []
for k in unigram_word_hash:
    dist = unigram_word_hash[k]
    print(k)
    # discrete prob {word: P(w|tag)}
    discrete_dist = DiscreteDistribution(dist.set_index("Word").T.to_dict("Records")[0])
    state_list.append(State(discrete_dist, name=k))


model = HiddenMarkovModel('example')
model.add_states(state_list)
model.add_transition(model.start, state_list[0], 1.0)
model.add_transition(state_list[0], state_list[0], 1.0)
model.add_transition(state_list[0], model.end, 1.0)
model.bake()
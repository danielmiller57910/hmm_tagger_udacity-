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

for d in distinct_types:
    sample_df = df[df[d] > 0]
    unigram_word_hash[d] = sample_df[['Word', d]]


for k in unigram_word_hash:
    subset = unigram_word_hash[k]
    print(f'sum {k} = > {subset[k].sum()}')
    subset[k] = subset[k] / subset[k].sum()
    unigram_word_hash[k] = subset

tst = unigram_word_hash['NOUN']
print(tst['NOUN'].sum())

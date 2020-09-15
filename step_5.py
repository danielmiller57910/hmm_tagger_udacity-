import os
import pandas as pd
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from helpers import show_model, Dataset
import pdb
from pandas import json_normalize

TRAINING_EMISSION_MAX_PATH = os.path.join(os.getcwd(), "training_emission_max.csv")
TRAINING_ALL_WORD_PATH = os.path.join(os.getcwd(), "training_all_words.csv")

TAG_PATH = os.path.join(os.getcwd(), "tags-universal.txt")
BROWN_PATH  = os.path.join(os.getcwd(), "brown-universal.txt")

# data = Dataset(TAG_PATH, BROWN_PATH, train_test_split=0.8)
all_word = pd.read_csv(TRAINING_ALL_WORD_PATH, index_col=False)
all_word.drop(columns=['Unnamed: 0'], inplace=True)
sample = all_word[all_word['Word'] == 'time'].groupby(['Word', 'Type']).size()
all_word = all_word.groupby(['Type', 'Word']).size()

word_probability_matrix = {}

for s in sample.items():
    word, word_type, count = s[0][0], s[0][1], s[1]
    print(word, word_type, count)
    if word not in word_probability_matrix:
        word_probability_matrix[word] = {word_type: count}
    else:
        current = word_probability_matrix[word]
        current[word_type] = count
        word_probability_matrix[word] = current

word_probability_list = []
for k in word_probability_matrix:
    insertion = word_probability_matrix[k]
    insertion['Word'] = k
    word_probability_list.append(insertion)

res = pd.DataFrame(word_probability_list)
print(res.head(10))
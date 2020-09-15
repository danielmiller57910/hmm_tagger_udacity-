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
all_word = all_word.groupby(['Word', 'Type']).size()

word_probability_matrix = {}
i = 0
for s in all_word.items():
    word, word_type, count = s[0][0], s[0][1], s[1]
    if word not in word_probability_matrix:
        word_probability_matrix[word] = {word_type: count}
    else:
        current = word_probability_matrix[word]
        current[word_type] = count
        word_probability_matrix[word] = current

insertion_list = []
for k in word_probability_matrix:
    current = word_probability_matrix[k]
    current['Word'] = k
    insertion_list.append(current)

res = pd.DataFrame.from_records(insertion_list)
res.fillna(0, inplace=True)

value_columns = [col for col in res.columns if col not in ['Word', 'SUM']]
res['SUM'] = res[value_columns].sum(axis=1)
sample = res[res['Word'] == 'time']
print(sample.columns)
print(value_columns)
for col in value_columns:
    sample[col] = sample[col] / sample['SUM']

print(sample.head(10))
# res['SUM'] = df[list(df.columns)].sum(axis=1)
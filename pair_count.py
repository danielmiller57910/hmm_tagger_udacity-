from helpers import show_model, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
import json
import pandas as pd
TAG_PATH = os.path.join(os.getcwd(), "tags-universal.txt")
BROWN_PATH  = os.path.join(os.getcwd(), "brown-universal.txt")
ALL_WORD_PATH = os.path.join(os.getcwd(), "all_words.csv")
UNIQUE_WORD_PATH = os.path.join(os.getcwd(), "unique_words.csv")
EMISSION_MAX_PATH = os.path.join(os.getcwd(), "emission_max.csv")
JSON_MAX_EMISSION_PATH = os.path.join(os.getcwd(), "out_emission.json")

def create_dataframes(sequences_A, sequences_B):
    unique_word_frame = pd.DataFrame(sequences_A, columns=["unique_words"])
    unique_word_frame.to_csv(UNIQUE_WORD_PATH)
    all_word_frame = pd.DataFrame(sequences_B.stream(), columns=["Word", "Type"])
    all_word_frame.to_csv(ALL_WORD_PATH)

def write_max_word_to_csv(word_aggregate):
    max_word = word_aggregate.loc[
        word_aggregate.groupby(['Type', 'Word'])['Count'].idxmax()
    ]

    max_word.to_csv(EMISSION_MAX_PATH)

def pair_counts(sequences_A, sequences_B):

    if not os.path.exists(UNIQUE_WORD_PATH):
        create_dataframes(sequences_A, sequences_B)

    if not os.path.exists(EMISSION_MAX_PATH):
        unique_df, all_df = pd.read_csv(UNIQUE_WORD_PATH), pd.read_csv(ALL_WORD_PATH)
        unique_df.drop(columns=['Unnamed: 0'], inplace=True)
        all_df.drop(columns=['Unnamed: 0'], inplace=True)
        word_aggregate = all_df.groupby(['Type', 'Word']).size().reset_index(name='Count')
        write_max_word_to_csv(word_aggregate)

    if not os.path.exists(JSON_MAX_EMISSION_PATH):
        word_by_max = pd.read_csv(EMISSION_MAX_PATH)
        word_by_max.drop(columns=['Unnamed: 0'], inplace=True)
        word_by_max = word_by_max.groupby('Type').apply(lambda x: x.to_dict(orient='records')).to_dict()

        sequences = {}

        for word_type in word_by_max:
            for i, j in enumerate(word_by_max[word_type]):
                if word_type not in sequences:
                    sequences[word_type] = {}
                sequences[word_type][j['Word']] = j['Count']
    
        with open(JSON_MAX_EMISSION_PATH, 'w') as max_path:
            json.dump(sequences, max_path)

data = Dataset(TAG_PATH, BROWN_PATH, train_test_split=0.8)
pair_counts(data.vocab, data)
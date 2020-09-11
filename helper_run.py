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

def create_dataframes(sequences_A, sequences_B):
    unique_word_frame = pd.DataFrame(sequences_A, columns=["unique_words"])
    unique_word_frame.to_csv(UNIQUE_WORD_PATH)
    all_word_frame = pd.DataFrame(sequences_B.stream(), columns=["Word", "Type"])
    all_word_frame.to_csv(ALL_WORD_PATH)

def pair_counts(sequences_A, sequences_B):
    if not os.path.exists(UNIQUE_WORD_PATH):
        create_dataframes(sequences_A, sequences_B)
    
    unique_df, all_df = (
        pd.read_csv(UNIQUE_WORD_PATH), pd.read_csv(ALL_WORD_PATH)
    )

    unique_df.drop(columns=['Unnamed: 0'], inplace=True)
    all_df.drop(columns=['Unnamed: 0'], inplace=True)
    word_aggregate = all_df.groupby(['Type', 'Word']).size().reset_index(name='Count')
    print(f'Number of unique words => {len(unique_df)}')
    print(f'Number of aggregate words => {len(word_aggregate)}')
    
    max_word = word_aggregate.loc[
        word_aggregate.groupby(['Type', 'Word'])['Count'].idxmax()
    ]

    print(max_word.loc[max_word['Word'] == 'time'])
    print(max_word.head(10))

data = Dataset(TAG_PATH, BROWN_PATH, train_test_split=0.8)
pair_counts(data.vocab, data)
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

def create_dataframes(sequences_A, sequences_B, unique_csv_path, all_csv_path):
    unique_word_frame = pd.DataFrame(sequences_A, columns=["unique_words"])
    unique_word_frame.to_csv(unique_csv_path)
    all_word_frame = pd.DataFrame(sequences_B.stream(), columns=["Word", "Type"])
    all_word_frame.to_csv(all_csv_path)

def pair_counts(sequences_A, sequences_B, unique_csv_path, all_csv_path, emission_max_path, json_max_path):
    #checkpoint 1: create raw dataframes from tuples
    if not os.path.exists(unique_csv_path):
        create_dataframes(sequences_A, sequences_B, unique_csv_path, all_csv_path)
        
    #checkpoint 2: create dataframe consisting of word, category, count V max(count) in category
    if not os.path.exists(emission_max_path):
        unique_df, all_df = pd.read_csv(unique_csv_path), pd.read_csv(all_csv_path)
        unique_df.drop(columns=['Unnamed: 0'], inplace=True)
        all_df.drop(columns=['Unnamed: 0'], inplace=True)
        word_aggregate = all_df.groupby(['Type', 'Word']).size().reset_index(name='Count')
        word_aggregate.to_csv(emission_max_path)

    #checkpoint 3: convert df -> {V category 'Category': {'Word': 'Count' V word}}
    if not os.path.exists(json_max_path):
        word_by_max = pd.read_csv(emission_max_path)
        word_by_max.drop(columns=['Unnamed: 0'], inplace=True)
        word_by_max = word_by_max.groupby('Type').apply(lambda x: x.to_dict(orient='records')).to_dict()
        sequences = {}
        for word_type in word_by_max:
            for word in word_by_max[word_type]:
                if word_type not in sequences:
                    sequences[word_type] = {}
                sequences[word_type][word['Word']] = word['Count']
            
        with open(json_max_path, 'w') as max_path:
            json.dump(sequences, max_path)
    
    with open(json_max_path, 'r') as pair_count_file:
        mapping_dict = json.load(pair_count_file)
    
    return mapping_dict


data = Dataset(TAG_PATH, BROWN_PATH, train_test_split=0.8)
emission_counts = pair_counts(
    data.vocab, 
    data, 
    UNIQUE_WORD_PATH, 
    ALL_WORD_PATH, 
    EMISSION_MAX_PATH,
    JSON_MAX_EMISSION_PATH)

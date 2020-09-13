import pandas as pd
import os
import pdb
TRAINING_ALL_WORD_PATH = os.path.join(os.getcwd(), "training_all_words.csv")

def unigram_counts(sequences):
    df = sequences.drop(columns=["Unnamed: 0", "Word"])
    df = df.groupby("Type").size().reset_index()
    return df.set_index("Type").T.to_dict("Records")[0]

tag_unigrams = unigram_counts(pd.read_csv(TRAINING_ALL_WORD_PATH))
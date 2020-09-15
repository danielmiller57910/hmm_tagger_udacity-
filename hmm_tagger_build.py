import pandas as pd
import os
import pdb
from helpers import show_model, Dataset

TRAINING_ALL_WORD_PATH = os.path.join(os.getcwd(), "training_all_words.csv")
TAG_PATH = os.path.join(os.getcwd(), "tags-universal.txt")
BROWN_PATH  = os.path.join(os.getcwd(), "brown-universal.txt")
BIGRAM_SEQUENCE_TRAINING_PATH = os.path.join(os.getcwd(), "bigram_sequence_training_path.csv")
TAG_TRAINING_PATH = os.path.join(os.getcwd(), "tag_training.csv")

def unigram_counts(sequences):
    df = sequences.drop(columns=["Unnamed: 0", "Word"])
    df = df.groupby("Type").size().reset_index()
    return df.set_index("Type").T.to_dict("Records")[0]


def bigram_counts(sequences, bigram_training_path):

    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.
    
    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582
    """
    key_to_tuple = lambda x: (x.split(' ')[0], x.split(' ')[-1])

    if not os.path.exists(bigram_training_path):
        bigram_sequence_set = []
        for seq in sequences:
            seq = list(seq)
            for j in range(len(seq)-1):
                tup = (seq[j], seq[j + 1])
                bigram_sequence_set.append(tup)

        res = pd.DataFrame([" ".join(tup) for tup in bigram_sequence_set], columns=["bigram_sequence"])
        res = res.groupby("bigram_sequence").size().reset_index(name="Count")
        res.to_csv(bigram_training_path)
    
    df = pd.read_csv(bigram_training_path)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    dct = df.set_index("bigram_sequence").T.to_dict("Records")[0]
    return {key_to_tuple(k): dct[k] for k in dct}

def tag_aggregate(sequences):

    if not os.path.exists(TAG_TRAINING_PATH):
        start_end_frame = []
        for i, seq in enumerate(sequences):
            tup = (seq[0], seq[-1])
            start_end_frame.append(tup)
    
        df = pd.DataFrame(start_end_frame, columns=["start_type", "end_type"])
        df.to_csv(TAG_TRAINING_PATH)
    
    df = pd.read_csv(TAG_TRAINING_PATH)
    start_frame, end_frame = df.start_type.value_counts(), df.end_type.value_counts()
    return start_frame.to_dict, end_frame.to_dict()

data = Dataset(TAG_PATH, BROWN_PATH, train_test_split=0.8)
tag_starts, tag_end = tag_aggregate(data.training_set.Y)

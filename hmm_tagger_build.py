import pandas as pd
import os
import pdb
from helpers import show_model, Dataset

TRAINING_ALL_WORD_PATH = os.path.join(os.getcwd(), "training_all_words.csv")
TAG_PATH = os.path.join(os.getcwd(), "tags-universal.txt")
BROWN_PATH  = os.path.join(os.getcwd(), "brown-universal.txt")
BIGRAM_SEQUENCE_TRAINING_PATH = os.path.join(os.getcwd(), "bigram_sequence_training_path.csv")

def unigram_counts(sequences):
    df = sequences.drop(columns=["Unnamed: 0", "Word"])
    df = df.groupby("Type").size().reset_index()
    return df.set_index("Type").T.to_dict("Records")[0]

def bigram_counts(sequences):

    if not os.path.exists(BIGRAM_SEQUENCE_TRAINING_PATH):
        bigram_sequence_set = []
        for seq in sequences:
            seq = list(seq)
            for j in range(len(seq)-1):
                tup = (seq[j], seq[j + 1])
                bigram_sequence_set.append(tup)

        res = pd.DataFrame([" ".join(tup) for tup in bigram_sequence_set], columns=["bigram_sequence"])
        res = res.groupby("bigram_sequence").size().reset_index(name="Count")
        res.to_csv(BIGRAM_SEQUENCE_TRAINING_PATH)
    
    df = pd.read_csv(BIGRAM_SEQUENCE_TRAINING_PATH)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df
        
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.
    
    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582
    """

# code passing
# tag_unigrams = unigram_counts(pd.read_csv(TRAINING_ALL_WORD_PATH))
data = Dataset(TAG_PATH, BROWN_PATH, train_test_split=0.8)

print(len(data.training_set.Y))
tag_bigrams = bigram_counts(data.training_set.Y)
print(tag_bigrams.head(100))
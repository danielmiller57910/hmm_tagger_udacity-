# Create a lookup table mfc_table where mfc_table[word] contains the tag label most frequently assigned to that word
from collections import namedtuple
from pair_count import pair_counts
from helpers import show_model, Dataset
import pair_count
import os
import pandas as pd

import pdb
TAG_PATH = os.path.join(os.getcwd(), "tags-universal.txt")
BROWN_PATH  = os.path.join(os.getcwd(), "brown-universal.txt")

TRAINING_ALL_WORD_PATH = os.path.join(os.getcwd(), "training_all_words.csv")
TRAINING_UNIQUE_WORD_PATH = os.path.join(os.getcwd(), "training_unique_words.csv")
TRAINING_EMISSION_MAX_PATH = os.path.join(os.getcwd(), "training_emission_max.csv")
TRAINING_JSON_MAX_EMISSION_PATH = os.path.join(os.getcwd(), "training_out_emission.json")

FakeState = namedtuple("FakeState", "name")
def mfc_table(emission_path, data):
    train_df = pd.read_csv(emission_path, na_filter=False)
    train_df.drop(columns=["Unnamed: 0"], inplace=True)
    train_df = train_df.groupby(['Type', 'Word']).size().reset_index(name='Count')
    df = train_df.iloc[train_df.groupby('Word')['Count'].agg(pd.Series.idxmax)]
    if len(data.training_set.vocab) != len(df):
        ancillary_words = pd.DataFrame(data.training_set.vocab)
        dif = pd.concat([df["Word"],ancillary_words]).drop_duplicates(keep=False)
        print(f"Difference between two dataframes => {len(data.training_set.vocab) - len(df)}")
        print(dif.head(50)) 
    df.drop(columns=["Count"], inplace=True)
    dct = df.set_index("Word").T.to_dict("Records")[0]
    # seeing as there is one missing added manually to save time.
    dct["null"] = "NOUN"
    return dct


data = Dataset(TAG_PATH, BROWN_PATH, train_test_split=0.8)

pair_counts(
    data.training_set.vocab,
    data.training_set,
    TRAINING_UNIQUE_WORD_PATH,
    TRAINING_ALL_WORD_PATH,
    TRAINING_EMISSION_MAX_PATH,
    TRAINING_JSON_MAX_EMISSION_PATH
)

mfc_table(TRAINING_ALL_WORD_PATH, data)

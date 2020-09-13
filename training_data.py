# Create a lookup table mfc_table where mfc_table[word] contains the tag label most frequently assigned to that word
from collections import namedtuple
from pair_count import pair_counts
from helpers import show_model, Dataset
import pair_count
import os

TAG_PATH = os.path.join(os.getcwd(), "tags-universal.txt")
BROWN_PATH  = os.path.join(os.getcwd(), "brown-universal.txt")

TRAINING_ALL_WORD_PATH = os.path.join(os.getcwd(), "training_all_words.csv")
TRAINING_UNIQUE_WORD_PATH = os.path.join(os.getcwd(), "training_unique_words.csv")
TRAINING_EMISSION_MAX_PATH = os.path.join(os.getcwd(), "training_emission_max.csv")
TRAINING_JSON_MAX_EMISSION_PATH = os.path.join(os.getcwd(), "training_out_emission.json")

FakeState = namedtuple("FakeState", "name")

class MFCTagger:
    # NOTE: You should not need to modify this class or any of its methods
    missing = FakeState(name="<MISSING>")
    
    def __init__(self, table):
        self.table = defaultdict(lambda: MFCTagger.missing)
        self.table.update({word: FakeState(name=tag) for word, tag in table.items()})
        
    def viterbi(self, seq):
        """This method simplifies predictions by matching the Pomegranate viterbi() interface"""
        return 0., list(enumerate(["<start>"] + [self.table[w] for w in seq] + ["<end>"]))


data = Dataset(TAG_PATH, BROWN_PATH, train_test_split=0.8)
word_counts = pair_counts(
    data.training_set.vocab,
    data.training_set,
    TRAINING_UNIQUE_WORD_PATH,
    TRAINING_ALL_WORD_PATH,
    TRAINING_EMISSION_MAX_PATH,
    TRAINING_JSON_MAX_EMISSION_PATH
)

print(word_counts["NOUN"]["time"])

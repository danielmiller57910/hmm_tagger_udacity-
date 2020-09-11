# Create a lookup table mfc_table where mfc_table[word] contains the tag label most frequently assigned to that word
from collections import namedtuple
from pair_count import pair_counts
from helpers import show_model, Dataset
import pair_count


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


data = Dataset(pair_count.TAG_PATH, pair_count.BROWN_PATH, train_test_split=0.8)

print(dir(data.training_set))

for i, j in enumerate(data.training_set.stream()):
    print(i, j)
    if i == 10:
        break
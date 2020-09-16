import os
import pandas as pd
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

def emission_state_list(training_word_prob_path, training_all_word_path):
    if not os.path.exists(training_word_prob_path):
        word_by_tag(training_all_word_path, training_word_prob_path)

    df = pd.read_csv(training_word_prob_path)
    df.drop(columns=['Unnamed: 0', 'SUM'], inplace=True)
    distinct_types = [col for col in df.columns if col not in ['Word', 'SUM']]

    unigram_word_hash = {}

    """
    Split df based on distinct tags
    """
    for d in distinct_types:
        sample_df = df[df[d] > 0]
        unigram_word_hash[d] = sample_df[['Word', d]]

    """
    P W|T V word in tag
    """
    for k in unigram_word_hash:
        subset = unigram_word_hash[k]
        subset[k] = subset[k] / subset[k].sum()
        unigram_word_hash[k] = subset

    """
    Assert probability distribution == 1
    """
    for k in unigram_word_hash:
        subset = unigram_word_hash[k]
        print(f'{k} distribution sum => {subset[k].sum()}')

    """
    Create discrete distribution objects
    """
    for k in unigram_word_hash:
        dist = unigram_word_hash[k]
        print(k)
        dist_dict = dist.set_index("Word").T.to_dict("Records")[0]
        if k == 'NOUN':
            print(dist_dict['time'])
        discrete_dist = DiscreteDistribution(dist_dict)
        unigram_word_hash[k] = State(discrete_dist, name=k)
    
    return unigram_word_hash

def tag_aggregate_start_end(tag_training_path):

    df = pd.read_csv(tag_training_path)
    start_frame, end_frame = df.start_type.value_counts(), df.end_type.value_counts()
    return start_frame.to_dict(), end_frame.to_dict()

def bigram_sequence_probability(bigram_training_path):
    df = pd.read_csv(bigram_training_path)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    print(df.head(50))
    total_count = df['Count'].sum()
    df['Count'] = df['Count'] / total_count
    print(df['Count'].sum())
    dct = df.set_index("bigram_sequence").T.to_dict("Records")[0]
    return dct
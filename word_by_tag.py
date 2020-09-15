import pandas as pd

def word_by_tag(raw_path, word_probability_path):
    all_word = pd.read_csv(raw_path, index_col=False)
    all_word.drop(columns=['Unnamed: 0'], inplace=True)
    all_word = all_word.groupby(['Word', 'Type']).size()

    word_probability_matrix = {}

    for s in all_word.items():
        word, word_type, count = s[0][0], s[0][1], s[1]
        if word not in word_probability_matrix:
            word_probability_matrix[word] = {word_type: count}
        else:
            current = word_probability_matrix[word]
            current[word_type] = count
            word_probability_matrix[word] = current

    insertion_list = []
    for k in word_probability_matrix:
        current = word_probability_matrix[k]
        current['Word'] = k
        insertion_list.append(current)

    res = pd.DataFrame.from_records(insertion_list)
    res.fillna(0, inplace=True)

    value_columns = [col for col in res.columns if col not in ['Word', 'SUM']]
    res['SUM'] = res[value_columns].sum(axis=1)
    res.to_csv(word_probability_path)
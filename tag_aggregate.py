import pandas as pd

def tag_aggregate_start_end(tag_training_path):

    df = pd.read_csv(tag_training_path)
    start_frame, end_frame = df.start_type.value_counts(), df.end_type.value_counts()
    return start_frame.to_dict(), end_frame.to_dict()
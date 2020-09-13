import pandas as pd
import os

TRAINING_EMISSION_MAX_PATH = os.path.join(os.getcwd(), "training_emission_max.csv")


train_df = pd.read_csv(TRAINING_EMISSION_MAX_PATH)
train_df.drop(columns=["Unnamed: 0"], inplace=True)
print(train_df.head(50))
print(train_df.loc[train_df["Word"] == "time"])
df = train_df.groupby("Count").max()
df.drop(columns=["Count"], inplace=True)
print(df.loc[df["Word"] == "time"])

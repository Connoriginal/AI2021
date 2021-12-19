import pandas as pd

df = pd.read_csv("./ratings.csv")

train_data = df.sample(frac=0.8)
valid_data = df.drop(train_data.index)

train_data.to_csv("./train.csv")
valid_data.to_csv("./valid.csv")
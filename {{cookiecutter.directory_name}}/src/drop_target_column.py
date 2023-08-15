import pandas as pd

df = pd.read_csv("./data/bank_preprocessed.csv")

df = df.drop(columns=["deposit"])

df.to_csv("./data/bank_predict.csv", index=False)
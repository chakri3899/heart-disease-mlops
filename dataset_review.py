import pandas as pd

df = pd.read_csv("data/heart.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.columns)
import pandas as pd

df = pd.read_csv("test.csv")

# print(df.iloc[0:3,2:4])
print(df.loc[[0,3,5], ['1','2','4']])

print(df[df["1"] >15])


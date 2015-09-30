import pandas as pd


avengers = pd.read_csv("avengers.csv")
print(avengers.head(5))

true_avengers = pd.DataFrame()
avengers['Year'].hist()

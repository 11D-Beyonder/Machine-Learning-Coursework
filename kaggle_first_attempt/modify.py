import pandas as pd

# 划分一份统一的数据
dataset = pd.read_excel('Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx')
# shuffle
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset.to_csv('Date_Fruit_Datasets/modified.csv')

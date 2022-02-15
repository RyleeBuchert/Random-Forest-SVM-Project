import pandas as pd

mushrooms = pd.read_csv('data\\agaricus-lepiota.data')

print()

mushrooms.to_csv('data\\mushrooms_data.csv', index=False)
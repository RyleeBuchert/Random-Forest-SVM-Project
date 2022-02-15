import pandas as pd

car = pd.read_csv('data\\car.data')

print()

car.to_csv('data\\car_data.csv', index=False)
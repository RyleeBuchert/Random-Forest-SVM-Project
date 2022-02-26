import pandas as pd

car = pd.read_csv('data\\car_data.csv')

mapping1 = {
    'vhigh': 4,
    'high': 3,
    'med': 2,
    'low': 1
}

mapping2 = {
    '2':2,
    '3':3,
    '4':4,
    '5more':5
}

mapping3 = {
    '2':2,
    '3':3,
    '4':4,
    'more':5
}

mapping4 = {
    'small':1,
    'med':2,
    'big':3
}

mapping5 = {
    'high': 3,
    'med': 2,
    'low': 1
}

label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}

car['buying_price'] = car.buying_price.map(mapping1)
car['maintenance_price'] = car.maintenance_price.map(mapping1)
car['doors'] = car.doors.map(mapping2)
car['persons'] = car.persons.map(mapping3)
car['lug_boot'] = car.lug_boot.map(mapping4)
car['safety'] = car.safety.map(mapping5)
car['label'] = car.label.map(label_mapping)

print()

car.to_csv('data\\car_bins.csv')
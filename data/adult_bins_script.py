import pandas as pd
from sklearn.preprocessing import LabelEncoder

adult = pd.read_csv('data\\adult_data.csv')

le = LabelEncoder()
adult['workclass'] = le.fit_transform(adult['workclass'])
adult['marital-status'] = le.fit_transform(adult['marital-status'])
adult['occupation'] = le.fit_transform(adult['occupation'])
adult['relationship'] = le.fit_transform(adult['relationship'])
adult['race'] = le.fit_transform(adult['race'])
adult['sex'] = le.fit_transform(adult['sex'])
adult['native-country'] = le.fit_transform(adult['native-country'])
adult['income'] = le.fit_transform(adult['income'])

adult = adult.drop(columns='education', axis=1)
adult.to_csv('data\\adult_bins.csv')
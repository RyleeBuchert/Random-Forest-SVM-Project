import pandas as pd
import numpy as np

cols=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
adult_test = pd.read_csv('data\\adult_test.csv', skipinitialspace=True)
adult_train = pd.read_csv('data\\adult_train.csv', skipinitialspace=True)

# adult_test = adult_test[~(adult_test.astype(str) == ' ?').any(1)]
# adult_train = adult_train[~(adult_train.astype(str) == ' ?').any(1)]

one = adult_test.iloc[0]['income']
two = adult_test.iloc[0]['income']

adult_test['income'] = adult_test['income'].str.replace('.', '')

adult_test.to_csv('data\\adult_test.csv')
adult_train.to_csv('data\\adult_train.csv')
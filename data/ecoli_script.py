import pandas as pd

cols=['sequence_name','mcg','gvh']
ecoli = pd.read_csv('data\\ecoli.data', delim_whitespace=True)
ecoli = ecoli.drop(columns='sequence_name', axis=1)

ecoli.to_csv('data\\ecoli_data.csv')

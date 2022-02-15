import pandas as pd

test = pd.read_csv('data\\digit_continuous.csv')
test_label = test['label']
test = test.drop(columns='label')
labels = [0, 1, 2, 3, 4]
bins = [0, 51.5, 102.5, 153.5, 204.5, 255.5]
for col in test.columns.tolist():
    test[col] = pd.cut(test[col], bins, right=False, labels=labels, include_lowest=True)
test = pd.concat([test_label, test], axis=1)
test.dropna(axis=0, inplace=True)
test.to_csv('data\\digit_bins.csv')
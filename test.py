dict = {
    'No': {'Count': 2, 'Percent': 0.5},
    'Yes': {'Count': 1, 'Percent': 0.5}
}

print(max(dict, key=dict.get))

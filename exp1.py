import pandas as pd

data = pd.read_csv("finds1.csv")

h = ['Ø'] * (len(data.columns) - 1)

for row in data.values:
    if str(row[-1]).lower() == 'yes':
        if 'Ø' in h:
            h = list(row[:-1])
        else:
            for i in range(len(h)):
                if h[i] != row[i]:
                    h[i] = '?'

print("Final Hypothesis:", h)

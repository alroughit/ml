
import pandas as pd

data = pd.read_csv("enjoysport.csv")
S = ['∅'] * (len(data.columns) - 1)
G = [['?'] * len(S)]

def match(h, x):
    return all(a == '?' or a == b for a, b in zip(h, x))

for row in data.values.tolist():
    x, y = row[:-1], row[-1]

    if y == "Yes":
        for i in range(len(S)):
            if S[i] == '∅':
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = '?'
        G = [g for g in G if match(g, x)]

    else:
        new_G = []
        for g in G:
            if match(g, x):
                for i in range(len(S)):
                    if g[i] == '?' and S[i] != '?':
                        h = g.copy()
                        h[i] = S[i]
                        if not match(h, x):
                            new_G.append(h)
            else:
                new_G.append(g)
        G = new_G

G = [list(h) for h in dict.fromkeys(map(tuple, G))]

print("S =", S)
print("G =", G)

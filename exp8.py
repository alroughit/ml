```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

df = pd.read_csv("simple_diabetes_dataset.csv")
X = df.drop("Outcome", axis=1).values

kmeans = KMeans(n_clusters=2, random_state=42)
k = kmeans.fit_predict(X)

gmm = GaussianMixture(n_components=2, random_state=42)
g = gmm.fit_predict(X)

print("K-Means Score:", silhouette_score(X, k))
print("EM Score:", silhouette_score(X, g))

plt.scatter(X[:, 0], X[:, 1], c=k)
plt.title("K-Means Clustering")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=g)
plt.title("EM Clustering")
plt.show()
```

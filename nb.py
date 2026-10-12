import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load data and split
df = pd.read_csv('Naive-Bayes-Classification-Data.csv')
X, y = df.drop('diabetes', axis=1), df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train & Predict
model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test).plot()

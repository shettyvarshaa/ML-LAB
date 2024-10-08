import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv("iris.csv", header=None, names=column_names)
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Optional: if dataset isn't given 
# iris = load_iris()
# X, y = iris.data, iris.target
# df = pd.DataFrame(X, columns=iris.feature_names)
# X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

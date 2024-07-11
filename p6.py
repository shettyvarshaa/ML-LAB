from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

iris= load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.35, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))

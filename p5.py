import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Classification Report:\n', classification_report(y_test,y_pred))

plt.figure(figsize=(12, 8))
plot_tree(clf,feature_names=load_iris().feature_names, class_names=load_iris().target_names,  filled=True)
plt.show()

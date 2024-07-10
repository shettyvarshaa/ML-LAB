from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

newsgroups = fetch_20newsgroups_vectorized()

X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)
nb_classifier = MultinomialNB().fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
print(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=newsgroups.target_names)}")
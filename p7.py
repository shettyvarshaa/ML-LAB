from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

data = [
    ("I love this movie", "positive"),
    ("This film was amazing", "positive"),
    ("I actually enjoyed it", "positive"),
    ("I hated that movie", "negative"),
    ("This film was terrible", "negative"),
    ("I did not like it", "negative")
]

text, label = zip(*data)
vect = CountVectorizer()
X = vect.fit_transform(text)
clf = MultinomialNB().fit(X, label)

test = [
    "I love this film",
    "I hated the movie",
    "It was an awesome movie",
    "This movie was not good"
]

X_test = vect.transform(test)
y_pred = clf.predict(X_test)
print("Predicted labels:", y_pred)


# Actual labels for the test data
true_labels = ["positive", "negative", "positive", "negative"]
print("Classification Report:")
print(classification_report(true_labels, y_pred))
print("Accuracy Score:", accuracy_score(true_labels, y_pred))

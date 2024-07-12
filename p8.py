import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()

## IF CSV file isn't given, then do this :
# iris = load_iris()
# df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# df['species'] = iris.target
# species_mapping = dict(zip(range(3), iris.target_names))
# df['species'] = df['species'].map(species_mapping)
# df.to_csv('iris_dataset.csv', index=False)

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv("./iris.csv", header=None, names=column_names)
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy:{accuracy_score(y_test, y_pred)}\n")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")


## WITHOUT CSV FILE
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score,classification_report
# from sklearn.naive_bayes import GaussianNB
# import pandas as pd
# data=load_breast_cancer()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# print(df.head())
# x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2,random_state=42)
# model=GaussianNB()
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test) 
# print(accuracy_score(y_test,y_pred))
# print(classification_report(y_test,y_pred))

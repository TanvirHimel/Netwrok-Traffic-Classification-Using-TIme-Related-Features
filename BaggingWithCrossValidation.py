import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn import tree

bankdata = pd.read_csv("C:/Users/USER/Desktop/Thesis Work/Scenario B/CSV/TimeBasedFeatures_Dataset_15s.csv")

X = bankdata.drop('class1', axis=1)
y = bankdata['class1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

scores = []
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
cv = KFold(n_splits=5, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))
cm=confusion_matrix(y_test,predictions)
print(cm)
print("Accuracy : ")
print(np.mean(scores))
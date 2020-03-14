import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

bankdata = pd.read_csv("C:/Users/USER/Desktop/Thesis Work/Scenario A2/CSV/TimeBasedFeatures_Dataset_15s_NO_VPN.csv")

X = bankdata.drop('class1', axis=1)
y = bankdata['class1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 100)

dtree = DecisionTreeClassifier(criterion='gini',max_depth=None)

scores = []

cv = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):
        dtree.fit(X_train,y_train)
        scores.append(dtree.score(X_test, y_test))
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))
cm=confusion_matrix(y_test,predictions)
print(cm)
print("Accuracy : ")
print(np.mean(scores))
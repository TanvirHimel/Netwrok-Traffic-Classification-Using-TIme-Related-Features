import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split

bankdata = pd.read_csv("C:/Users/USER/Desktop/Thesis Work/Scenario A2/CSV/TimeBasedFeatures_Dataset_15s_NO_VPN.csv")

X = bankdata.drop('class1', axis=1)
y = bankdata['class1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(X_train, y_train)
print("Accuracy: ")
print(model.score(X_test,y_test))
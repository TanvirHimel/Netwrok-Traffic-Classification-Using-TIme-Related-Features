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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state= 100)

from sklearn.svm import LinearSVC

clf = LinearSVC(C=10.0, class_weight=None, dual=True, fit_intercept=True,
                intercept_scaling=1, loss='squared_hinge', max_iter=10000,
                multi_class='ovr', penalty='l2', random_state=100, tol=0.001,
                verbose=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test,predictions))
cm=confusion_matrix(y_test,predictions)
print(cm)
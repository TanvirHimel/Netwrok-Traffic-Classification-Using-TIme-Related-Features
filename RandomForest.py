import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("C:/Users/USER/Desktop/Thesis Work/Scenario A2/CSV/TimeBasedFeatures_Dataset_15s_VPN.csv")

X = data.drop('class1', axis=1)
X = StandardScaler().fit_transform(X)
y = data['class1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

clf = RandomForestClassifier(n_estimators=100)

#create a dictionary of all values we want to test for n_estimators
#params_rf = {"n_estimators": [50, 100, 200]}#use gridsearch to test all values for n_estimators
#rf_gs = GridSearchCV(clf, params_rf, cv=5)#fit model to training data
#rf_gs.fit(X_train, y_train)
#save best model
#rf_best = rf_gs.best_estimator_
#check best n_estimators value
#print(rf_gs.best_params_)

scores = []

cv = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
predictions = clf.predict(X_test)
print(classification_report(y_test,predictions))
cm=confusion_matrix(y_test,predictions)
print(cm)
print("accuracy : ")
print(np.mean(scores))
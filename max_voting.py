import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

bankdata = pd.read_csv("C:/Users/USER/Desktop/Thesis Work/Scenario A2/CSV/TimeBasedFeatures_Dataset_15s_VPN.csv")

X = bankdata.drop('class1', axis=1)
y = bankdata['class1']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

from sklearn.ensemble import VotingClassifier
model1 = KNeighborsClassifier()
model2 = DecisionTreeClassifier(random_state=1)
model3= RandomForestClassifier(n_estimators=100)
model = VotingClassifier(estimators=[('knn', model1), ('dt', model2),('rf',model3)], voting='hard')
model.fit(x_train,y_train)
model.score(x_test,y_test)




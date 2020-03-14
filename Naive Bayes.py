import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/USER/Desktop/Thesis Work/Scenario B/CSV/TimeBasedFeatures_Dataset_15s_AllinOne.csv")
df.head()

test_size=0.3 # Test-set fraction

X = df.drop('class1',axis=1)
y = df['class1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

X_train.shape

X_train.head()

from sklearn.naive_bayes import GaussianNB

nbc = GaussianNB()
scores = []
cv = KFold(n_splits=3, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):
        nbc.fit(X_train, y_train)
        scores.append(nbc.score(X_test, y_test))

y_pred = nbc.predict(X_test)
mislabel = np.sum((y_test!=y_pred))
print("Total number of mislabelled data points from {} test samples is {}".format(len(y_test),mislabel))

from sklearn.metrics import classification_report

print("The classification report is as follows...\n")
print(classification_report(y_pred,y_test))
print("accuracy : ")
print(np.mean(scores))
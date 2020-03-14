import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold

df = pd.read_csv("C:/Users/USER/Desktop/Thesis Work/Scenario B/CSV/TimeBasedFeatures_Dataset_15s_AllinOne.csv")
#df.head()
#df.info()
#df.describe()

scaler = StandardScaler()
scaler.fit(df.drop('class1',axis=1))
scaled_features = scaler.transform(df.drop('class1',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
X = df_feat
y = df['class1']
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['class1'],test_size=0.30, random_state=100)

knn = KNeighborsClassifier(n_neighbors=1)

#create a dictionary of all values we want to test for n_neighbors
#params_knn = {"n_neighbors": np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
#knn_gs = GridSearchCV(knn, params_knn, cv=5)
#fit model to training data
#knn_gs.fit(X_train, y_train)

#save best model
#knn_best = knn_gs.best_estimator_
#check best n_neigbors value
#print(knn_gs.best_params_)


scores = []

cv = KFold(n_splits=5, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
predictions = knn.predict(X_test)
print(classification_report(y_test,predictions))
print("Misclassification error rate:",round(np.mean(predictions!=y_test),3))
print("accuracy : ")
print(np.mean(scores))

error_rate = []

# Will take some time
#for i in range(1, 60):
#    knn = KNeighborsClassifier(n_neighbors=i)
#    knn.fit(X_train, y_train)
#    pred_i = knn.predict(X_test)
#    error_rate.append(np.mean(pred_i != y_test))

#plt.figure(figsize=(10,6))
#plt.plot(range(1,60),error_rate,color='blue', linestyle='dashed', marker='o',
#         markerfacecolor='red', markersize=8)
#plt.title('Error Rate vs. K Value', fontsize=20)
#plt.xlabel('K',fontsize=15)
#plt.ylabel('Error (misclassification) Rate',fontsize=15)
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
import seaborn as sns

dataframe=read_csv("data2.csv")
print(dataframe.shape)
dataframe=dataframe.drop(columns=['id','area_mean','area_worst','perimeter_mean','area_se','perimeter_worst'],axis=1)
dataframe.dropna(inplace=True)

print(dataframe.groupby('diagnosis').size(),"\n")

array=dataframe.values
Y=array[:,0]
dataframe=dataframe.drop(columns='diagnosis',axis=1)
X=dataframe.values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=8)

def accuracy():
    result=model.score(X_test,Y_test)
    print("Accuracy:",result*100.0)


def confusion():
    predicted=model.predict(X_test)
    matrix=confusion_matrix(Y_test,predicted)
    print("Confusion matrix:\n",matrix,"\n")
    
def calculate():
    accuracy()
    confusion()

print("Logistic Regression")
model=LogisticRegression()
model.fit(X_train,Y_train)
calculate()

print("SVC")
model_svm=SVC(kernel='linear',gamma=1)
model_svm.fit(X_train,Y_train)
predicion_svm = model_svm.predict(X_test)
results_svm=metrics.classification_report(y_true=Y_test, y_pred=predicion_svm)
#print(results_svm)
cm_svm=metrics.confusion_matrix(y_true=Y_test, y_pred=predicion_svm)
f,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_svm, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
plt.show()
calculate()

print("Random Forest")
model=RandomForestClassifier(n_estimators=4,max_features=3)
model.fit(X_train,Y_train)
print("Importances:\n",model.feature_importances_)
print("Probability of predicted values:\n",model.predict_proba(X_test))
calculate()

print("Decision Tree")
model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
print("Importances:\n",model.feature_importances_)
print("Probability of predicted values:\n",model.predict_proba(X_test))
calculate()

print("KNN")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train)
calculate()

print("Gaussian_NB")
model=GaussianNB()
model.fit(X_train, Y_train)
calculate()
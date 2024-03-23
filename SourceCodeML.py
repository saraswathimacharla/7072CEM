# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:23:47 2024

@author: User
"""

#importing necessary libraries
import sklearn.neighbors as ne
import sklearn.tree as tr
import sklearn.linear_model as lm

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as ms
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score


#reading the Dataset
herfail = pd.read_csv(
    "D:/Machinelearning7072/heart_failure_clinical_records_dataset.csv")
print(herfail.columns)

#Configuring the test and train data from the dataset
X=herfail.drop("DEATH_EVENT",axis=1)
print(X.columns)
y=herfail['DEATH_EVENT']
print(y)

#Heart failure percentages
explode = (0.025,0.025)
herfail.groupby(['DEATH_EVENT']).sum().plot(
    kind = 'pie', y='sex', autopct='%1.0f%%',explode = explode)
plt.title('Percentages of Heart Failures')
plt.show()


# Train and test data subsets of dataset
X_train, X_test, y_train, y_test = ms.train_test_split(
    X, y, test_size=0.2,random_state=0)
#printing the test and train datsets
print(X_train,"\n")
print(X_test,"\n")
print(y_train,"\n")
print(y_test,"\n")

#Scaling the Dataset
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.fit_transform(X_test))

k_values = [i for i in range (1,31)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(X)

for k in k_values:
    knn = ne.KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=3)
    scores.append(np.mean(score))
  
sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.title('Accuracy scores on different k values')



#------------------------------KNN Algorithm-----------------------
KNN=ne.KNeighborsClassifier(n_neighbors=9)
KNN.fit(X_train,y_train)
KNN.fit(X_test,y_test)
KNN_pred= KNN.predict(X_test)
trACC_KNN=KNN.score(X_train,y_train)
tesACC_KNN=KNN.score(X_test,y_test)

KNN_accuracy = accuracy_score(y_test, KNN_pred)
print("KNN Accuracy:", KNN_accuracy)

#Test and Train Accuracy for KNN
print('Train Accuracy for KNN ',trACC_KNN)
print('Test Accuracy for KNN ', tesACC_KNN,"\n")

#Confusion matrix
KNN_cf = confusion_matrix(y_test,KNN_pred)
print('confusion matrix for KNN:\n',KNN_cf)

#Confusion matrix metrics
KNN_matrix = classification_report(y_test,KNN_pred)
print('Classification Report for KNN algorithm:\n',KNN_matrix)

#plotting Confusion matrix 
KNN_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = KNN_cf,display_labels = [False,True])
KNN_cm_display.plot()
plt.title('Confusion matrix for KNN algorithm')
plt.show()

#---------------------------Decision tree-----------------------
#Create a Decision Tree classifier with the current max_depth = 4
#from sklearn import 

DT=tr.DecisionTreeClassifier(max_depth=7)
DT.fit(X_train,y_train)
DT_pred= DT.predict(X_test)
trACC_DT=DT.score(X_train,y_train)
tesACC_DT=DT.score(X_test,y_test)

print('Train Accuracy for Decision Tree=', trACC_DT)
print('Test Accuracy for Decision Tree=', tesACC_DT)

DT_accuracy = accuracy_score(y_test, DT_pred)
print("DT Accuracy:", DT_accuracy)

DT_model = DT.fit(X_train,y_train)
text_representation = tr.export_text(DT)
print(text_representation)

# To Visualize Decision Tree
plt.figure()
tr.plot_tree(DT_model)
plt.title('Decision Tree')
plt.show()

#Confusion matrix
DT_pred= DT.predict(X_test)
DT_cf = confusion_matrix(y_test,DT_pred)
print('confusion matrix for DT :\n',DT_cf)

#Confusion matrix metrics
DT_matrix = classification_report(y_test,DT_pred)
print('Classification Report for Decision Tree:\n',DT_matrix)

#plotting Confusion matrix 
DT_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = DT_cf,display_labels = [False,True])
DT_cm_display.plot()
plt.title('Confusion matrix for Decision Tree')
plt.show()

#------------------------------Logistic Regression-----------------------
LR=lm.LogisticRegression()
LR.fit(X_train,y_train)
LR_pred= LR.predict(X_test)
trACC_LR=LR.score(X_train,y_train)
tesACC_LR=LR.score(X_test,y_test)
print('Train Accuracy for Logistic Regression : ',trACC_LR)
print('Test Accuracy for Logistic Regression : ', tesACC_LR,"\n")

LR_accuracy = accuracy_score(y_test, LR_pred)
print("LR Accuracy:", LR_accuracy)
#confusion matrix for logistic regression
LR_cf = confusion_matrix(y_test,LR_pred)
print('confusion matrix for LR:\n',LR_cf)

#Confusion matrix metrics
LR_matrix = classification_report(y_test,LR_pred)
print('Classification Report for LR:\n',LR_matrix)

#plotting Confusion matrix 
LR_cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix = LR_cf,display_labels = [False,True])
LR_cm_display.plot()
plt.title('Confusion matrix for Logistic Regression')
plt.show()

#---------------------ROC curve-----------------------
#ROC curve for KNN algorithm
KNN_pred_prob = KNN.predict_proba(X_test)
KNN_pred_prob = KNN_pred_prob[:,1]
KNN_fpr,KNN_tpr,_=roc_curve(y_test,KNN_pred_prob)

# ROC curve for Decision Tree
DT_pred_prob = DT.predict_proba(X_test)
DT_pred_prob = DT_pred_prob[:,1]
DT_fpr,DT_tpr,_=roc_curve(y_test,DT_pred_prob)

# ROC for Logistic regression
LR_pred_prob = LR.predict_proba(X_test)
LR_pred_prob = LR_pred_prob[:,1]
LR_fpr,LR_tpr,_=roc_curve(y_test,LR_pred_prob)

#plotting Roc_curve
plt.figure()
plt.plot(LR_fpr,LR_tpr,label = 'Logistic Regression')
plt.plot(DT_fpr,DT_tpr,label = 'Decision Tree')
plt.plot(KNN_fpr,KNN_tpr,label = 'KNN')
plt.title('ROC Curves of Classification Algorithms')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#plot for the different accuracies
data = [['KNN',KNN_accuracy*100],
        ['DT',DT_accuracy*100],
        ['LR',LR_accuracy*100]]
df = pd.DataFrame(data, columns=['Name','Acr'])
print(df)


df.plot.bar(x='Name',y = 'Acr',width = 0.2)
plt.title('Accuracies of different algorithms')
plt.xlabel('ML Algorithms')
plt.ylabel('Train Accuracy (%)')
plt.show()


precision = [['KNN',100,35,52],['DT',76,57,65],['LR',100,48,65]]
df_pre = pd.DataFrame(precision, columns = ['Name','Precision','recall','f1score'])
df_pre.plot.bar(x='Name',y = 'Precision',width = 0.3)
plt.title('Precision of different ML algorithms')
plt.xlabel('ML Algorithms')
plt.ylabel('Precision (%)')
plt.show()

df_pre.plot.bar(x='Name',y = 'recall',width = 0.3)
plt.title('Recall scores of different ML algorithms')
plt.xlabel('ML Algorithms')
plt.ylabel('Recall (%)')
plt.show()

df_pre.plot.bar(x='Name',y = 'f1score',width = 0.3)
plt.title('f1score of different ML algorithms')
plt.xlabel('ML Algorithms')
plt.ylabel('f1score (%)')
plt.show()





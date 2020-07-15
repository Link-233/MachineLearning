import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC  
import numpy as np

data=pd.read_csv('voice.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
y=LabelEncoder().fit_transform(y)
imp=SimpleImputer(missing_values=0,strategy ='mean')
x=imp.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

scaler1 = StandardScaler()
scaler1.fit(x_train)
x_train = scaler1.transform(x_train)
x_test = scaler1.transform(x_test)

knn=KNeighborsClassifier() 
knn.fit(x_train,y_train) 
y_train_result=knn.predict(x_train)
y_pred=knn.predict(x_test)
print('knn总正确率:%.3f%%'%(round(knn.score(x_test,y_test)*100.0,2)))
print(classification_report(y_test,y_pred))

rf=RandomForestClassifier(n_estimators=10,criterion="gini") 
rf.fit(x_train,y_train)
y_train_result=rf.predict(x_train)
y_pred=rf.predict(x_test)
print('随机森岭总正确率:%.3f%%'%(round(rf.score(x_test,y_test)*100.0,2)))
print(classification_report(y_test,y_pred))

svc=SVC(C=1,kernel='rbf',probability=True) 
svc.fit(x_train,y_train) 
y_train_result=svc.predict(x_train)
y_pred=svc.predict(x_test)
print('SVM总正确率:%.3f%%'%(round(svc.score(x_test,y_test)*100.0,2)))
print(classification_report(y_test,y_pred))

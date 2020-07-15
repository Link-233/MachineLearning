import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
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

mnb=GaussianNB()
mnb.fit(x_train,y_train)
y_predict=mnb.predict(x_test)
print('总正确率:%.2f%%'%(round(mnb.score(x_test,y_test)*100.0,2)))
print(classification_report(y_test,y_predict))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR

import warnings
warnings.filterwarnings('ignore')

data= pd.read_csv('Real_satePrice.csv')
data.drop('Unnamed: 0',axis=1,inplace=True)
y=data['log_SalePrice']
x= data.drop(['log_SalePrice'],axis=1)
print(x.shape)

rb_scaler=RobustScaler()
x_rb=pd.DataFrame(rb_scaler.fit_transform(x),columns=x.columns)
x_train,x_test,y_train,y_test=train_test_split(x_rb,y,test_size=0.25,random_state=42)

L_SVR=LinearSVR(C=10,random_state=198,max_iter=10000)
L_SVR.fit(x_train,y_train)

y_predict=np.exp(L_SVR.predict(x_test))

import pickle

pickle.dump(L_SVR,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

print(y_predict)
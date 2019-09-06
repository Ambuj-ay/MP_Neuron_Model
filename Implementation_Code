# Loading_Data
"""

import sklearn.datasets
import numpy as np

breast_cancer=sklearn.datasets.load_breast_cancer()

X=breast_cancer.data
Y=breast_cancer.target

print(X.shape,Y.shape)

import pandas as pd

data=pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

data['class']=breast_cancer.target

data.head()

data.describe()

print(data['class'].value_counts())

print(breast_cancer.target_names)

"""# Training_Test_Split"""

from sklearn.model_selection import train_test_split

X=data.drop('class', axis=1)

Y=data['class']

type(X)

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.1)

print(X.shape,X_train.shape,X_test.shape)

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.1, stratify = Y)

print(Y.mean(),Y_train.mean(),Y_test.mean())

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.1, stratify = Y , random_state=1)

"""# Binarisation"""

import matplotlib.pyplot as plt

plt.plot(X_train.T,'*')
plt.xticks(rotation='vertical')
plt.show()

X_binarised_3_train=X_train['mean area'].map(lambda x: 0 if x <1000 else 1)

plt.plot(X_binarised_3_train,'*')

X_binarised_train=X_train.apply(pd.cut, bins=2, labels=[1,0])

plt.plot(X_binarised_train.T,'*')
plt.xticks(rotation='vertical')
plt.show()

X_binarised_test=X_test.apply(pd.cut, bins=2, labels=[1,0])

X_binarised_train=X_binarised_train.values
X_binarised_test=X_binarised_test.values

type(X_binarised_test)

"""# MP_Neuron_Model"""

from random import randint

b = 3


i = randint(0, X_binarised_train.shape[0])

print('For row',i)

if(np.sum(X_binarised_train[100, :]) >= b):
   print('MP neuron inference is malignent')
else:
   print('MP neuron inference is benign')

if(Y_train[i]==1):
  print("ground truth is malignent")
else:
  print("Ground truth is benign")

b=3

Y_pred_train=[]

accurate_rows=0

for x,y in zip(X_binarised_train,Y_train):
  y_pred=np.sum(x) >=b
  Y_pred_train.append(y_pred)
  accurate_rows+= (y== y_pred)

print(accuracy, accurate_rows/X_binarised_train.shape[0])

for b in range(X_binarised_train.shape[1] + 1):
  
  Y_pred_train=[]
  accurate_rows=0

  for x,y in zip(X_binarised_train,Y_train):
    y_pred= (np.sum(x) >=b)
    Y_pred_train.append(y_pred)
    accurate_rows+= (y== y_pred)
  print(b, accurate_rows/X_binarised_train.shape[0])

from sklearn.metrics import accuracy_score
b=28

Y_pred_test=[]


for x in X_binarised_test:
  y_pred= (np.sum(x) >=b)
  Y_pred_test.append(y_pred)
  

accuracy=accuracy_score(Y_pred_test, Y_test)

print(b, accuracy)

"""# MP_Neuron_Class"""

class MPNeuron:
  def __init__(self):
    self.b=None
  
  def model(self,x):
    return(sum(x) >= self.b)
  
  def predict(self,X):
    Y=[]
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
  
  def fit(self, X, Y):
    accuracy={}
    
    for b in range(X.shape[1]+1):
      self.b=b
      Y_pred=self.predict(X)
      accuracy[b]=accuracy_score(Y_pred,Y)
      
    best_b=max(accuracy, key=accuracy.get)
    self.b=best_b
    
    print('Optimal value of b is',best_b)
    print('Highest accuracy is',accuracy[best_b])

mp_neuron=MPNeuron()
mp_neuron.fit(X_binarised_train, Y_train)

Y_test_pred=mp_neuron.predict(X_binarised_test)
accuracy_test=accuracy_score(Y_test_pred,Y_test)

print(accuracy_test)



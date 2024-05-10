import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


test_size = 0.8
df = pd.read_csv("FuelConsumption.csv")
df = df[["ENGINESIZE", "CO2EMISSIONS"]]


X = np.array(df["ENGINESIZE"])
Y = np.array(df["CO2EMISSIONS"])
random.shuffle(X)
random.shuffle(Y)


#Optimization data
xavg = sum(X)/len(X)
yavg = sum(Y)/len(Y)
length0 = len(X)

for i in range(length0):
    if X[i] > (1.6 * xavg) or (Y[i] > 1.6 * yavg):
       X[i] = 0
       Y[i] = 0

X = X[X != 0]
Y = Y[Y != 0]



x_data_length = len(X)
y_data_length = len(Y)
x_train = X[:int(x_data_length*test_size)]
x_test = X[int(x_data_length * test_size) :x_data_length ]
y_train = Y[:int(y_data_length * test_size) ]
y_test = Y[int(y_data_length * test_size) : y_data_length ]

x_train_avg = sum(x_train)/len(x_train)
y_train_avg = sum(y_train)/len(y_train)



def coef_intercept(xtrain, ytrain):
   s =0
   length = len(xtrain)
   for i in range(length):
       s = s + (xtrain[i] - x_train_avg) * (ytrain[i] - y_train_avg)
   j = 0
   for i in range(length):
       j = j + (xtrain[i] - x_train_avg) *(xtrain[i] - x_train_avg)
   _coef = s/j
   _intercept = y_train_avg - _coef * x_train_avg

   return _coef , _intercept

cof, intercept = coef_intercept(x_train, y_train)



def _predict(xtest):
    _predictList = []
    for i in xtest:
        _predictList.append(cof*i + intercept)
    return _predictList

predicted = _predict(x_test)
predicted = np.array(predicted)

def _predict(predict):
    for i in range(len(predicted)):
        print(f"Real: {y_test[i]}, Predicted: {predicted[i]}")
        print("*"*36)


def accurancy(predict, ytest):
    acc = 0
    length = len(predict)
    for i in range(length):
        acc = acc + abs((predict[i] - ytest[i])/predict[i])
    acc = acc /(length)

    return (1 - acc)*100

accur = accurancy(predicted, y_test)
print(accur)

plt.scatter(x_train, y_train)
plt.show()






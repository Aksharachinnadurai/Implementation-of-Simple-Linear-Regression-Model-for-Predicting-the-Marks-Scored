# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: AKSHARA C
RegisterNumber:  212223220004
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()
X = df.iloc[:,:-1].values
X
Y = df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="yellow")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE= ",mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
## df.head()

![image](https://github.com/user-attachments/assets/1e63dca9-f1cb-4273-a907-a63850daed72)

## df.tail()

![image](https://github.com/user-attachments/assets/0c9226a8-7c2c-47d6-9dea-f23cb5d52128)

## Array value of X

![image](https://github.com/user-attachments/assets/8ca3e12b-f4ec-4b26-9b81-3c11024c46ed)

## Array value of Y

![image](https://github.com/user-attachments/assets/d1e906ca-865c-4bad-bb1b-7cbb557c687c)

## Values of Y prediction

![image](https://github.com/user-attachments/assets/87a75384-f7b9-4c7d-98c2-3fb54de8804c)

## Array values of Y test

![image](https://github.com/user-attachments/assets/a8c24042-2aaa-43e0-a8b2-061c1f2aa2d0)

## Training set graph

![image](https://github.com/user-attachments/assets/04c3d140-1b6e-4a05-a020-64eb28731da6)

## Test set graph

![image](https://github.com/user-attachments/assets/74cbe812-7fc7-4612-ab93-f24c92f6d21b)

## Values of MSE,MAE and RMSE

![image](https://github.com/user-attachments/assets/ea8ae93b-3de4-4ce4-9657-d9d7c5d9555c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

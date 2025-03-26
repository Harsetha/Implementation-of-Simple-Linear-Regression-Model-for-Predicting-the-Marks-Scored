# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Preprocess Data: Read the CSV file containing student scores, extract the independent variable (X) and dependent variable (Y), and split the data into training and testing sets.

2. Train Model: Use the training data to train a linear regression model using LinearRegression() from scikit-learn.

3. Predict Test Set Results: Predict the target variable Y for the test set using the trained model.

4. Visualize Results: Plot the training data with the regression line, and similarly plot the test data with the predicted line.

5. Evaluate Model: Calculate and display performance metrics (Mean Squared Error, Mean Absolute Error, and Root Mean Squared Error).

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HARSETHA J
RegisterNumber: 212223220032
*/
```
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by:  Roshini S
RegisterNumber: 212223230174


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
 
Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output:
![simple linear regression model for predicting the marks scored](sam.png)

TRAINING SET:
![363060378-2e450af3-e206-4b55-b7e4-ca87b65a9a68](https://github.com/user-attachments/assets/fbce205b-5a5c-4638-9b50-4979bc7a3075)

TEST SET:
![363060495-2c896797-52df-4b23-8780-47f25bd58f43](https://github.com/user-attachments/assets/45826fa6-303c-46d7-922e-9f28dede0868)

MEAN SQUARE ERROR, MEAN ABSOLUTE ERROR AND RMSE:
![363103157-1f07520b-d4c7-4681-a04c-d6133f20952e](https://github.com/user-attachments/assets/b872a654-587e-4ade-8fd2-5c0f7d77d812)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

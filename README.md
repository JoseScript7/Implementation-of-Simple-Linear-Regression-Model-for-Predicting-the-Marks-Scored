# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import all the required libraries
Start by importing the standard libraries needed for data handling, plotting graphs, and building the model.

2.Assign values from the dataset to variables
Load the dataset and assign values to the variables for hours studied and marks scored.

3.Bring in the Linear Regression model from sklearn
Use the LinearRegression module from sklearn to create the regression model.

4.Plot the data points on a graph
Represent the values (hours and marks) on a scatter plot to visualize the relationship.

5.Draw the regression line based on the predicted values
Use the model to predict marks and show the regression line that fits the plotted data.

6.Compare actual and predicted values using the graph
Match the predicted line with actual points to see how well the model works and confirm that linear regression has been achieved for the dataset.

## Program / Output:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: A.Ranen Joseph Solomon
RegisterNumber: 212224040269
*/
```
LINEAR REGRESSION MODEL 

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    df=pd.read_csv("scores.csv")
    df
![image](https://github.com/user-attachments/assets/2eedde1d-1b33-4e1a-a956-7306f522d99f)

    df.head()
![image](https://github.com/user-attachments/assets/dab29f3c-7ab7-40eb-911f-9b110ea6ff4a)

    df.tail()
![image](https://github.com/user-attachments/assets/3b6c173f-7707-4895-8e9d-ad76478ece81)

    x = df.iloc[:, :-1].values
    print(x)
![image](https://github.com/user-attachments/assets/f8b77706-11dc-40b4-a4f0-5d7435b149f5)

    y = df.iloc[:, 1].values  
    printf(y)
![image](https://github.com/user-attachments/assets/ec2a94b9-cb23-493e-afc1-2907d61b5307)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)
    y_pred
![image](https://github.com/user-attachments/assets/e2f1b0a3-2367-48fe-bacd-71208d0fe4e8)

    y_test
![image](https://github.com/user-attachments/assets/0353407d-8c51-4fec-a37f-3c8f9d8bd66b)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)
![image](https://github.com/user-attachments/assets/1021f735-83b5-40c0-942b-7c841e62b97d)

    plt.scatter(x_train, y_train, color='black')
    plt.plot(x_train, regressor.predict(x_train), color='blue')
    plt.title("Hours vs Scores (Training Set)")
    plt.xlabel("Hours")
    plt.ylabel("Scores")
    plt.show()
![image](https://github.com/user-attachments/assets/b64faa39-6ed4-49c9-ba10-067621914e9b)

    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_train, regressor.predict(x_train), color='red')  # Using same regression line
    plt.title("Hours vs Scores (Testing Set)")
    plt.xlabel("Hours")
    plt.ylabel("Scores")
    plt.show()
![image](https://github.com/user-attachments/assets/31a0909d-3f38-4d6b-9ea7-2c39108c7066)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

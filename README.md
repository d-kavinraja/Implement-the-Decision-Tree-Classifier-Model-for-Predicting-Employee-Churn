# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
## Equipments Required:
1. Hardware – PCs.
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.    
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.
## Program:
```py
 # Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
 # Developed by: Kavinraja D
 # RegisterNumber: 212222240047
import pandas as pd
data=pd.read_csv("dataset/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evalution","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### Data Head
![](./images/head.png)
### Information:
![](./images/info.png)
### Null dataset:
![](./images/null.png)
### Value_counys:
![](./images/v_count.png)
### Data Type Convertion:
![](./images/change%20to%20number.png)
### Data Info:
![](./images/data%20info.png)
### Accuracy:
![](./images/accu.png)
### Data Prediction:
![](./images/pred.png)
## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

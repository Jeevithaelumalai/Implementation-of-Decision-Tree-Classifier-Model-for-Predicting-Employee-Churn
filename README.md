# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas and read the csv file.
2. Import Decision tree classifier.
3. Fit the data in the model
4. Find the accuracy score.

## Program:
```
Developed by: Jeevitha E
RegisterNumber: 212222230054  

```
```
import pandas as pd
df=pd.read_csv("CSVs/Employee.csv")
df.head()
df.info()
df.isnull().sum()
df['left'].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df['salary'])
df.head()
x=df[['satisfaction_level','last_evaluation','number_project','average_montly_hours',
      'time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
y=df['left']
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(Xtrain,Ytrain)
Ypred=dt.predict(Xtest)
from sklearn import metrics
accuracy=metrics.accuracy_score(Ytest,Ypred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
df.head()
![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708245/ade6275d-96d9-433d-bc1d-44bf5e49da84)
![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708245/049d38b9-eb7e-43df-9b35-299c1c3a07a8)
![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708245/35f01762-79a1-42b5-9d18-d8b5530ef750)
![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708245/26756af5-eb72-4ca9-9b13-691b2d135b22)
![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708245/3f7de6c7-d297-4280-8a45-a5d3dd0b0371)
![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708245/8bcf3055-708a-4f79-a65e-c2b3796fd2d5)
![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708245/17d2efce-e2a3-4d30-9424-e8bbe031f2bd)

![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118708245/12a01398-880d-4d44-b491-20eb3824cd01)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

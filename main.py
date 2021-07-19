import pandas as pd
import numpy as np
import pickle

# Load dataset
data=pd.read_csv("Salary_Prediction_Data.csv")

#split the data set

x = data.drop('income', axis=1)
y = data['income']


# perform train test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# model creation

from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier()
m=model.fit(x_train, y_train)



pickle.dump(model,open('model.pkl','wb'))



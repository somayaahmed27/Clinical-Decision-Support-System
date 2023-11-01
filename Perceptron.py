import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

data = pd.read_csv(
    "D:/unicourses/machine learning/project/Clinical Decision Support System/data.csv")

# print(data.shape)
# print(data.head())
x = data.drop(['prognosis'], axis=1)
y = data['prognosis']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=324)
classifier = Perceptron(max_iter=40, random_state=0)
classifier.fit(x_train, y_train.values.ravel())

y_predictied = classifier.predict(x_test)
acc = accuracy_score(y_test, y_predictied)*100
print(acc)

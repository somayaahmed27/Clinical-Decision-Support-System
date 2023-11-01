from turtle import pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv("D:/unicourses/machine learning/project/Clinical Decision Support System/data.csv")
print(data.shape)
print(data.head())

x = np.array(data.drop(['prognosis'], axis=1))
y = np.array(data['prognosis'])

x_train, x_text, y_train, y_text = train_test_split(x, y, test_size=0.4)


diagnosis_classifier = KNeighborsClassifier(n_neighbors=5)
diagnosis_classifier.fit(x_train, y_train)
diagnosis_predict = diagnosis_classifier.predict(x_text)
acc = accuracy_score(diagnosis_predict, y_text)
print(acc*100)

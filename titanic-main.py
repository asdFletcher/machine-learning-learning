import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


titanic_data = pd.read_csv("./titanic-data/datasets_1275_2286_train_and_test2.csv")

# remove zeros
for column in titanic_data.columns:
    if "zero" in column:
        del titanic_data[column]


print(titanic_data.head(5))

print(f"number of passengers: {len(titanic_data.index)}")

# sns.countplot(x="Survived", data=titanic_data)
# plt.show()

# sns.countplot(x="Survived", hue="Sex", data=titanic_data)
# plt.show()

# sns.countplot(x="Survived", hue="Pclass", data=titanic_data)

# titanic_data["Age"].plot.hist()
# plt.show()

# plt.hist(bins=40, data=titanic_data, x="Fare")
# plt.show()

# titanic_data.info()
# sns.countplot(x="SibSp", data=titanic_data)
# plt.show()

# titanic_data.isnull().sum()

# sns.countplot(x="Pclass", data=titanic_data)
# sns.boxplot(x="Pclass", y="Age", data=titanic_data)
# plt.show()

# remove na values
titanic_data["Embarked"].fillna(2, inplace=True)
# titanic_data.drop(columns="Embarked", axis=1, inplace=True)

titanic_data.head()
titanic_data.rename(columns={"Sex": "Male"}, inplace=True)
# sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
# print(sex.head(5))
print("1")
print(titanic_data.head(5))

# sanitize embarked data P, Q, S
embarked_columns = pd.get_dummies(titanic_data["Embarked"], drop_first=True)
embarked_columns.rename(columns={0: "P", 1: "Q", 2: "S"}, inplace=True)
print(f"new embarked columns:")
print(embarked_columns.head(5))

# sanitize class data, 1st 2nd 3rd, 0 1 2
class_columns = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
class_columns.rename(columns={1: "First", 2: "Second", 3: "Third"}, inplace=True)
print(f"new class columns:")
print(class_columns.head(5))

# concat
titanic_data = pd.concat([titanic_data, embarked_columns, class_columns], axis=1)
# remove old
titanic_data.drop(columns=["Pclass", "Embarked", "Passengerid"], inplace=True)
print(f"new data set:")
print(titanic_data.head(5))

# training
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


x = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter=10000)
logmodel.fit(x_train, y_train)
print("3")
predictions = logmodel.predict(x_test)
print(predictions)

print("4")
print(classification_report(y_test, predictions))

print("5")
print(confusion_matrix(y_test, predictions))

print("6")
print(accuracy_score(y_test, predictions))

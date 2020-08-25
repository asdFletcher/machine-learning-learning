import pandas
from pandas.plotting import scatter_matrix
import scipy
import matplotlib.pyplot as plt
import sys
import sklearn

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class",
]

dataset = pandas.read_csv("iris-data/iris.data", names=names)

# print(dataset.shape)
# print(dataset.head(20))
# print(dataset.describe())
print(dataset.groupby("class").size())

dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
# plt.show()

dataset.hist()
# plt.show()

scatter_matrix(dataset)
# plt.show()


all_values = dataset.values

x = all_values[:, 0:4]
y = all_values[:, 4]

validation_size = 0.2
seed = 6
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=validation_size, random_state=seed
)

scoring = "accuracy"

models = []
models.append(("LR", LogisticRegression()))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC()))

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, x_train, y_train, cv=kfold, scoring=scoring
    )
    results.append(cv_results)
    names.append(name)
    msg = f"{name}: {cv_results.mean()} ({cv_results.std()})"
    print(msg)

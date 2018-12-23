import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv(
    '/Users/markfrost/learn/course_machine_learning/week_1/titanic/data/train.csv', index_col='PassengerId')[['Survived', 'Pclass', 'Fare', 'Age', 'Sex']]


data = data.dropna()
data['Sex'] = data['Sex'].apply(lambda x: [0, 1][x == 'male'])
print(data)
array = data.values
print(array)
X = array[:, 1:5]
y = array[:, 0]
print(y)
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

print(clf)

print(clf.feature_importances_)

# X = np.array([[1, 2], [3, 4], [5, 6], [4, 2]])
# y = np.array([0, 1, 0, 2])
# clf = DecisionTreeClassifier()
# clf.fit(X, y)


# print(clf.feature_importances_)

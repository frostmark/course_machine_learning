import pandas

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

data = pandas.read_csv(
    '/Users/markfrost/learn/course_machine_learning/week_2/data/data.csv')

array = data.values

X = array[:, 1:14]
y = array[:, 0]

X_scaled = scale(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

svc = svm.SVC(C=1, kernel='linear')
result = []
for n in range(2, 51):
    neigh = KNeighborsClassifier(n_neighbors=n)

    neigh.fit(X_scaled, y)
    scores = []

    mean = np.mean(cross_val_score(
        neigh, X_scaled, y, cv=kf, scoring='accuracy'))
    result.append(mean)
    print(n, mean)

#     for train_index, test_index in kf.split(X_scaled):
#         X_train, X_test = X_scaled[train_index], X_scaled[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         result.append(neigh.score(X_train, y_train))
#         scores.append(neigh.score(X_train, y_train))

#     # print(scores)
#     print(n, "Accuracy: %0.2f (+/- %0.2f)" %
#           (np.array(scores).mean(), np.array(scores).std() * 2), max(scores))

print(max(result))

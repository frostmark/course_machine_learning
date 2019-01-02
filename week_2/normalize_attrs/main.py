import pandas

import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def target_and_data(data):
    return data[:, 1:3], data[:, 0]


train_data = pandas.read_csv(
    '/Users/markfrost/learn/course_machine_learning/week_2/normalize_attrs/data/perceptron-test.csv', header=None).values

test_data = pandas.read_csv(
    '/Users/markfrost/learn/course_machine_learning/week_2/normalize_attrs/data/perceptron-test.csv', header=None).values

scaler = StandardScaler()
acc = []

for n in range(1, 1000):
    perceptron_no_scaled = Perceptron(random_state=241, max_iter=n, tol=None)

    X, y = target_and_data(train_data)
    X_test, y_test = target_and_data(test_data)

    perceptron_no_scaled.fit(X, y)
    accuracy_before_normalize = accuracy_score(
        y_test, perceptron_no_scaled.predict(X_test))

    print('accuracy score without normalize: ', accuracy_before_normalize)

    perceptron_with_scaled = Perceptron(
        random_state=241, max_iter=n, tol=None)

    X_train_scaled = scaler.fit_transform(X)

    X_test_scaled = scaler.transform(X_test)

    perceptron_with_scaled.fit(X_train_scaled, y)

    accuracy_after_normalize = accuracy_score(
        y_test, perceptron_with_scaled.predict(X_test_scaled))
    print('accuracy score with normalize: ', accuracy_after_normalize)

    print((accuracy_after_normalize - accuracy_before_normalize).round(decimals=3))
    acc.append((accuracy_after_normalize -
                accuracy_before_normalize).round(decimals=3))

print(acc)
print(max(acc))

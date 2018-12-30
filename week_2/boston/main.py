import pandas

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np

boston = datasets.load_boston()

scaled_data = scale(boston['data'])
target = boston['target']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# for p in np.linspace(1, 10, num=200):
#     print(p)
#     model = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
#     model.fit(scaled_data, target)

#     print(np.mean(cross_val_score(model, scaled_data,
#                                   target, cv=kf, scoring='neg_mean_squared_error')))

model = KNeighborsRegressor(n_neighbors=5, weights='distance', p=1)
model.fit(scaled_data, target)

cost = model.predict([[
    -0.41978194, 0.28482986, 1.2879095, 0.27259857, 0.14421743, 0.41367189,
    0.12001342, 0.1402136, 0.98284286, 0.66660821, 1.45900038, 0.44105193, 1.0755623
]])

print(cost)

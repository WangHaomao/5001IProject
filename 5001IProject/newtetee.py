from dataEngineer import get_EngineeriedData
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
import time
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.utils.testing import (assert_raises, assert_greater, assert_equal,
                                   assert_false)
import warnings

from sklearn.neural_network import MLPRegressor
if __name__ == '__main__':

    train_X,train_y,test_X = get_EngineeriedData()
    # print(train_X)
    # train_X, val_X, train_y, val_y = train_test_split(train_X, train_y,random_state=1)

    universal_model = MLPRegressor(hidden_layer_sizes=(5,),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01)
    universal_model.fit(train_X,train_y)
    predicted_y = universal_model.predict(test_X)
    print(predicted_y)
    np.savetxt('data/testResult1101.csv', predicted_y, delimiter=",", fmt="%.2f")
    # print(mean_squared_error(val_y, predicted_y))
    # for momentum in [0, .9]:
    #     mlp = MLPRegressor(max_iter=100, activation='relu',
    #                        random_state=1, learning_rate_init=0.01,
    #                        batch_size=train_X.shape[0], momentum=momentum)
    #     with warnings.catch_warnings(record=True):
    #         # catch convergence warning
    #         mlp.fit(train_X, train_y)
    #     pred1 = mlp.predict(train_X)
    #     mlp = MLPRegressor(activation='relu',
    #                        learning_rate_init=0.01, random_state=1,
    #                        batch_size=train_X.shape[0], momentum=momentum)
    #     for i in range(100):
    #         mlp.partial_fit(train_X, train_y)
    #
    #     pred2 = mlp.predict(train_X)
    #     assert_almost_equal(pred1, pred2,decimal=2)
    #     score = mlp.score(train_X, train_y)
    #     assert_greater(score, 0.75)
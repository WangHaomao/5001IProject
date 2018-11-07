from dataEngineer import get_EngineeriedData
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from dataEngineer import get_data
from sklearn.neural_network import MLPRegressor
if __name__ == '__main__':

    train_data,test_data = get_data()
    train_y = train_data['time']
    train_data.drop('time',axis = 1, inplace = True)
    train_X = train_data
    # print(train_X)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y,random_state=1)

    universal_model = MLPRegressor(hidden_layer_sizes=(5,),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01)
    universal_model.fit(train_X,train_y)
    predicted_y = universal_model.predict(val_X)
    print("error:", mean_squared_error(predicted_y, val_y))
    # print()


    predicted_y = np.power(np.e,predicted_y)
    print(predicted_y)
    # np.savetxt('data/testResult1107.csv', predicted_y, delimiter=",", fmt="%.2f")
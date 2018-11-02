# from __future__import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split

print(tf.__version__)
train_data = pd.read_csv('data/train.csv')
# print(train_data.columns)
all_features = ['id', 'penalty', 'l1_ratio', 'alpha', 'max_iter', 'random_state',
       'n_jobs', 'n_samples', 'n_features', 'n_classes',
       'n_clusters_per_class', 'n_informative', 'flip_y', 'scale', 'time']
train_labels = train_data['time']
train_features = train_data[['penalty', 'l1_ratio', 'alpha', 'max_iter', 'random_state',
       'n_jobs', 'n_samples', 'n_features', 'n_classes',
       'n_clusters_per_class', 'n_informative', 'flip_y', 'scale']]

train_continues_features = train_data[['l1_ratio', 'alpha', 'max_iter', 'random_state',
       'n_jobs', 'n_samples', 'n_features', 'n_classes',
       'n_clusters_per_class', 'n_informative', 'flip_y', 'scale']]

train_categorical_features = train_data[['penalty']]


mean = train_continues_features.mean(axis=0)
std = train_continues_features.std(axis=0)

train_continues_features = (train_continues_features - mean) / std
train_categorical_features = pd.get_dummies(train_categorical_features)

train_data = train_continues_features
train_data[train_categorical_features.columns] = train_categorical_features

# test_data = (test_data - mean) / std
# print(train_data)
# print(train_labels)


from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y = train_test_split(train_data,train_labels,random_state=1)


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(train_X.shape[1],)),
        # keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        # keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    # keras.layers.Dense(64, activation=tf.nn.relu),
    # keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  # optimizer = tf.train.RMSPropOptimizer(0.001)
  #
  # model.compile(loss='mse',
  #               optimizer=optimizer,
  #               metrics=['mae'])
  # return model
model = build_model()
print(model.summary())

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 500

# history = model.fit(train_data, train_labels, epochs=EPOCHS,
#                     validation_split=0.2, verbose=0,callbacks=[PrintDot()])
history = model.fit(train_X, train_y, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,callbacks=[PrintDot()])
print("\nfinish training")
test_predictions = model.predict(val_X)

print("error:",mean_squared_error(test_predictions,val_y))



# test_data = pd.read_csv('data/test.csv')
#
# test_continues_features = test_data[['l1_ratio', 'alpha', 'max_iter', 'random_state',
#        'n_jobs', 'n_samples', 'n_features', 'n_classes',
#        'n_clusters_per_class', 'n_informative', 'flip_y', 'scale']]
# test_continues_features = (test_continues_features - mean) / std
# test_categorical_features = pd.get_dummies(test_data[['penalty']])
#
# test_data = test_continues_features
# test_data[test_categorical_features.columns] = test_categorical_features
#
# # print(test_data)
# # test_predictions = model.predict(test_data).flatten()
# test_predictions = model.predict(test_data)
#
# print(test_predictions)
from sklearn.metrics import mean_absolute_error,mean_squared_error
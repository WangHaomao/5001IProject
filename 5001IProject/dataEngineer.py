
import tensorflow as tf
import itertools
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.ensemble import IsolationForest
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
def read_trainData():
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

    return train_data,train_labels

def get_EngineeriedData():
    train = pd.read_csv('data/train.csv')

    train.drop('id', axis=1, inplace=True)
    train_numerical = train.select_dtypes(exclude=['object'])
    train_numerical.fillna(0, inplace=True)
    train_categoric = train.select_dtypes(include=['object'])
    train_categoric.fillna('NONE', inplace=True)
    train = train_numerical.merge(train_categoric, left_index=True, right_index=True)

    test = pd.read_csv('data/test.csv')
    ID = test.id
    test.drop('id', axis=1, inplace=True)
    test_numerical = test.select_dtypes(exclude=['object'])
    test_numerical.fillna(0, inplace=True)
    test_categoric = test.select_dtypes(include=['object'])
    test_categoric.fillna('NONE', inplace=True)
    test = test_numerical.merge(test_categoric, left_index=True, right_index=True)

    # outlines
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(max_samples=100, random_state=42)
    clf.fit(train_numerical)
    y_noano = clf.predict(train_numerical)
    y_noano = pd.DataFrame(y_noano, columns=['Top'])
    # print(y_noano[y_noano['Top'] == 1].index.values)
    #
    train_numerical = train_numerical.iloc[y_noano[y_noano['Top'] == 1].index.values]
    train_numerical.reset_index(drop=True, inplace=True)
    #
    train_categoric = train_categoric.iloc[y_noano[y_noano['Top'] == 1].index.values]
    train_categoric.reset_index(drop=True, inplace=True)
    #
    train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
    train.reset_index(drop=True, inplace=True)

    col_train_num = list(train_numerical.columns)
    col_train_num_bis = list(train_numerical.columns)
    #
    col_train_cat = list(train_categoric.columns)

    col_train_num_bis.remove('time')

    mat_train = np.matrix(train_numerical)
    mat_test = np.matrix(test_numerical)
    mat_new = np.matrix(train_numerical.drop('time', axis=1))
    mat_y = np.array(train.time)

    # print(mat_train)
    #
    prepro_y = MinMaxScaler()
    prepro_y.fit(mat_y.reshape(360, 1))
    #
    prepro = MinMaxScaler()
    prepro.fit(mat_train)
    #
    prepro_test = MinMaxScaler()
    prepro_test.fit(mat_new)

    # train_num_scale = pd.DataFrame(prepro.transform(mat_train),columns = col_train)
    # test_num_scale  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)
    train_num_scale = pd.DataFrame(prepro.transform(mat_train), columns=col_train_num)
    test_num_scale = pd.DataFrame(prepro_test.transform(mat_test), columns=col_train_num_bis)
    #
    train[col_train_num] = pd.DataFrame(prepro.transform(mat_train), columns=col_train_num)
    test[col_train_num_bis] = test_num_scale
    #
    # List of features
    COLUMNS = col_train_num
    FEATURES = col_train_num_bis
    LABEL = "time"
    #
    FEATURES_CAT = col_train_cat
    #
    engineered_features = []
    #
    for continuous_feature in FEATURES:
        engineered_features.append(
            tf.contrib.layers.real_valued_column(continuous_feature))

    for categorical_feature in FEATURES_CAT:
        sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
            categorical_feature, hash_bucket_size=1000)

        engineered_features.append(
            tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16, combiner="sum"))

    # Training set and Prediction set with the features to predict
    training_set = train[FEATURES + FEATURES_CAT]
    prediction_set = train.time
    #
    # Train and Test
    # x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES + FEATURES_CAT],
    #                                                     prediction_set, test_size=0.33, random_state=42)
    # y_train = pd.DataFrame(y_train, columns=[LABEL])
    #
    # training_set = pd.DataFrame(x_train, columns=FEATURES + FEATURES_CAT).merge(y_train, left_index=True, right_index=True)
    train_X = training_set[FEATURES + FEATURES_CAT]
    train_y = pd.DataFrame(prediction_set, columns=[LABEL])
    testing_sub = test[FEATURES + FEATURES_CAT]

    return pd.get_dummies(train_X),train_y,testing_sub

# get_EngineeriedData()
def get_data():
    # a = ['penalty', 'l1_ratio', 'alpha', 'max_iter', 'random_state', 'n_jobs',
    #  'n_samples', 'n_features', 'n_classes', 'n_clusters_per_class',
    #  'n_informative', 'flip_y', 'scale', 'time']

    selected_features_without_label = ['penalty','time', 'n_samples', 'max_iter',
                         'n_features', 'n_classes', 'flip_y', 'n_informative',
                         'n_jobs','n_clusters_per_class']

    selected_features_with_label = selected_features_without_label.copy()
    selected_features_without_label.remove('time')

    df_train = pd.read_csv('data/train.csv')
    df_train = df_train[selected_features_with_label]


    train_numerical = df_train.select_dtypes(exclude=['object'])
    columns_numerical_no_label = list(train_numerical.columns)
    columns_numerical_no_label.remove('time')
    train_numerical.fillna(0, inplace=True)
    train_categoric = df_train.select_dtypes(include=['object'])
    train_categoric.fillna('NONE', inplace=True)

    clf = IsolationForest(max_samples=100, random_state=42)
    clf.fit(pd.get_dummies(train_numerical))
    y_noano = clf.predict(train_numerical)
    y_noano = pd.DataFrame(y_noano, columns=['Top'])
    # print(y_noano[y_noano['Top'] == 1].index.values)

    train_numerical = train_numerical.iloc[y_noano[y_noano['Top'] == 1].index.values]
    train_numerical.reset_index(drop=True, inplace=True)
    print(len(train_numerical))
    train_categoric = train_categoric.iloc[y_noano[y_noano['Top'] == 1].index.values]
    train_categoric.reset_index(drop=True, inplace=True)

    df_train = df_train.iloc[y_noano[y_noano['Top'] == 1].index.values]
    df_train.reset_index(drop=True, inplace=True)

    mean = train_numerical[columns_numerical_no_label].mean(axis=0)
    std = train_numerical[columns_numerical_no_label].std(axis=0)

    train_numerical[columns_numerical_no_label] = (train_numerical[columns_numerical_no_label] - mean) / std
    train_categoric = pd.get_dummies(train_categoric)
    train_numerical[['time']] = np.log(train_numerical[['time']])


    """Test Dataset"""
    df_test = pd.read_csv('data/test.csv')
    df_test = df_test[selected_features_without_label]
    test_numerical = df_test.select_dtypes(exclude=['object'])
    test_numerical.fillna(0, inplace=True)
    test_categoric = df_test.select_dtypes(include=['object'])
    test_categoric.fillna('NONE', inplace=True)

    test_numerical[columns_numerical_no_label] = (test_numerical[columns_numerical_no_label] - mean) / std
    test_categoric = pd.get_dummies(test_categoric)

    test_data = test_numerical.merge(test_categoric,left_index=True, right_index=True)
    train_data = train_numerical.merge(train_categoric,left_index=True, right_index=True)

    return train_data,test_data



get_data()
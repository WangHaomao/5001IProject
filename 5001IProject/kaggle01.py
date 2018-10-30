import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE

# import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor,ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor

class time_predictor:
    def __init__(self,train_data):
        # self.selected_features = ['penalty', 'l1_ratio', 'alpha', 'max_iter', 'random_state', 'n_jobs',
        #    'n_samples', 'n_features', 'n_classes', 'n_clusters_per_class',
        #    'n_informative', 'flip_y', 'scale']
        self.selected_features = ['penalty', 'max_iter', 'random_state', 'n_jobs',
                                  'n_samples', 'n_features', 'n_classes', 'n_clusters_per_class',
                                  'n_informative', 'scale']

        # self.selected_features = ['penalty', 'alpha', 'max_iter', 'random_state', 'n_jobs',
        #                            'n_samples', 'n_classes', 'n_informative', 'flip_y', 'scale']
        self.train_features = train_data[self.selected_features]
        # self.train_features['alpha'] = self.train_features['alpha'] * 10000
        # print(self.train_features['alpha'])
        self.train_features = pd.get_dummies(self.train_features)
        self.train_labels = train_data[['time']]
        print(self.train_features.columns)

    def features_selection(self):
        model = LinearRegression()
        # print(model.max_features)
        rfe = RFE(model, 3)
        fit = rfe.fit(self.train_features, self.train_labels)
        print("Selected Rinks:", fit.ranking_)
        print("Selected Features:", fit.support_)

    def cross_validation(self):
        # model = LinearRegression(normalize=True)
        model = LinearRegression()
        my_pipeline = make_pipeline(SimpleImputer(), model)
        scores = cross_val_score(my_pipeline, self.train_features, self.train_labels, scoring='neg_mean_absolute_error',cv=3)
        print('Mean Absolute Error %2f' % (-1 * scores.mean()))
    def get_testX(self):
        test_x = pd.read_csv("data/test.csv")
        test_x = pd.get_dummies(test_x[self.selected_features])
        return test_x

    def ensemble_predict(self):
        model1 = ExtraTreesRegressor(n_estimators=1500,random_state=8,max_features="log2")
        # model = RandomForestRegressor(n_estimators=1200)
        model2 = GradientBoostingRegressor(n_estimators=1200,random_state=8,max_features="log2")
        # model = AdaBoostRegressor(n_estimators=1200,random_state=1,learning_rate=0.15)

        # my_pipeline = make_pipeline(SimpleImputer(), model)
        # scores = cross_val_score(model, self.train_features, self.train_labels, scoring='neg_mean_absolute_error',
        #                          cv=3)
        # print('Mean Absolute Error %2f' % (-1 * scores.mean()))


        # train_X,val_X,train_y,val_y = train_test_split(self.train_features,self.train_labels)
        # model.fit(train_X, train_y.values.ravel())
        # print("error:", mean_squared_error(val_y,model.predict(val_X)))

        model1.fit(self.train_features,self.train_labels.values.ravel())
        model2.fit(self.train_features, self.train_labels.values.ravel())
        test_x = self.get_testX()
        # test_x = pd.read_csv("nee.csv")
        # print(test_x.columns)
        # print(self.train_features.columns)
        test1_y = model1.predict(test_x)
        test2_y = model2.predict(test_x)
        ll = len(test1_y)
        for i in range(0,ll):
            print(test1_y[i],test2_y[i])
        # print(ans_y)
        # print(len(test_y))
        # np.savetxt('data/testResult4.csv', test_y, delimiter=",", fmt="%.2f")


def predict(train_data):
    all_features = ['penalty', 'l1_ratio', 'alpha', 'max_iter', 'random_state', 'n_jobs',
           'n_samples', 'n_features', 'n_classes', 'n_clusters_per_class',
           'n_informative', 'flip_y', 'scale', 'time']

    X = pd.get_dummies(train_data[['penalty', 'alpha', 'max_iter', 'random_state', 'n_jobs',
           'n_samples', 'n_classes','n_informative', 'flip_y', 'scale']])
    # X = pd.get_dummies(train_data[['penalty', 'l1_ratio', 'alpha', 'max_iter', 'random_state', 'n_jobs',
    #  'n_samples', 'n_features', 'n_classes', 'n_clusters_per_class',
    #  'n_informative', 'flip_y', 'scale']])

    y = train_data[['time']]
    # print(X.columns)
    # train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    model = LinearRegression(normalize=True)
    model = LinearRegression()

    # tmp_X = pd.read_csv("nee.csv")
    my_pipeline = make_pipeline(SimpleImputer(), model)
    # scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error',cv=4)
    # print('Mean Absolute Error %2f' % (-1 * scores.mean()))

    # my_pipeline.fit(train_X,train_y)
    # ans_y = my_pipeline.predict(val_X)
    #
    # print(mean_squared_error(val_y, ans_y))

    my_pipeline.fit(X,y)

    test_x = pd.read_csv("data/test.csv")
    test_x = pd.get_dummies(test_x[['penalty', 'alpha', 'max_iter', 'random_state', 'n_jobs',
                                    'n_samples', 'n_classes', 'n_informative', 'flip_y', 'scale']])

    ans_y = my_pipeline.predict(X)
    test_y = my_pipeline.predict(test_x)

    print(mean_squared_error(y, ans_y))
    lens = len(y['time'])


    # print(ans_y)

    features2 = {'x':[]}
    labels2 ={'y':[]}
    for i in range(0, lens):
        if(ans_y[i][0] < 0):
            features2['x'].append(ans_y[i][0])
            labels2['y'].append(y['time'][i])

    X2_DataFrame = pd.DataFrame(data=features2)
    y2_DataFrame = pd.DataFrame(data=labels2)

    model2 = LinearRegression()
    pipeline2 = make_pipeline(model2)
    pipeline2.fit(X2_DataFrame,y2_DataFrame)

    lens = len(test_y)
    for i in range(0, lens):
        if(test_y[i][0] < 0):
            # ans_y[i][0] = pipeline2.predict(pd.DataFrame(ans_y[i][0]))

            tmp = pd.DataFrame(data={'X':[test_y[i][0]]})
            test_y[i][0] = pipeline2.predict(tmp)[0][0]

    print(test_y)
    # np.savetxt('data/testResult.csv', test_y, delimiter=",",fmt="%.2f")
    # ans = pipeline2.predict(X2_DataFrame)
    # print(ans)
    # print(y2_DataFrame)

if __name__ == '__main__':
    train_data = pd.read_csv("data/train.csv")
    predictor = time_predictor(train_data)
    # predictor.features_selection()
    # print(train_data.describe())
    # print(train_data.columns)
    # predictor.cross_validation()
    # predict(train_data)
    predictor.ensemble_predict()



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import datetime

import pandas as pd

import numpy as np

from sklearn.cross_validation import train_test_split

import xgboost as xgb

import random

import zipfile

import time

import shutil

from sklearn.metrics import log_loss

from pandas import DataFrame



random.seed(2016)



def run_xgb(train, test, features, target, random_state=0):

    eta = 0.1

    max_depth = 3

    subsample = 0.7

    colsample_bytree = 0.7

    start_time = time.time()



    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))

    params = {

        "objective": "multi:softprob",

        "num_class": 12,

        "booster" : "gbtree",

        "eval_metric": "mlogloss",

        "eta": eta,

        "max_depth": max_depth,

        "subsample": subsample,

        "colsample_bytree": colsample_bytree,

        "silent": 1,

        "seed": random_state,

    }

    num_boost_round = 500

    early_stopping_rounds = 50

    test_size = 0.3



    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)

    print('Length train:', len(X_train.index))

    print('Length valid:', len(X_valid.index))

    y_train = X_train[target]

    y_valid = X_valid[target]

    dtrain = xgb.DMatrix(X_train[features], y_train)

    dvalid = xgb.DMatrix(X_valid[features], y_valid)



    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)



    print("Validating...")

    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)

    score = log_loss(y_valid.tolist(), check)



    print("Predict test set...")

    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)



    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))

    return test_prediction.tolist(), score





def create_submission(score, test, prediction):

    # Make Submission

    now = datetime.datetime.now()

    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

    print('Writing submission: ', sub_file)

    f = open(sub_file, 'w')

    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')

    total = 0

    test_val = test['device_id'].values

    for i in range(len(test_val)):

        str1 = str(test_val[i])

        for j in range(12):

            str1 += ',' + str(prediction[i][j])

        str1 += '\n'

        total += 1

        f.write(str1)

    f.close()







def map_column(table, f):

    labels = sorted(table[f].unique())

    mappings = dict()

    for i in range(len(labels)):

        mappings[labels[i]] = i

    table = table.replace({f: mappings})

    return table





def read_data():

    # Events

    print('Read events...')

    events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})

    events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')

    events_small = events[['device_id', 'counts', 'timestamp']].drop_duplicates('device_id', keep='first')



    # App events

    print('Read app_events...')

    app_events = pd.read_csv("../input/app_events.csv", dtype={'app_id': np.str})

    app_events.drop_duplicates('event_id', keep='first', inplace=True)



    # App labels

    print('Read app_labels...')

    app_labels = pd.read_csv("../input/app_labels.csv", dtype={'app_id': np.str})



    # Phone brand

    print('Read brands...')

    models = pd.read_csv("../input/phone_brand_device_model.csv", dtype={'device_id': np.str})

    models.drop_duplicates('device_id', keep='first', inplace=True)

    map_column(models, 'device_model')

    map_column(models, 'phone_brand')

    

    # training data

    print('Read training data...')

    train = pd.read_csv("../input/gender_age_train.csv", dtype={'device_id': np.str})

    

    # test data

    print('Read test data...')

    test = pd.read_csv('../input/gender_age_test.csv', dtype={'device_id': np.str})



    nevents = events.event_id.nunique()

    nlabels = app_labels.label_id.nunique()

    napps = app_labels.app_id.nunique()

    ntrain = train.shape[0]

    ntest = test.shape[0]



    print('Altogether %d events' % nevents)

    print('Altogether %d different apps' %napps)

    print('Altogether %d labels' %nlabels)

    print('Altogether %d devices in the training data' %ntrain)

    print('Altogether %d devices in the test data' % ntest)



    return events_small,app_events , app_labels, models, train, test





# raw data from the given datasets

events, app_events, app_labels, models, otrain, otest = read_data()


# create train and test sets

# feature space: brand, model, frequency of events, number of involved apps,

# distribution of events during the day, distribution of app labels

import time

from time import mktime

from datetime import datetime

def create_train_test(events, app_events, app_labels, models, otrain, otest):

    train = otrain

    test = otest

    

    # model and brand

    train = pd.merge(train, models, how='left', on='device_id')

    train.set_index('device_id')

      

    test = pd.merge(test, models, how='left', on='device_id')

    test.set_index('device_id')

    

    # number of events

    ec = events.drop_duplicate('device_id')['device_id','counts']

    ec = ec.set_index('device_id')

    

    # average longitude and latitude

    lo = events.groupby('device_id')['longitude'].mean()

    la = events.groupby('device_id')['latitude'].mean()

    

    # distribution of event time during the day

    def timestamp_to_hour(row):

        return datetime.strptime(row.timestamp[0], "%Y-%m-%d %H:%M:%S").hour

    hour = events.apply(timestamp_to_hour, axis=1)

    events['hour'] = hour

    et = events.groupby['device_id']['hour'].count()



    train = pd.concat([train, ec, lo, la, et], axis =1)

    test = pd.concat([test, ec, lo, la, et], axis=1)

    

    return train, test
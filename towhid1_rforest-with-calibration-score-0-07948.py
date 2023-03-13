

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV as cc



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print (train.shape)

print (test.shape)

test_ids = test.id 



levels=train.species

train.drop(['species', 'id'], axis=1,inplace=True) 

test.drop(['id'],axis=1,inplace=True)

print ("after ")

print (levels.shape)

print (train.shape)

print (test.shape)

print ("number of classes =",levels.unique().shape)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder().fit(levels)

levels=le.transform(levels)
Model=RandomForestClassifier(n_estimators=1000)

Model = cc(Model, cv=3, method='isotonic')

Model.fit(train, levels)
predictions = Model.predict_proba(test)

print (predictions.shape)

sub = pd.DataFrame(predictions, columns=list(le.classes_))

sub.insert(0, 'id', test_ids)

sub.reset_index()

sub.to_csv('submit.csv', index = False)

sub.head()            
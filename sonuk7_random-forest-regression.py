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
train = pd.read_csv(r'../input/train.csv')

test = pd.read_csv(r'../input/test.csv')
features = [x for x in train.columns if x not in ['id','loss']]

#print(features)



cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]

num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id','loss']]

#print(cat_features)

#print(num_features)
ntrain = train.shape[0]

ntest = test.shape[0]

train_test = pd.concat((train[features], test[features])).reset_index(drop=True)

for c in range(len(cat_features)):

    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes



train_x = train_test.iloc[:ntrain,:]

test_x = train_test.iloc[ntrain:,:]



#print(train_x)

#print(test_x)

print('Ok')
train['log_loss'] = np.log(train['loss'])
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=30, min_samples_split=1)

regressor.fit(train_x, train['log_loss'])

res = np.exp(regressor.predict(test_x))

print('Ok')
df = pd.DataFrame()

df['id']=test['id']

df['loss']=res

df.to_csv('submission.csv',index=False)

print('end')
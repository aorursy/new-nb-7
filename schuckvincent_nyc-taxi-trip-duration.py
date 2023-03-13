import os

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

import math

from datetime import datetime

sns.set({'figure.figsize':(16,8), 'axes.titlesize':15, 'axes.labelsize':10})
FILEPATH = os.path.join("..", "input")

TRAINPATH = os.path.join(FILEPATH, "train.csv")

TESTPATH = os.path.join(FILEPATH, "test.csv")
df = pd.read_csv(TRAINPATH, index_col=0)

df.head()
df_test = pd.read_csv(TESTPATH)

df_test.head()
df.info()
df.describe().T
df_test.info()
df_test.describe().T
plt.hist(df.loc[df.trip_duration<6000,"trip_duration"], bins=100);

plt.title('Répartition de voyage en taxi à NYC')

plt.xlabel('Durée d\'un voyage en taxi à NYC (sec)')

plt.ylabel('Nombre d\'enregistrement')

plt.show()
plt.hist(np.log(df.trip_duration), bins=200);

plt.title('Répartition de voyage en taxi à NYC (après utilisation d\'un logarithme sur les valeurs)')

plt.xlabel('Durée d\'un voyage en taxi à NYC')

plt.ylabel('Nombre d\'enregistrement')

plt.show()
plt.figure()

g = sns.boxplot(x = 'vendor_id', y = 'trip_duration', data=df[df['trip_duration'] < 3000])

plt.title('Corrélation entre le vendeur et le temps de trajet d\'une course de taxi')

plt.xlabel('Identifiant du vendeur')

plt.ylabel('Temps de trajet d\'une course de taxi')

plt.show()
plt.figure()

g = sns.boxplot(x = 'passenger_count', y = 'trip_duration', data=df[df['trip_duration'] < 3000])

plt.title('Corrélation entre le nombre de passager et le temps de trajet d\'une course de taxi')

plt.xlabel('Nombre de passager')

plt.ylabel('Temps de trajet d\'une course de taxi')

plt.show()
df = df[df.passenger_count != 0]
df.duplicated().sum()
df = df.drop_duplicates()

df.duplicated().sum()
df.isna().sum()
fig, ax = plt.subplots()

df.boxplot(['trip_duration'], fontsize=12)

fig.suptitle('Visualisation des outliers', fontsize=20)
df = df[(df['trip_duration'] > 60) & (df['trip_duration'] < 3600 * 6)]
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].astype('category').cat.codes

df_test['store_and_fwd_flag'] = df_test['store_and_fwd_flag'].astype('category').cat.codes
from sklearn.decomposition import PCA
coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,

                    df[['dropoff_latitude', 'dropoff_longitude']].values,

                    df_test[['pickup_latitude', 'pickup_longitude']].values,

                    df_test[['dropoff_latitude', 'dropoff_longitude']].values))



pca = PCA().fit(coords)



#Pour le fichier de train

df['pickup_pca0'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 0]

df['pickup_pca1'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 1]

df['dropoff_pca0'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

df['dropoff_pca1'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 1]



#Pour le fichier de test

df_test['pickup_pca0'] = pca.transform(df_test[['pickup_latitude', 'pickup_longitude']])[:, 0]

df_test['pickup_pca1'] = pca.transform(df_test[['pickup_latitude', 'pickup_longitude']])[:, 1]

df_test['dropoff_pca0'] = pca.transform(df_test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

df_test['dropoff_pca1'] = pca.transform(df_test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
df.head(3)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
df['hour'] = df.pickup_datetime.dt.hour

df['day'] = df.pickup_datetime.dt.dayofweek

df['month'] = df.pickup_datetime.dt.month

df_test['hour'] = df_test.pickup_datetime.dt.hour

df_test['day'] = df_test.pickup_datetime.dt.dayofweek

df_test['month'] = df_test.pickup_datetime.dt.month
df['distance2'] = np.sqrt((df['pickup_pca0']-df['dropoff_pca0'])**2

                        + (df['pickup_pca1']-df['dropoff_pca1'])**2)

df_test['distance2'] = np.sqrt((df_test['pickup_pca0']-df_test['dropoff_pca0'])**2

                        + (df_test['pickup_pca1']-df_test['dropoff_pca1'])**2)
df['log_trip_duration'] = np.log(df['trip_duration'])
df.head(3)
df_test.head(3)
NUM_VARS = ['pickup_pca0', 'pickup_pca1', 'dropoff_pca0', 'dropoff_pca1', 'month', 'hour', 'day', 'distance2']

TARGET = 'log_trip_duration'
num_features = NUM_VARS
X_train = df.loc[:, num_features]

y_train = df[TARGET]

X_test = df_test.loc[:, num_features]

X_train.shape, y_train.shape, X_test.shape
from sklearn.ensemble import RandomForestRegressor
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, min_samples_split=15, max_features='auto', bootstrap=True)

m.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(m, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')

cv_scores
y_test_pred = m.predict(X_test)

y_test_pred[:5]
my_submission = pd.DataFrame({'id': df_test.id, 'trip_duration': np.exp(y_test_pred)})

my_submission.to_csv('submission.csv', index=False)
my_submission.head(100)
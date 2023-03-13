# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

from pandas import read_csv, DataFrame

import math

from keras.models import Sequential

from keras.layers import Dense, Dropout,Flatten, Reshape, Activation

from keras.layers import LSTM

from keras.layers import noise

from keras.models import load_model

from keras import optimizers

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.metrics import f1_score, accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#df_1=pd.read_csv('../input/2chan10dbnoise/output3.csv',header=None,nrows=1000)

df=pd.read_csv('../input/liverpool-ion-switching/train.csv')

dataset=df.values

print(dataset[0:20,:])
categorical_labels = to_categorical(dataset[:,2], num_classes= 11)

print(categorical_labels.shape)
#does this help?

scaler = MinMaxScaler(feature_range=(0, 1))

dataset[:,0:2] = scaler.fit_transform(dataset[:,0:2])

print(dataset[:10,:])
from keras import backend as K



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
batch_size=100

model = Sequential()

timestep=1

input_dim=1

model.add(LSTM(64, batch_input_shape=(batch_size, timestep, input_dim), stateful=True, return_sequences=True))

model.add(Flatten())

model.add(Dense(11))

model.add(Activation('softmax'))

#binary sinks like a stone! b/c not binary @@

#model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=[f1_m])



print(model.summary())
import math

train_size = math.floor(len(dataset) * 0.80/100)

train_size = int (train_size*100)

test_size = math.floor((len(dataset) - train_size)/100)

test_size = int(test_size*100)

print ('training set= ',train_size)

print('test set =', test_size)

print ('total length', test_size+train_size)

print ('Dataset= ', len(dataset))
in_train, in_test = dataset[0:train_size,1], dataset[train_size:train_size+test_size,1]

target_train, target_test = categorical_labels[0:train_size,:], categorical_labels[train_size:train_size+test_size,:]



in_train = in_train.reshape(len(in_train),1,1)

in_test = in_test.reshape(len(in_test), 1,1)



print('in_train Shape',in_train.shape)

print(in_train[0:2,:])

print('target_train Shape',target_train.shape)

state=np.argmax(target_train,axis=-1)

print(state)



print('in_test Shape',in_test.shape)

print(in_test[0:2,:])

print('target_test Shape',in_test.shape)

state=np.argmax(target_test,axis=-1)

print(state)
epochers=3

history=model.fit(x=in_train,y=target_train, initial_epoch=0, epochs=epochers, batch_size=batch_size, verbose=2, shuffle=False)
plt.plot(history.history['f1_m'])
predict = model.predict(in_train, batch_size=batch_size)

print(predict.shape)

print(predict[:5,:])
state_train=np.argmax(target_train,axis=-1)

class_predict_train=np.argmax(predict,axis=-1)

print(state_train[:20])

print(class_predict_train[:20])
print('F1_macro = ',f1_score(state_train,class_predict_train, average='macro'))
plotlen=test_size

starting_point = 15000

lenny=1000

#target_test = dataset[train_size:len(dataset),3]

#target_test = target_test.reshape(plotlen, 1)

plt.figure(figsize=(30,6))

plt.subplot(2,1,1)

#temp=scaler.inverse_transform(dataset)

#plt.plot (temp[train_size:len(dataset),1], color='blue', label="some raw data")

plt.plot (dataset[starting_point:starting_point+lenny,1], color='blue', label="some raw data")

plt.title("The raw test")

df=DataFrame(dataset[starting_point:starting_point+lenny,1])

plt.subplot(2,1,2)

#plt.plot(target_test.reshape(plotlen,1)*maxchannels, color='black', label="the actual idealisation")

plt.plot(state_train[starting_point:starting_point+lenny], color='black', label="the actual idealisation")

#plt.plot(spredict, color='red', label="predicted idealisation")

line,=plt.plot(class_predict_train[starting_point:starting_point+lenny], color='red', label="predicted idealisation")

plt.setp(line, linestyle='--')

plt.xlabel('timepoint')

plt.ylabel('current')

#plt.savefig(name)

plt.legend()

plt.show()
predict_test = model.predict(in_test, batch_size=batch_size)

print(predict_test.shape)

print(predict_test[:5,:])
state_test=np.argmax(target_test,axis=-1)

class_predict=np.argmax(predict_test,axis=-1)

print(state[:20])

print(class_predict[:20])
print('F1_macro = ',f1_score(state,class_predict, average='macro'))
plotlen=test_size

lenny=1000

#target_test = dataset[train_size:len(dataset),3]

#target_test = target_test.reshape(plotlen, 1)

plt.figure(figsize=(30,6))

plt.subplot(2,1,1)

#temp=scaler.inverse_transform(dataset)

#plt.plot (temp[train_size:len(dataset),1], color='blue', label="some raw data")

plt.plot (dataset[train_size:train_size+lenny,1], color='blue', label="some raw data")

plt.title("The raw test")

df=DataFrame(dataset[train_size:train_size+lenny,1])

plt.subplot(2,1,2)

#plt.plot(target_test.reshape(plotlen,1)*maxchannels, color='black', label="the actual idealisation")

plt.plot(state[0:lenny], color='black', label="the actual idealisation")

#plt.plot(spredict, color='red', label="predicted idealisation")

line,=plt.plot(class_predict[:lenny], color='red', label="predicted idealisation")

plt.setp(line, linestyle='--')

plt.xlabel('timepoint')

plt.ylabel('current')

#plt.savefig(name)

plt.legend()

plt.show()
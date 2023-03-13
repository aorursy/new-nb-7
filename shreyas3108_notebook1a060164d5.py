# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense , Conv2D , Dropout , Flatten

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/labels.csv")

test = pd.read_csv("../input/sample_submission.csv")
train.head()
train_label = pd.Series(train['breed'])
classes, counts = np.unique(train_label, return_counts=True)

print("Some classes with count:")

print(np.asarray((classes, counts)))

print("Number of class: %d" % classes.size)
import seaborn as sns

from matplotlib import pyplot as plt
#plt.barh(classes[:10] , counts[:10])

plt.barh(classes[:10],counts[:10])
plt.barh(classes[-10:],counts[-10:])
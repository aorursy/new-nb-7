# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')



df_train.head(5)
import seaborn as sns

corr = df_train.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
import matplotlib as plt

plt.pyplot.scatter(df_train['budget'],df_train['revenue'])

plt.pyplot.ylabel('Revenue')

plt.pyplot.xlabel('Budget')
import matplotlib as plt

plt.pyplot.scatter(df_train['popularity'],df_train['revenue'])

plt.pyplot.ylabel('Revenue')

plt.pyplot.xlabel('Popularity')
import matplotlib as plt

plt.pyplot.scatter(df_train['runtime'],df_train['revenue'])

plt.pyplot.ylabel('Revenue')

plt.pyplot.xlabel('Runtime')
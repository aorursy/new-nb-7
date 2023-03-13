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
train = pd.read_csv('../input/train.csv',
                    dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},
                    usecols=['srch_destination_id','is_booking','hotel_cluster'],
                    chunksize=1000000)
aggs = []
test = None
print('-'*38)
for i, chunk in enumerate(train):
    print(type(chunk))
    if i == 1:
        test = chunk
    agg = chunk.groupby(['srch_destination_id',
                         'hotel_cluster'])['is_booking'].agg(['sum','count'])
    agg.reset_index(inplace=True)
    aggs.append(agg)
    print('.',end='')
print('')
aggs = pd.concat(aggs, axis=0)
aggs.head()
type(test)
test.head()

test2 = test.groupby(['srch_destination_id',
                         'hotel_cluster'])['is_booking'].agg(['sum'])
test2.head()
test2 = test.groupby(by=['srch_destination_id',
                         'hotel_cluster']).sum().groupby(level=[0]).cumsum()
test2.head(10)
data1=pd.DataFrame({'Dir':['E','E','W','W','E','W','W','E'], 'Bool':['Y','N','Y','N','Y','N','Y','N'], 'Data':[4,5,6,7,8,9,10,11], 'Date':pd.DatetimeIndex(['12/30/2000','12/30/2000','12/30/2000','1/2/2001','1/3/2001','1/3/2001','12/30/2000','12/30/2000'])} )
data1.sort_values(by=['Bool', 'Dir','Date'], inplace=True)
data1.reset_index(inplace=True)
print(data1)
running_sum = data1.groupby(['Bool', 'Dir', 'Date']).apply(lambda x: x['Data'].cumsum())
print(running_sum)
data1['test'] = pd.DataFrame(running_sum.values)
data1
test2 = data1.groupby(by=['Bool',
                         'Dir', 'Date'], level=[0,1,2]).sum().groupby(level=[2]).cumsum()
test2.head(10)

data1=pd.DataFrame({'Dir':['E','E','W','W','E','W','W','E'], 'Bool':['Y','N','Y','N','Y','N','Y','N'], 'Data':[4,5,6,7,8,9,10,11]}, index=pd.DatetimeIndex(['12/30/2000','12/30/2000','12/30/2000','1/2/2001','1/3/2001','1/3/2001','12/30/2000','12/30/2000']))
data2 = data1.reset_index()
data3 = data2.set_index(["Bool", "Dir", "index"]) 
running_sum = data1.groupby(['Bool', 'Dir', 'Date']).sum().groupby(level=[0, 1]).cumsum()
running_sum
data3
CLICK_WEIGHT = 0.05
agg = aggs.groupby(['srch_destination_id','hotel_cluster']).sum().reset_index()
agg['count'] -= agg['sum']
agg = agg.rename(columns={'sum':'bookings','count':'clicks'})
agg['relevance'] = agg['bookings'] + CLICK_WEIGHT * agg['clicks']
agg.head()
def most_popular(group, n_max=5):
    ind = group.relevance.nlargest(n_max).index
    most_popular = group.hotel_cluster[ind].values
    return np.array_str(most_popular)[1:-1] # remove square brackets
most_pop = agg.groupby(['srch_destination_id']).apply(most_popular)
most_pop = pd.DataFrame(most_pop).rename(columns={0:'hotel_cluster'})
most_pop.head()
test = pd.read_csv('../input/test.csv',
                    dtype={'srch_destination_id':np.int32},
                    usecols=['srch_destination_id'],)
test = test.merge(most_pop, how='left',left_on='srch_destination_id',right_index=True)
test.head()
test.hotel_cluster.isnull().sum()
most_pop_all = agg.groupby('hotel_cluster')['relevance'].sum().nlargest(5).index
most_pop_all = np.array_str(most_pop_all)[1:-1]
most_pop_all
test.hotel_cluster.fillna(most_pop_all,inplace=True)
test.hotel_cluster.to_csv('predicted_with_pandas.csv',header=True, index_label='id')

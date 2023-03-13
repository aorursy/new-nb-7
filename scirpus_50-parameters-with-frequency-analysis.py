import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score
def add_noise(series, noise_level):

    #return series * (1 + noise_level * np.random.randn(len(series)))

    series[series!=-9999] = series[series!=-9999]* (1 + .01 * np.random.randn(len(series[series!=-9999])))

    return series



def target_encode(trn_series=None, 

                  tst_series=None, 

                  target=None, 

                  min_samples_leaf=100, 

                  smoothing=10,

                  noise_level=0):

    """

    Smoothing is computed like in the following paper by Daniele Micci-Barreca

    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf

    trn_series : training categorical feature as a pd.Series

    tst_series : test categorical feature as a pd.Series

    target : target data as a pd.Series

    min_samples_leaf (int) : minimum samples to take category average into account

    smoothing (int) : smoothing effect to balance categorical average vs prior  

    """ 

    assert len(trn_series) == len(target)

    assert trn_series.name == tst_series.name

    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean 

    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    # Compute smoothing

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data

    prior = target.mean()

    # The bigger the count the less full_avg is taken into account

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing

    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to trn and tst series

    ft_trn_series = pd.merge(

        trn_series.to_frame(trn_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=trn_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_trn_series.index = trn_series.index 

    ft_tst_series = pd.merge(

        tst_series.to_frame(tst_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=tst_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_tst_series.index = tst_series.index

    if(noise_level>0):

        return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

    return ft_trn_series, ft_tst_series



def GrabFrequencies(data, myfeatures):

    for c in myfeatures:

        print(c)

        x = data[['year','month','weekday','hour',c,'TransactionID']].groupby(['year','month','weekday','hour',c]).TransactionID.count().reset_index(drop=False)

        y = data[['year','month','weekday','hour','TransactionID']].groupby(['year','month','weekday','hour']).TransactionID.count().reset_index(drop=False)

        x = x.merge(y,on=['year','month','weekday','hour'],how='left')

        x['freq_'+c] = x.TransactionID_x/x.TransactionID_y

        del x['TransactionID_x']

        del x['TransactionID_y']

        data = data.merge(x,on=['year','month','weekday','hour',c],how='left')

        

    return data
features = ['card1', 'C1', 'card4', 'C6', 'C14', 'V45', 'M6', 'M5', 'card2',

           'C5', 'V283', 'V294', 'D8', 'M4', 'C2', 'D14',

           'dist2', 'C10', 'D2', 'R_emaildomain', 'D10', 'V315', 'D1',

           'dist1', 'D11', 'C12', 'P_emaildomain', 'D15', 'C11', 'D4', 'V313',

           'C8', 'D9', 'V312', 'C13', 'D3', 'C9', 'V310', 'V133', 'V314',

           'V130', 'V317', 'V83', 'card5', 'V308', 'addr1', 'V127', 'V307',

           'D5']

len(features)
train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
import datetime

def convert_TranactionDT(df):

    try:

        START_DATE = "2017-12-01"

        startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

        df["TransactionDT"] = df["TransactionDT"].apply(

            lambda x: (startdate + datetime.timedelta(seconds=x))

        )

        return df

    except TypeError:

        """Already converted?"""

        return df



train = convert_TranactionDT(train)

test = convert_TranactionDT(test)
train['year'] = train['TransactionDT'].dt.year

test['year'] = test['TransactionDT'].dt.year



train['month'] = train['TransactionDT'].dt.month

test['month'] = test['TransactionDT'].dt.month



train['weekday'] = train['TransactionDT'].dt.weekday

test['weekday'] = test['TransactionDT'].dt.weekday



train['hour'] = train['TransactionDT'].dt.hour

test['hour'] = test['TransactionDT'].dt.hour
allfeatures = list(features)+list(['year','month','weekday','hour'])

print(allfeatures)
train.head()
test.head()
trainfreq = GrabFrequencies(train[list(['TransactionID','TransactionAmt'])+list(allfeatures)].fillna(-9999), list(features)+list(['TransactionAmt']))

testfreq = GrabFrequencies(test[list(['TransactionID','TransactionAmt'])+list(allfeatures)].fillna(-9999), list(features)+list(['TransactionAmt']))
trainfreq.insert(1,'isFraud',train.isFraud)

trainfreq.head()
for c in features:

    print(c)

    

    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    trainpreds =  np.zeros(trainfreq.shape[0])

    testpreds =  np.zeros(testfreq.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(trainfreq, trainfreq.isFraud)):

        a, b = target_encode( trainfreq.iloc[trn_idx][c].copy().reset_index(drop=True),

                              trainfreq.iloc[val_idx][c].copy().reset_index(drop=True),

                              trainfreq.iloc[trn_idx].isFraud.copy().reset_index(drop=True),

                              min_samples_leaf=80, 

                              smoothing=1,

                              noise_level=.0)

        trainpreds[val_idx] = b

        a, b = target_encode( trainfreq.iloc[trn_idx][c].copy().reset_index(drop=True),

                              testfreq[c].copy().reset_index(drop=True),

                              trainfreq.iloc[trn_idx].isFraud.copy().reset_index(drop=True),

                              min_samples_leaf=80, 

                              smoothing=1,

                              noise_level=.0)

        

        testpreds += (b/5)

        

    trainfreq[c] = trainpreds

    testfreq[c] = testpreds

trainfreq['TransactionAmt'] = train['TransactionAmt'].values

testfreq['TransactionAmt'] = test['TransactionAmt'].values
trainfreq.head()
testfreq.head()
def Output(p):

    return 1./(1.+np.exp(-p))





def GPI(data):

    return Output(  -3.317076 +

                    0.040000*np.tanh(((((np.tanh(((((5.0)) * (((((data["card1"]) * 2.0)) * 2.0)))))) + ((((5.0)) * (((((data["D8"]) - (((data["freq_C8"]) - (((((((5.0)) * (((((((data["card1"]) * 2.0)) - (data["D8"]))) * 2.0)))) > (data["freq_V317"]))*1.)))))) * 2.0)))))) * 2.0)) +

                    0.039360*np.tanh(((((((((((((((((-1.0*((data["freq_C10"])))) + ((((((((((data["freq_V317"]) <= (data["V310"]))*1.)) + (((data["C14"]) * 2.0)))) * 2.0)) * 2.0)))/2.0)) * 2.0)) * 2.0)) + (data["card2"]))) - (data["freq_C2"]))) * 2.0)) * 2.0)) +

                    0.001840*np.tanh(((((((((((((data["card1"]) * ((((13.91004276275634766)) - ((((data["card1"]) > (((data["freq_V294"]) * 2.0)))*1.)))))) + ((((((data["card1"]) <= (((data["freq_C8"]) - (data["D8"]))))*1.)) * ((-1.0*((((((data["freq_V294"]) - (data["card1"]))) * 2.0))))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.010800*np.tanh(((data["TransactionAmt"]) * (((((data["card1"]) * 2.0)) + ((((((((np.tanh((data["D8"]))) - (np.tanh((data["freq_C10"]))))) + (((((((((data["card1"]) > (data["D8"]))*1.)) * (data["card1"]))) > (data["freq_V317"]))*1.)))) + (data["card1"]))/2.0)))))) +

                    0.044634*np.tanh(((((-2.0) * (((data["freq_V294"]) - ((((-1.0*((((data["freq_D8"]) - (((((((((data["C10"]) + (((((((data["D2"]) + (data["V312"]))) + (data["card1"]))) * 2.0)))) * 2.0)) + ((((data["freq_V283"]) <= (((data["card1"]) * 2.0)))*1.)))) * 2.0))))))) * 2.0)))))) * 2.0)) +

                    0.029261*np.tanh(((data["card1"]) - (((data["TransactionAmt"]) * (((((data["freq_C8"]) - ((((((data["freq_V294"]) <= (((((((data["card1"]) * 2.0)) * 2.0)) * 2.0)))*1.)) * 2.0)))) - ((((((data["freq_C8"]) <= (((data["C14"]) * 2.0)))*1.)) * 2.0)))))))) +

                    0.024032*np.tanh(((data["freq_C5"]) - (((((((((((((((((data["freq_V283"]) - ((((((data["C14"]) * 2.0)) > (((data["freq_C12"]) - ((((((data["C14"]) * 2.0)) > (((data["freq_V294"]) * (data["freq_C8"]))))*1.)))))*1.)))) * 2.0)) * 2.0)) - (data["freq_C5"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +

                    0.096640*np.tanh((((((((((((data["dist2"]) <= (((((data["card1"]) * 2.0)) - (data["freq_V308"]))))*1.)) * 2.0)) - (((data["freq_V317"]) - (((((((((((((((data["card1"]) * 2.0)) * 2.0)) * 2.0)) - (data["freq_C2"]))) * 2.0)) - (((data["freq_C10"]) - (data["freq_C5"]))))) * 2.0)))))) * 2.0)) * 2.0)) +

                    0.0*np.tanh((((((((((((((((((((((((((((((data["card1"]) * 2.0)) + (np.tanh((((((((data["card1"]) * 2.0)) * 2.0)) - (((data["freq_V294"]) * (data["freq_C10"]))))))))/2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.023600*np.tanh((-1.0*((((data["card1"]) + (((((data["freq_V133"]) + (((((np.tanh((data["freq_M4"]))) + (((((data["freq_V294"]) + ((((((((((data["freq_M4"]) <= (((data["freq_C8"]) * 2.0)))*1.)) / 2.0)) - ((((12.65574836730957031)) * (((data["card1"]) * 2.0)))))) * 2.0)))) * 2.0)))) * 2.0)))) * 2.0))))))) +

                    0.028000*np.tanh((((((((((((((data["freq_V83"]) * (((data["freq_C8"]) / 2.0)))) <= (data["C13"]))*1.)) + (((((((((((((data["card1"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - (((data["freq_C8"]) / 2.0)))) - (data["freq_V317"]))))) * 2.0)) * 2.0)) * 2.0)) +

                    0.061283*np.tanh(((((data["TransactionAmt"]) * (((((((data["TransactionAmt"]) * ((((((data["card1"]) - (((data["freq_V294"]) * ((((((data["freq_C1"]) - (data["V314"]))) + (((((data["freq_C8"]) / 2.0)) - (((data["R_emaildomain"]) * 2.0)))))/2.0)))))) + (data["card1"]))/2.0)))) * 2.0)) * 2.0)))) * 2.0)) +

                    0.099993*np.tanh(((((((data["card1"]) * 2.0)) - (np.tanh((((data["M5"]) + (((data["freq_V308"]) * (((((data["freq_M5"]) / 2.0)) - ((((data["R_emaildomain"]) > (data["M5"]))*1.)))))))))))) * (((((data["freq_M6"]) * (data["year"]))) - (data["freq_C5"]))))) +

                    0.099979*np.tanh((((11.35198402404785156)) * (((data["freq_C5"]) + (((((((data["freq_V308"]) + ((((data["freq_C1"]) > (data["freq_D2"]))*1.)))) + ((((data["M5"]) > (data["D2"]))*1.)))) * (((np.tanh(((-1.0*((((((data["freq_V133"]) + (data["freq_V133"]))) + (data["freq_C11"])))))))) / 2.0)))))))) +

                    0.099936*np.tanh(((((((np.tanh((data["card1"]))) + ((((data["freq_C2"]) <= (np.tanh((data["freq_D2"]))))*1.)))) - (((data["freq_C12"]) + (((data["freq_C1"]) + (((((((data["freq_C10"]) * (data["freq_V294"]))) * 2.0)) + (((data["freq_V294"]) - (((((data["TransactionAmt"]) * (data["card1"]))) / 2.0)))))))))))) * 2.0)) +

                    0.078717*np.tanh(((data["year"]) * ((((((((data["card1"]) + (data["D15"]))/2.0)) * ((6.35084676742553711)))) + ((((((-1.0*((np.tanh(((((((data["freq_V133"]) + (data["freq_C10"]))) + (data["card1"]))/2.0))))))) + (((((((data["freq_C10"]) + ((-1.0*((data["freq_C1"])))))/2.0)) + (data["card1"]))/2.0)))) / 2.0)))))) +

                    0.098160*np.tanh(((data["year"]) * ((((data["C14"]) + (((((((((data["card1"]) > (((((((data["freq_V312"]) * (((data["freq_C2"]) * 2.0)))) + (data["card1"]))) * (((data["freq_V133"]) - (np.tanh((data["freq_D2"]))))))))*1.)) + (((data["freq_D2"]) + ((-1.0*((data["freq_V317"])))))))/2.0)) / 2.0)))/2.0)))) +

                    0.099176*np.tanh(((data["TransactionAmt"]) * (((data["V312"]) + (((((((((((-1.0*(((((data["freq_C2"]) + (data["card4"]))/2.0))))) / 2.0)) * 2.0)) + (((data["V317"]) - (((data["freq_C12"]) - (data["card4"]))))))) + (((((((data["card4"]) <= (data["C5"]))*1.)) > (data["freq_C5"]))*1.)))/2.0)))))) +

                    0.099250*np.tanh(((((((((((((((((data["card1"]) + (((data["C1"]) * 2.0)))) * 2.0)) + ((((((((data["freq_V45"]) > ((((data["card4"]) > (((((((data["V314"]) + (data["V313"]))/2.0)) + (data["C14"]))/2.0)))*1.)))*1.)) * 2.0)) - (data["freq_V45"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.099806*np.tanh(((((((((((((((-1.0*((((np.tanh((((((data["freq_R_emaildomain"]) - (data["card1"]))) - ((-1.0*((((np.tanh((data["freq_C1"]))) * 2.0))))))))) * 2.0))))) + (((np.tanh((data["card1"]))) * (((data["TransactionAmt"]) / 2.0)))))/2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.099400*np.tanh((((((data["freq_V307"]) * 2.0)) + ((((data["card4"]) + (((((data["year"]) + (data["year"]))) * ((((np.tanh((data["R_emaildomain"]))) + (((((-1.0*((data["freq_C1"])))) + (((data["card1"]) + (((data["freq_D9"]) - ((((data["card4"]) > ((((data["card1"]) + (data["C5"]))/2.0)))*1.)))))))/2.0)))/2.0)))))/2.0)))/2.0)) +

                    0.097400*np.tanh(((-3.0) - (((((data["month"]) - (data["card2"]))) - (((data["dist1"]) + ((((data["D5"]) + ((-1.0*((((((data["P_emaildomain"]) * (((data["year"]) * (data["C1"]))))) * (((-3.0) * (((data["card1"]) * (((data["year"]) * (data["C5"])))))))))))))/2.0)))))))) +

                    0.098635*np.tanh(((((((data["freq_C5"]) - (((((((((((((((((data["freq_C14"]) * (data["freq_C8"]))) <= (data["C13"]))*1.)) * (data["freq_D2"]))) * 2.0)) * 2.0)) - ((((((-1.0*((data["V314"])))) * (data["freq_C8"]))) + ((-1.0*((((data["card1"]) * 2.0))))))))) <= (data["freq_V283"]))*1.)))) * 2.0)) * 2.0)) +

                    0.099980*np.tanh(((data["year"]) * ((((((np.tanh((data["V310"]))) / 2.0)) + ((((((data["D10"]) + ((((((data["V310"]) + ((((((data["C13"]) + ((((data["card1"]) > (((data["freq_C1"]) / 2.0)))*1.)))/2.0)) - (data["M5"]))))/2.0)) - (data["M5"]))))/2.0)) - (data["M5"]))))/2.0)))) +

                    0.098712*np.tanh((((((((data["TransactionAmt"]) + ((7.0)))/2.0)) * (((data["card1"]) * ((((((((7.0)) * 2.0)) * (data["freq_C5"]))) - (((data["freq_V310"]) + (((((data["month"]) * (data["freq_V317"]))) / 2.0)))))))))) - (data["month"]))) +

                    0.100000*np.tanh(((((((((((((((((((((data["freq_C13"]) > (data["freq_C1"]))*1.)) * 2.0)) + ((((data["C14"]) > (data["card4"]))*1.)))) > (((data["freq_V283"]) - (data["V313"]))))*1.)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) + (-2.0))) +

                    0.094408*np.tanh((((((((data["freq_C11"]) <= ((((14.45279026031494141)) * ((((data["D2"]) + (data["card1"]))/2.0)))))*1.)) * (((data["year"]) * (((((data["card1"]) * (((data["year"]) * (data["D2"]))))) - (np.tanh(((14.45279026031494141)))))))))) - ((((14.45279026031494141)) + (((data["year"]) * (data["card1"]))))))) +

                    0.099989*np.tanh(((((((((((-1.0*((data["freq_V45"])))) + ((((-1.0*((data["freq_C6"])))) + (data["card1"]))))/2.0)) * 2.0)) - ((((((data["freq_C6"]) * (((data["month"]) + (data["month"]))))) + (((((data["C13"]) * (data["card1"]))) * ((-1.0*((((data["year"]) + (data["card1"])))))))))/2.0)))) * 2.0)) +

                    0.099980*np.tanh(((((((data["card1"]) * ((((((((data["freq_D5"]) * 2.0)) * (data["freq_D5"]))) + (((((data["year"]) * (data["freq_D5"]))) * (((((((data["year"]) * (((((data["D2"]) * 2.0)) * 2.0)))) * (data["dist1"]))) - (2.0))))))/2.0)))) * 2.0)) - ((6.0)))) +

                    0.091696*np.tanh(((data["year"]) * ((((((data["C14"]) > (((data["D9"]) * 2.0)))*1.)) + ((((((((data["V313"]) + ((((data["freq_C1"]) <= (((data["C14"]) * 2.0)))*1.)))/2.0)) - ((((((data["C14"]) * (data["C14"]))) + (((data["C10"]) + (((data["freq_C6"]) * (data["C14"]))))))/2.0)))) * 2.0)))))) +

                    0.099956*np.tanh((((((((np.tanh(((((((data["freq_C1"]) - (data["C14"]))) <= (data["V310"]))*1.)))) * 2.0)) + (((((((((data["C14"]) * 2.0)) * 2.0)) + ((((data["freq_C1"]) + (data["C14"]))/2.0)))) - (data["freq_M5"]))))/2.0)) * (((((data["C14"]) - (data["V130"]))) + (data["year"]))))) +

                    0.099200*np.tanh(((data["TransactionAmt"]) * (((data["freq_P_emaildomain"]) - ((((((data["freq_M5"]) * ((((data["freq_C1"]) > (data["freq_D15"]))*1.)))) + (((data["freq_C10"]) * (((data["freq_C8"]) * (((data["freq_V312"]) + (((data["freq_M5"]) - ((((((((5.0)) * (data["dist1"]))) * (data["freq_D14"]))) * 2.0)))))))))))/2.0)))))) +

                    0.099002*np.tanh((-1.0*((((((data["month"]) - (((-1.0) * 2.0)))) - (((data["TransactionAmt"]) * (((data["V45"]) + (((((((((data["C11"]) > (data["M5"]))*1.)) * ((((((data["D15"]) > (data["M5"]))*1.)) + ((((data["V45"]) > (data["M5"]))*1.)))))) > (data["M5"]))*1.))))))))))) +

                    0.086080*np.tanh(((((((((((((data["card1"]) * 2.0)) - (((data["freq_V133"]) + ((-1.0*(((((((((data["freq_C5"]) <= (((data["freq_D2"]) * ((((((data["freq_V133"]) + (data["TransactionAmt"]))/2.0)) / 2.0)))))*1.)) * 2.0)) * ((((data["freq_C12"]) <= (((data["freq_C5"]) * 2.0)))*1.))))))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.099950*np.tanh((((((((data["freq_C10"]) + (data["freq_C9"]))/2.0)) * (((((((data["freq_C11"]) - (data["freq_C10"]))) - ((((data["card1"]) > (np.tanh((data["freq_V130"]))))*1.)))) + (data["freq_C9"]))))) * (((data["year"]) * (((((((data["D9"]) - (data["freq_C1"]))) + (data["C14"]))) * 2.0)))))) +

                    0.051587*np.tanh(((((data["TransactionAmt"]) * ((((((10.0)) * (data["R_emaildomain"]))) - (((((((((data["freq_C1"]) * ((((((data["freq_C6"]) * (((((10.0)) > (data["V310"]))*1.)))) > (data["freq_C1"]))*1.)))) + (data["freq_C6"]))/2.0)) > (((data["V314"]) * (((data["TransactionAmt"]) * (data["card1"]))))))*1.)))))) * 2.0)) +

                    0.099006*np.tanh((((((((((((((((((((np.tanh((data["freq_C13"]))) > (data["freq_C1"]))*1.)) - (data["freq_C6"]))) - (((np.tanh((data["C14"]))) - ((((data["M6"]) <= (data["C14"]))*1.)))))) * 2.0)) * 2.0)) - ((((np.tanh((data["freq_V127"]))) <= (data["M6"]))*1.)))) * 2.0)) * 2.0)) * 2.0)) +

                    0.091120*np.tanh((((((((((((((((((((data["V294"]) <= (((((data["R_emaildomain"]) / 2.0)) / 2.0)))*1.)) * 2.0)) - (((data["freq_C11"]) - ((((data["freq_C13"]) > (data["freq_C6"]))*1.)))))) * 2.0)) + (((((((data["card1"]) * 2.0)) * 2.0)) - (data["freq_V45"]))))) * 2.0)) * 2.0)) - (data["freq_C11"]))) * 2.0)) +

                    0.082080*np.tanh(((((((((((((((((data["C13"]) * 2.0)) - (((data["freq_V83"]) - (((data["card1"]) + (((((((data["V315"]) + ((((data["freq_C13"]) > (data["freq_C11"]))*1.)))) * 2.0)) * 2.0)))))))) * 2.0)) * 2.0)) + (data["freq_V83"]))) + (((data["C13"]) - (data["freq_C11"]))))) * 2.0)) * 2.0)) +

                    0.098480*np.tanh((-1.0*((((((((((((((((data["freq_V312"]) - (((((((data["V133"]) <= (data["D4"]))*1.)) + (((data["freq_D3"]) * 2.0)))/2.0)))) - ((((((data["freq_C8"]) <= ((((((data["D4"]) / 2.0)) + (data["freq_C8"]))/2.0)))*1.)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0))))) +

                    0.070080*np.tanh(((((((((((((((((((data["V315"]) + (data["V310"]))) * 2.0)) * 2.0)) + ((((((data["C13"]) > (data["freq_C1"]))*1.)) + (((data["freq_P_emaildomain"]) + (np.tanh((((data["card1"]) - (((((9.0)) > (data["freq_P_emaildomain"]))*1.)))))))))))) * 2.0)) + (data["V313"]))) * 2.0)) * 2.0)) * 2.0)) +

                    0.100000*np.tanh(((((data["freq_V307"]) + (((((((data["freq_C1"]) + (((((((((((data["D15"]) + (((data["V315"]) + ((((data["freq_C1"]) <= (data["C13"]))*1.)))))) * 2.0)) - (((data["freq_M5"]) - (((data["card1"]) + (data["C13"]))))))) * 2.0)) * 2.0)))) - (data["freq_C6"]))) * 2.0)))) * 2.0)) +

                    0.099720*np.tanh(((((((((((data["V315"]) * 2.0)) * 2.0)) * 2.0)) - (data["month"]))) + ((((((((((((data["card1"]) / 2.0)) > ((((data["freq_C8"]) > ((((data["V294"]) <= (((data["dist1"]) + (data["card1"]))))*1.)))*1.)))*1.)) * ((((8.0)) - (data["month"]))))) * 2.0)) * 2.0)))) +

                    0.040128*np.tanh((((((-1.0*((data["freq_V130"])))) + (((data["card1"]) + (((((((((data["card1"]) + (((((((((-1.0*((((data["freq_V130"]) / 2.0))))) + (((data["freq_V130"]) * (data["freq_P_emaildomain"]))))/2.0)) + ((((data["freq_V130"]) > (((data["freq_V45"]) * 2.0)))*1.)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)))))) * 2.0)) +

                    0.029760*np.tanh(((((((data["V315"]) + (((data["V130"]) + ((-1.0*((((data["freq_M5"]) * (((((data["freq_C10"]) + ((-1.0*((((data["card1"]) * 2.0))))))) / 2.0))))))))))) * ((((data["TransactionAmt"]) + (((((data["card1"]) * 2.0)) * 2.0)))/2.0)))) * 2.0)) +

                    0.089504*np.tanh(((data["year"]) * (((data["V315"]) - ((-1.0*(((((((((data["freq_C13"]) > (data["freq_C1"]))*1.)) + (((((data["V315"]) + (data["R_emaildomain"]))) * 2.0)))) - ((((((((data["freq_C13"]) - ((((((((data["C8"]) <= (data["V283"]))*1.)) / 2.0)) / 2.0)))) <= (data["C10"]))*1.)) / 2.0))))))))))) +

                    0.097002*np.tanh(((((((((((((data["card1"]) + ((((((data["card4"]) <= ((((data["freq_C13"]) + (data["D8"]))/2.0)))*1.)) * ((((((data["V315"]) <= (data["C1"]))*1.)) * 2.0)))))) * 2.0)) - ((((data["C1"]) <= (data["freq_D9"]))*1.)))) + ((((data["M6"]) <= (data["C14"]))*1.)))) * 2.0)) * 2.0)) +

                    0.095216*np.tanh((((((((((((data["freq_P_emaildomain"]) > (((data["freq_C10"]) + (((data["freq_C1"]) - (data["V83"]))))))*1.)) - (((3.0) * ((((data["freq_C14"]) <= ((((((((((data["freq_C13"]) <= (((data["freq_C1"]) + (data["freq_C1"]))))*1.)) / 2.0)) / 2.0)) * (data["freq_V45"]))))*1.)))))) * 2.0)) * 2.0)) * 2.0)) +

                    0.099760*np.tanh(((((((((((((((((data["V315"]) + (data["card1"]))) * 2.0)) - ((((data["M6"]) > ((((data["M6"]) <= (data["V310"]))*1.)))*1.)))) + (((((((((data["R_emaildomain"]) > (data["M5"]))*1.)) > (data["freq_V294"]))*1.)) * 2.0)))) * 2.0)) * 2.0)) - ((((data["D9"]) <= (data["C1"]))*1.)))) * 2.0)) +

                    0.094800*np.tanh(((((((((((((((((((((data["card1"]) + ((((((((data["freq_C13"]) > (((((data["freq_C1"]) * 2.0)) * 2.0)))*1.)) * 2.0)) - ((((data["freq_V83"]) + (data["freq_C13"]))/2.0)))))) + (data["V313"]))) * 2.0)) + (data["D8"]))) + (data["D8"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.100000*np.tanh((((((((((((((((data["M6"]) <= ((((((data["C1"]) > (((data["C14"]) * 2.0)))*1.)) + (data["C11"]))))*1.)) * 2.0)) + (((data["V45"]) - ((((data["C14"]) <= (data["V130"]))*1.)))))) * 2.0)) * 2.0)) + ((1.0)))) * 2.0)) +

                    0.098037*np.tanh(((data["V45"]) - ((((((((data["C13"]) <= (((((((data["V283"]) > (data["D1"]))*1.)) > (data["freq_V307"]))*1.)))*1.)) * 2.0)) - (((((((((np.tanh(((((data["freq_V294"]) <= (((data["card1"]) - (data["D8"]))))*1.)))) + (data["D8"]))) * 2.0)) * 2.0)) * 2.0)))))) +

                    0.087610*np.tanh(((3.0) * (((3.0) * (((((((data["V313"]) + (((data["freq_V314"]) * ((((((data["V313"]) <= (((data["D8"]) * (data["freq_V308"]))))*1.)) + ((((data["freq_V45"]) <= (data["C8"]))*1.)))))))) + (((data["V313"]) - ((((data["freq_C9"]) + (data["V45"]))/2.0)))))) * 2.0)))))) +

                    0.091146*np.tanh((((((((((((((data["C14"]) > ((((data["V283"]) > (data["dist1"]))*1.)))*1.)) > ((((data["V283"]) > (data["C10"]))*1.)))*1.)) + ((((((((((data["freq_C13"]) > (data["freq_C6"]))*1.)) * 2.0)) - ((((data["freq_D14"]) > ((((data["C14"]) > (data["C10"]))*1.)))*1.)))) * 2.0)))/2.0)) * 2.0)) * 2.0)) +

                    0.099333*np.tanh((((((((((((data["freq_D14"]) <= (((data["V45"]) * 2.0)))*1.)) * 2.0)) - (data["freq_D9"]))) + (((data["freq_M4"]) * (((((((data["dist1"]) * 2.0)) * (((((data["TransactionAmt"]) - (data["dist1"]))) / 2.0)))) * (((data["freq_C14"]) + (data["freq_P_emaildomain"]))))))))) * 2.0)) +

                    0.094016*np.tanh(((((((((((((data["freq_P_emaildomain"]) + (((data["V45"]) - ((((data["V315"]) <= (((data["V307"]) * ((((data["card1"]) <= (data["V283"]))*1.)))))*1.)))))) * 2.0)) - ((((data["C14"]) <= (((data["freq_C1"]) * (data["freq_V310"]))))*1.)))) + (data["D4"]))) * 2.0)) * 2.0)) +

                    0.098112*np.tanh(((((((((data["freq_D3"]) + (((((data["M5"]) + ((((((((data["V307"]) <= (data["freq_C14"]))*1.)) * (((data["freq_D3"]) * 2.0)))) * 2.0)))) - ((((((data["V310"]) + (((data["V315"]) - (data["freq_D3"]))))) <= (((data["freq_V317"]) * (data["freq_M6"]))))*1.)))))) * 2.0)) * 2.0)) * 2.0)) +

                    0.095126*np.tanh(((((((data["freq_V310"]) * 2.0)) * 2.0)) + (((((((((((((data["card1"]) + ((((((data["freq_C13"]) > (data["freq_C11"]))*1.)) * 2.0)))) - (((((((data["card1"]) > (data["V315"]))*1.)) > ((((data["freq_V310"]) <= (((data["V45"]) * 2.0)))*1.)))*1.)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +

                    0.099832*np.tanh(((((((((((data["V45"]) + (((data["D14"]) + ((((-1.0*((data["freq_M5"])))) + ((((((((data["freq_C1"]) <= (((((data["freq_C14"]) / 2.0)) / 2.0)))*1.)) + (data["R_emaildomain"]))) + (((data["D8"]) + (data["R_emaildomain"]))))))))))) * 2.0)) * 2.0)) + (data["freq_C1"]))) * 2.0)) +

                    0.091960*np.tanh((((((data["card1"]) + (data["freq_V317"]))/2.0)) + (((((data["V45"]) + (((((((((data["card1"]) - (((data["V308"]) - (((data["C11"]) * 2.0)))))) * 2.0)) - ((((data["V45"]) > (((data["M5"]) - ((((data["V283"]) > (data["freq_C14"]))*1.)))))*1.)))) * 2.0)))) * 2.0)))) +

                    0.074880*np.tanh((((((((((data["V45"]) > ((((data["C9"]) + ((((data["freq_P_emaildomain"]) > (((((((data["freq_C2"]) <= (((data["freq_D3"]) * ((((data["V45"]) <= (data["D10"]))*1.)))))*1.)) > (data["card1"]))*1.)))*1.)))/2.0)))*1.)) * 2.0)) - ((((data["freq_dist1"]) <= (np.tanh(((((data["freq_dist1"]) > (data["card1"]))*1.)))))*1.)))) * 2.0)) +

                    0.094544*np.tanh(((((((((((((data["C13"]) + ((((data["C12"]) + (((data["V315"]) - (((((((data["C14"]) <= (data["C5"]))*1.)) + ((((data["C12"]) > (data["dist1"]))*1.)))/2.0)))))/2.0)))) + ((((data["V45"]) + (((data["card1"]) + (data["freq_card5"]))))/2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.093200*np.tanh(((((((data["D2"]) - (((data["freq_V83"]) - ((((((((data["freq_D14"]) + (data["freq_D2"]))) <= ((((((data["card2"]) > (data["V283"]))*1.)) + ((((((data["freq_V83"]) <= (data["R_emaildomain"]))*1.)) * 2.0)))))*1.)) * 2.0)))))) * 2.0)) - (data["freq_D14"]))) +

                    0.088547*np.tanh(((((((((((((((data["D14"]) * (data["D4"]))) * (data["TransactionAmt"]))) * 2.0)) + (((data["freq_C14"]) - ((((((data["M6"]) * ((((((((data["freq_C14"]) * (((data["C6"]) * 2.0)))) <= (data["freq_C1"]))*1.)) * 2.0)))) > (((data["dist1"]) * 2.0)))*1.)))))) * 2.0)) * 2.0)) * 2.0)) +

                    0.092976*np.tanh((((((((((((((((((((data["freq_dist1"]) <= (data["V315"]))*1.)) + (((data["freq_dist2"]) - ((((data["freq_C8"]) > (((data["freq_V294"]) - ((((data["freq_C8"]) > (data["freq_D4"]))*1.)))))*1.)))))) * 2.0)) * 2.0)) + (data["V315"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.099495*np.tanh((((((((-1.0*(((((data["M4"]) > (((((data["card2"]) * 2.0)) * 2.0)))*1.))))) + (((((((data["card1"]) + (((data["V130"]) * 2.0)))) * 2.0)) + ((-1.0*(((((data["V127"]) > (((data["D15"]) * 2.0)))*1.))))))))) * 2.0)) + (data["V130"]))) +

                    0.099480*np.tanh((((((((((((((data["R_emaildomain"]) / 2.0)) > (data["C12"]))*1.)) * 2.0)) * 2.0)) + ((((((((data["M6"]) > (data["R_emaildomain"]))*1.)) - ((((data["M6"]) > (((((data["freq_card1"]) * 2.0)) * 2.0)))*1.)))) * 2.0)))) + ((-1.0*(((((data["M6"]) > (((((data["freq_card1"]) * 2.0)) * 2.0)))*1.))))))) +

                    0.097800*np.tanh(((((data["card1"]) + (((((((data["C10"]) + (((data["V45"]) + (((data["card1"]) + (((data["D9"]) + (((((((((data["V310"]) > (data["V317"]))*1.)) - (data["freq_C5"]))) + (((data["V317"]) - ((((data["freq_M5"]) > (data["freq_M4"]))*1.)))))/2.0)))))))))) * 2.0)) * 2.0)))) * 2.0)) +

                    0.099961*np.tanh(((((((((data["C6"]) * 2.0)) + (((((data["C10"]) * 2.0)) - ((-1.0*((((data["D4"]) - ((((data["card2"]) <= (((((((data["D4"]) + ((((data["V314"]) > (data["freq_C14"]))*1.)))/2.0)) + ((((data["D11"]) <= ((((data["freq_V127"]) > (data["freq_dist2"]))*1.)))*1.)))/2.0)))*1.))))))))))) * 2.0)) * 2.0)) +

                    0.098576*np.tanh(((data["month"]) * (((data["card1"]) + (((((data["dist1"]) + (((data["freq_D1"]) - ((((((data["freq_D1"]) - (data["card2"]))) > (((((((data["freq_D14"]) + ((-1.0*((((data["dist2"]) - (data["freq_D14"])))))))/2.0)) <= ((((data["C11"]) > (data["card1"]))*1.)))*1.)))*1.)))))) * 2.0)))))) +

                    0.099430*np.tanh((((((((((((data["C1"]) <= ((((((-1.0*((((data["freq_C1"]) - (data["C13"])))))) * 2.0)) * 2.0)))*1.)) * 2.0)) * 2.0)) + (((data["C11"]) - ((((data["freq_V313"]) > (data["freq_V315"]))*1.)))))) + ((((((data["C8"]) <= (data["M6"]))*1.)) + (((data["freq_C1"]) - (data["freq_V313"]))))))) +

                    0.099755*np.tanh(((((data["V130"]) + (((np.tanh((data["D8"]))) * (((data["TransactionAmt"]) * (((((data["D8"]) + (((data["D14"]) * (((data["TransactionAmt"]) * (data["D1"]))))))) - ((((data["freq_C9"]) > (((data["freq_V308"]) + (data["dist1"]))))*1.)))))))))) - ((((data["freq_V308"]) > (data["freq_C1"]))*1.)))) +

                    0.099872*np.tanh(((data["V45"]) + (((data["V45"]) + ((((((((((((((data["C6"]) + (data["card1"]))/2.0)) * 2.0)) * 2.0)) - (((data["D5"]) - (((data["V310"]) + (((((data["card1"]) - (data["freq_M5"]))) + (((data["freq_C14"]) * (((data["freq_V317"]) * 2.0)))))))))))) * 2.0)) * 2.0)))))) +

                    0.099995*np.tanh((((((data["D8"]) <= (data["dist2"]))*1.)) + (((((data["D8"]) - ((((data["V313"]) <= (((((((data["card5"]) <= (((((((data["dist2"]) + ((((data["dist2"]) + (data["D8"]))/2.0)))/2.0)) + ((((data["D8"]) <= ((((data["C12"]) + (data["M5"]))/2.0)))*1.)))/2.0)))*1.)) <= (data["D8"]))*1.)))*1.)))) * 2.0)))) +

                    0.099165*np.tanh(((data["V314"]) + (((data["card2"]) - ((((((((((data["card2"]) > (((data["V314"]) + ((((data["V314"]) > ((((data["V314"]) > (data["D4"]))*1.)))*1.)))))*1.)) * 2.0)) * 2.0)) + (((data["freq_D15"]) - ((((data["C8"]) > (((data["month"]) * (data["dist2"]))))*1.)))))))))) +

                    0.099800*np.tanh(((data["dist1"]) - (((data["freq_C11"]) - (((((data["C10"]) - ((((((data["C12"]) > (data["dist1"]))*1.)) - ((((((((data["C10"]) * 2.0)) + (((((data["freq_dist2"]) - ((((data["V45"]) > ((((((data["freq_card1"]) * 2.0)) + (data["dist1"]))/2.0)))*1.)))) * 2.0)))/2.0)) * 2.0)))))) * 2.0)))))) +

                    0.003000*np.tanh(((((((((((data["freq_C2"]) - (data["freq_C12"]))) - ((-1.0*(((((((data["freq_C9"]) <= (data["R_emaildomain"]))*1.)) * 2.0))))))) * 2.0)) / 2.0)) + ((((((data["freq_C11"]) <= (((data["freq_C12"]) * ((((data["freq_C2"]) <= (np.tanh((np.tanh((((data["C10"]) + (data["card1"]))))))))*1.)))))*1.)) * 2.0)))) +

                    0.070179*np.tanh((((((((((((((((data["freq_C9"]) <= (((data["D8"]) * (data["weekday"]))))*1.)) - (data["freq_C9"]))) - (((data["C14"]) * (data["weekday"]))))) + ((((data["C5"]) <= ((((data["C14"]) + ((((data["V294"]) + (data["C6"]))/2.0)))/2.0)))*1.)))) * 2.0)) * 2.0)) * 2.0)) +

                    0.0*np.tanh(((((data["dist1"]) + ((((((((((((((((data["D14"]) * 2.0)) + ((((data["R_emaildomain"]) > (((data["C12"]) * 2.0)))*1.)))) - (((data["freq_dist1"]) - (data["V313"]))))) + ((((data["D9"]) <= (((data["V130"]) / 2.0)))*1.)))/2.0)) * 2.0)) + (data["dist1"]))) * 2.0)))) * 2.0)) +

                    0.099986*np.tanh(((data["C10"]) + ((((((((data["C12"]) <= (data["addr1"]))*1.)) + (((((data["V315"]) + (((data["C10"]) + ((((data["C8"]) > (data["C10"]))*1.)))))) - ((((((data["D8"]) + (((data["V130"]) + (data["D14"]))))) <= (((data["freq_dist1"]) / 2.0)))*1.)))))) * 2.0)))) +

                    0.099800*np.tanh((((((data["freq_addr1"]) > (data["C10"]))*1.)) - (((-3.0) * ((((((data["C13"]) > ((((((data["card2"]) + ((((data["C13"]) <= ((((data["V313"]) > (data["M4"]))*1.)))*1.)))) + ((((data["C10"]) > (data["card2"]))*1.)))/2.0)))*1.)) - ((((data["freq_addr1"]) > (data["C10"]))*1.)))))))) +

                    0.096984*np.tanh(((((data["V315"]) - ((((data["D1"]) <= ((((data["C14"]) <= (data["C1"]))*1.)))*1.)))) + (((((((((((((data["V315"]) * 2.0)) + (data["V45"]))) + ((((data["freq_V83"]) <= (data["C10"]))*1.)))) * 2.0)) * 2.0)) - ((((data["C14"]) <= (data["C1"]))*1.)))))) +

                    0.099600*np.tanh((((((data["freq_V312"]) > (((((data["freq_V45"]) + (((((data["freq_C2"]) - (((data["M4"]) * 2.0)))) - (((data["freq_V45"]) * ((((((((((data["M4"]) <= (((data["V312"]) / 2.0)))*1.)) * 2.0)) + (((data["freq_C2"]) - (((data["M4"]) / 2.0)))))) * 2.0)))))))) * 2.0)))*1.)) * 2.0)) +

                    0.000700*np.tanh((((((((((data["card1"]) > (data["D4"]))*1.)) * ((((((-1.0*(((((data["freq_D1"]) + (((((((data["freq_V307"]) - (((data["C10"]) - ((((data["freq_D14"]) <= ((((data["freq_D14"]) <= (((data["freq_C11"]) / 2.0)))*1.)))*1.)))))) - (data["C10"]))) * 2.0)))/2.0))))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +

                    0.099968*np.tanh((((((((data["freq_V127"]) <= (((data["V130"]) * (data["card2"]))))*1.)) - ((((((data["card2"]) * (data["V310"]))) <= ((((data["D11"]) <= (np.tanh(((((data["V130"]) <= ((((data["V315"]) <= ((((data["D1"]) <= (((data["card2"]) * (data["freq_V127"]))))*1.)))*1.)))*1.)))))*1.)))*1.)))) * 2.0)) +

                    0.099956*np.tanh((((((data["freq_C9"]) <= (((data["freq_C13"]) * 2.0)))*1.)) - ((((((((np.tanh((data["dist1"]))) - ((((data["D1"]) <= (((data["M6"]) / 2.0)))*1.)))) <= ((((data["V294"]) <= (((data["freq_C2"]) * (((((data["freq_C9"]) * ((((data["V315"]) > (data["M4"]))*1.)))) / 2.0)))))*1.)))*1.)) * 2.0)))) +

                    0.059594*np.tanh((((((((data["freq_V314"]) <= (((data["freq_V317"]) * (data["C14"]))))*1.)) + (np.tanh((((((((((((data["V317"]) * 2.0)) + (((data["V45"]) * 2.0)))) - ((((data["V45"]) > (data["V130"]))*1.)))) * 2.0)) - ((((data["V45"]) <= ((((data["freq_D1"]) > (data["freq_D11"]))*1.)))*1.)))))))) * 2.0)) +

                    0.099994*np.tanh(((((((((((((((((data["dist1"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * (data["freq_R_emaildomain"]))) - (((((((((data["freq_P_emaildomain"]) + ((-1.0*((data["card1"])))))) > ((((data["M5"]) > (data["card4"]))*1.)))*1.)) > (data["freq_V283"]))*1.)))) * 2.0)) * 2.0)) +

                    0.002968*np.tanh((((((((((((data["C2"]) <= (((data["C6"]) / 2.0)))*1.)) + (((((((data["freq_D3"]) + (data["V315"]))) * 2.0)) * ((((data["C2"]) > (data["C6"]))*1.)))))) * 2.0)) - ((((data["freq_dist1"]) > (data["freq_D3"]))*1.)))) * 2.0)) +

                    0.076400*np.tanh(((((((((((((data["dist2"]) - ((((data["M6"]) <= (((data["M5"]) / 2.0)))*1.)))) * 2.0)) - ((((data["D4"]) <= ((((data["D15"]) > (((data["P_emaildomain"]) * 2.0)))*1.)))*1.)))) * 2.0)) * 2.0)) + ((((((data["D4"]) > (data["P_emaildomain"]))*1.)) - ((((data["dist2"]) > (data["M5"]))*1.)))))) +

                    0.055072*np.tanh(((((data["C8"]) * 2.0)) - (((((((((data["freq_D10"]) + (((data["C12"]) * 2.0)))/2.0)) > (((data["freq_C14"]) * 2.0)))*1.)) + ((((((data["C12"]) > (((((data["dist1"]) * 2.0)) * 2.0)))*1.)) - ((((((((data["C10"]) > (((data["C8"]) * 2.0)))*1.)) * 2.0)) * 2.0)))))))) +

                    0.000001*np.tanh(((data["V313"]) + (((((data["freq_C10"]) - (data["freq_card4"]))) * (((((((((((data["freq_V45"]) - (((((((data["C9"]) <= (((data["freq_P_emaildomain"]) - (data["dist1"]))))*1.)) + (((((data["C10"]) * (data["freq_V45"]))) * 2.0)))/2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))))) +

                    0.099902*np.tanh((((((((((((((((((data["freq_C9"]) - ((-1.0*((data["freq_addr1"])))))) <= (((data["V310"]) * 2.0)))*1.)) + (((data["V317"]) - ((((np.tanh((((data["freq_addr1"]) - (data["dist2"]))))) > (data["freq_C14"]))*1.)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - ((((data["V294"]) > (data["V83"]))*1.)))) +

                    0.061328*np.tanh((((((data["V127"]) <= (((((data["freq_D15"]) * (data["C11"]))) * 2.0)))*1.)) + (((-2.0) * (((((data["freq_card4"]) - ((((((data["V127"]) * ((((data["freq_C12"]) > (data["freq_TransactionAmt"]))*1.)))) <= (((data["dist1"]) / 2.0)))*1.)))) - ((((data["card1"]) <= (((data["V130"]) / 2.0)))*1.)))))))) +

                    0.050323*np.tanh((((((((((((((((((((data["C11"]) + (((data["C6"]) / 2.0)))) + (data["C14"]))/2.0)) + (data["D1"]))/2.0)) > (((data["V127"]) * 2.0)))*1.)) - (((data["freq_dist1"]) / 2.0)))) * 2.0)) * 2.0)) + ((((((data["V310"]) > (((data["freq_C9"]) + (np.tanh((data["P_emaildomain"]))))))*1.)) * 2.0)))) +

                    0.098005*np.tanh((((((data["C8"]) <= (((((data["freq_card1"]) * ((((((((data["card2"]) > (data["C8"]))*1.)) * 2.0)) * 2.0)))) * 2.0)))*1.)) - ((((((data["freq_V308"]) + ((((((data["card2"]) > (data["D8"]))*1.)) - (data["D8"]))))) <= (((((data["freq_card1"]) * (data["freq_V308"]))) - (data["freq_card1"]))))*1.)))) +

                    0.099920*np.tanh((((((((((((((data["dist2"]) + (data["dist2"]))) * 2.0)) > ((((((((data["D9"]) * 2.0)) <= (((data["freq_M5"]) - ((((((data["freq_M5"]) - (data["freq_C2"]))) + (data["card1"]))/2.0)))))*1.)) / 2.0)))*1.)) - ((((data["C1"]) > (((data["R_emaildomain"]) * 2.0)))*1.)))) * 2.0)) * 2.0)) +

                    0.051520*np.tanh(((((((((data["C10"]) - (data["freq_D10"]))) + (((data["C10"]) + ((((data["C10"]) <= ((((((data["M4"]) + (((((data["P_emaildomain"]) + (((data["C6"]) - (data["C10"]))))) / 2.0)))/2.0)) / 2.0)))*1.)))))) * 2.0)) - ((((((data["M6"]) * 2.0)) <= (data["M5"]))*1.)))) +

                    0.075200*np.tanh((((((((((data["V283"]) <= ((((((data["freq_card2"]) + (data["freq_card1"]))/2.0)) / 2.0)))*1.)) - ((((data["V312"]) <= (((((((((data["M6"]) / 2.0)) <= (data["D4"]))*1.)) <= (data["freq_card1"]))*1.)))*1.)))) + (((((((data["dist1"]) * 2.0)) * 2.0)) - ((((data["freq_V83"]) <= (data["C6"]))*1.)))))) * 2.0)) +

                    0.000005*np.tanh((((((((((((((data["C5"]) > ((((data["freq_C2"]) + ((((data["C12"]) > ((((data["C12"]) <= (data["dist1"]))*1.)))*1.)))/2.0)))*1.)) * 2.0)) - ((((data["dist2"]) <= (data["card1"]))*1.)))) * 2.0)) - (data["C5"]))) + (((((((data["dist1"]) + (data["C5"]))/2.0)) > (data["V314"]))*1.)))) +

                    0.099900*np.tanh((((((((((data["V313"]) * (((data["card1"]) + (data["freq_V294"]))))) > (data["freq_V294"]))*1.)) + (((data["C8"]) - ((((((data["V313"]) * (((data["card1"]) + (data["card1"]))))) <= ((((((data["C8"]) / 2.0)) > ((((((data["C8"]) / 2.0)) <= (data["R_emaildomain"]))*1.)))*1.)))*1.)))))) * 2.0)) +

                    0.090760*np.tanh(((((data["C6"]) - ((((((data["freq_C2"]) <= ((((data["C2"]) + (((data["C5"]) - ((((((data["C1"]) + (data["C1"]))/2.0)) * 2.0)))))/2.0)))*1.)) * 2.0)))) + (((((((data["M4"]) <= (((data["C2"]) - ((((data["C2"]) <= (data["C6"]))*1.)))))*1.)) > (data["C5"]))*1.)))) +

                    0.099000*np.tanh((((((((data["C11"]) <= (((data["C6"]) / 2.0)))*1.)) * 2.0)) - ((((((data["freq_C6"]) <= (data["freq_D4"]))*1.)) - (((data["V313"]) - ((((((data["R_emaildomain"]) <= (((data["M4"]) - (((data["V313"]) / 2.0)))))*1.)) - ((((data["C11"]) > (((data["P_emaildomain"]) + (data["card5"]))))*1.)))))))))) +

                    0.002989*np.tanh((((((((data["freq_C11"]) <= (((((((((((-1.0*(((((data["freq_D10"]) <= ((((data["V83"]) + ((((data["V83"]) <= (((((((data["D15"]) > (data["V83"]))*1.)) <= (data["freq_C11"]))*1.)))*1.)))/2.0)))*1.))))) + (((data["R_emaildomain"]) - (data["M5"]))))/2.0)) * 2.0)) * 2.0)) * 2.0)))*1.)) * 2.0)) * 2.0)) +

                    0.068810*np.tanh(((((((data["freq_D3"]) + ((-1.0*(((((data["freq_D3"]) <= (((data["C6"]) - (data["freq_C5"]))))*1.))))))) + (((((data["freq_card4"]) - ((((data["freq_D3"]) > (((data["freq_card5"]) + (data["freq_D15"]))))*1.)))) * ((((((((data["C5"]) <= (data["C6"]))*1.)) - (data["freq_card5"]))) * 2.0)))))) * 2.0)) +

                    0.070064*np.tanh(((data["dist2"]) + (((((data["dist2"]) + (((((data["card2"]) - ((((((data["V83"]) <= ((-1.0*((((data["card4"]) + (((data["card2"]) * ((-1.0*(((((((data["freq_C11"]) > (data["card2"]))*1.)) + ((-1.0*((data["dist2"])))))))))))))))))*1.)) * 2.0)))) * 2.0)))) * 2.0)))) +

                    0.099845*np.tanh((((data["R_emaildomain"]) <= ((((((np.tanh((((((data["M4"]) * 2.0)) / 2.0)))) * (((data["freq_C6"]) + (data["R_emaildomain"]))))) + ((((data["C10"]) > ((((data["C14"]) <= ((((data["freq_C2"]) + (np.tanh(((((data["freq_V45"]) + (((data["freq_V127"]) * (((data["C12"]) * 2.0)))))/2.0)))))/2.0)))*1.)))*1.)))/2.0)))*1.)) +

                    0.065248*np.tanh((((((((((data["C6"]) > (((((data["C2"]) - ((((((data["freq_V294"]) - (((data["card1"]) - (((data["C12"]) * 2.0)))))) <= ((((data["card1"]) <= (((((data["M6"]) - (data["C6"]))) / 2.0)))*1.)))*1.)))) * 2.0)))*1.)) * 2.0)) - (data["C6"]))) * 2.0)) +

                    0.033760*np.tanh(((((data["C12"]) + ((((((data["C12"]) <= (((data["D14"]) / 2.0)))*1.)) + ((((((data["C12"]) <= (((data["dist1"]) / 2.0)))*1.)) - ((((data["freq_D8"]) <= ((((data["dist1"]) <= (((np.tanh((data["freq_R_emaildomain"]))) * (data["C14"]))))*1.)))*1.)))))))) - ((((data["freq_D8"]) <= (data["freq_card1"]))*1.)))) +

                    0.099080*np.tanh((((((((((data["dist1"]) > (((data["M5"]) + (data["freq_dist1"]))))*1.)) * 2.0)) - ((((data["dist2"]) <= (((((((data["V315"]) + (np.tanh((data["V310"]))))/2.0)) <= (((data["D3"]) - (data["freq_V294"]))))*1.)))*1.)))) - ((-1.0*(((((data["V307"]) <= (((data["D3"]) / 2.0)))*1.))))))) +

                    0.099890*np.tanh(((((data["D14"]) + ((-1.0*(((((data["V310"]) <= ((((data["freq_card2"]) <= (((((data["C11"]) - (data["freq_V310"]))) - (data["D15"]))))*1.)))*1.))))))) + (((data["P_emaildomain"]) + ((-1.0*(((((data["freq_D8"]) <= (((data["V315"]) - (data["V294"]))))*1.))))))))) +

                    0.097880*np.tanh((((((((((data["C10"]) - ((((data["C10"]) > (((data["card1"]) + (((((data["C10"]) * 2.0)) * (data["C8"]))))))*1.)))) / 2.0)) * 2.0)) > (((((((data["freq_C9"]) > (((((data["freq_C1"]) / 2.0)) / 2.0)))*1.)) > ((((data["C8"]) > (data["C10"]))*1.)))*1.)))*1.)) +

                    0.094566*np.tanh((((((data["freq_V310"]) <= (((data["freq_D3"]) - ((((data["D8"]) > (data["freq_D3"]))*1.)))))*1.)) - ((((((((data["V315"]) <= ((((data["addr1"]) <= (((data["D14"]) + ((((((data["D8"]) <= ((((data["D8"]) > (np.tanh((data["freq_D3"]))))*1.)))*1.)) - (data["freq_D2"]))))))*1.)))*1.)) * 2.0)) * 2.0)))) +

                    0.081200*np.tanh(((((((((((((((data["D3"]) <= ((((data["V130"]) <= (((data["D3"]) * 2.0)))*1.)))*1.)) * (data["D14"]))) <= ((((data["freq_C13"]) <= ((((data["freq_C11"]) <= (((((((data["freq_C13"]) * (data["freq_addr1"]))) * 2.0)) * 2.0)))*1.)))*1.)))*1.)) - (((data["freq_D15"]) - (data["D14"]))))) * 2.0)) * 2.0)) +

                    0.020803*np.tanh(((((((data["V315"]) * 2.0)) - (((data["freq_D2"]) - (((data["D3"]) - ((((((data["V314"]) * ((((data["freq_C6"]) > (data["freq_D2"]))*1.)))) > (data["D3"]))*1.)))))))) + ((((data["freq_M4"]) <= ((((((((((data["freq_V312"]) + (data["R_emaildomain"]))/2.0)) > (data["freq_M5"]))*1.)) + (data["freq_D2"]))/2.0)))*1.)))) +

                    0.091000*np.tanh(((((data["freq_V312"]) - ((((((data["M4"]) * (data["freq_C2"]))) <= (data["V307"]))*1.)))) + ((((((data["freq_P_emaildomain"]) <= (((((((((data["freq_V307"]) <= (data["freq_D2"]))*1.)) * (data["freq_V308"]))) + (((data["P_emaildomain"]) - (((data["M4"]) * (data["freq_C2"]))))))/2.0)))*1.)) * 2.0)))) +

                    0.098643*np.tanh(((((((((data["freq_V317"]) * ((((data["freq_V312"]) <= (data["freq_V130"]))*1.)))) * 2.0)) + ((-1.0*(((((data["M6"]) <= (data["C12"]))*1.))))))) + ((((data["freq_V83"]) > ((((data["freq_dist1"]) > (((((data["freq_D5"]) + (data["V310"]))) * ((((data["V45"]) <= (data["dist1"]))*1.)))))*1.)))*1.)))) +

                    0.099560*np.tanh(((((((((((data["freq_V45"]) * ((((data["P_emaildomain"]) <= (data["V130"]))*1.)))) - ((((data["freq_R_emaildomain"]) > ((((((-1.0*((data["freq_V307"])))) + (((data["month"]) + ((-1.0*((data["V83"])))))))) / 2.0)))*1.)))) * 2.0)) - ((((((data["V83"]) / 2.0)) > (np.tanh((data["P_emaildomain"]))))*1.)))) * 2.0)) +

                    0.026800*np.tanh(((((((((((((data["month"]) - (data["freq_V314"]))) <= (((((np.tanh((((data["freq_D2"]) + (data["card1"]))))) * 2.0)) * 2.0)))*1.)) + ((((np.tanh((data["dist2"]))) > (((data["freq_D2"]) + ((((data["C9"]) <= (data["C11"]))*1.)))))*1.)))/2.0)) * 2.0)) * 2.0)) +

                    0.099930*np.tanh((((((((data["freq_D1"]) <= (((((((data["V310"]) * ((((((data["card1"]) + (np.tanh(((((-1.0*((data["freq_V283"])))) / 2.0)))))/2.0)) * 2.0)))) * 2.0)) * 2.0)))*1.)) * 2.0)) - ((((data["card4"]) > ((((((data["V310"]) <= (data["freq_V283"]))*1.)) + ((((data["V310"]) <= (data["freq_V283"]))*1.)))))*1.)))) +

                    0.007200*np.tanh((((data["freq_C12"]) <= ((((((((((np.tanh((data["freq_card1"]))) <= (data["freq_V130"]))*1.)) <= ((((((data["freq_V294"]) + ((((data["freq_D3"]) + (((data["freq_TransactionAmt"]) / 2.0)))/2.0)))) <= ((((data["V312"]) + ((((data["card1"]) + (data["V312"]))/2.0)))/2.0)))*1.)))*1.)) + ((((data["freq_card1"]) <= (np.tanh((data["freq_TransactionAmt"]))))*1.)))/2.0)))*1.)) +

                    0.084080*np.tanh(((((data["freq_C13"]) + ((((((data["C2"]) > (data["card2"]))*1.)) - ((((data["freq_D5"]) <= ((((data["D9"]) <= ((((data["card2"]) + ((((data["V315"]) <= ((((((data["card2"]) <= ((((data["D9"]) + ((((data["freq_C13"]) > (data["freq_D5"]))*1.)))/2.0)))*1.)) / 2.0)))*1.)))/2.0)))*1.)))*1.)))))) * 2.0)) +

                    0.099922*np.tanh((((((data["freq_C11"]) <= (data["D11"]))*1.)) + (((((data["V313"]) * 2.0)) - ((((((((data["D2"]) <= ((((data["V313"]) <= ((((data["V312"]) <= ((((data["M5"]) > ((((data["C13"]) + (((data["freq_C11"]) + (data["D2"]))))/2.0)))*1.)))*1.)))*1.)))*1.)) * 2.0)) * 2.0)))))) +

                    0.098120*np.tanh(((((((((((data["dist2"]) - ((((data["dist2"]) <= ((((data["V314"]) + ((((data["V317"]) <= ((((data["V133"]) + ((((data["V133"]) + (((((((data["M5"]) <= ((((np.tanh((data["card1"]))) + (data["freq_C8"]))/2.0)))*1.)) <= (data["M5"]))*1.)))/2.0)))/2.0)))*1.)))/2.0)))*1.)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.099188*np.tanh(((data["freq_C1"]) - (((((((data["freq_dist1"]) > (((data["R_emaildomain"]) / 2.0)))*1.)) <= ((((((data["card1"]) * (data["card2"]))) <= ((((data["V310"]) <= ((((data["freq_P_emaildomain"]) <= ((((data["freq_C1"]) + (((data["freq_D1"]) * ((((data["freq_dist1"]) > (data["R_emaildomain"]))*1.)))))/2.0)))*1.)))*1.)))*1.)))*1.)))) +

                    0.096832*np.tanh(((((((data["freq_V133"]) - (((data["freq_V130"]) + (((data["D8"]) * 2.0)))))) * 2.0)) + (((((data["C10"]) - (((((((data["freq_V133"]) - (((data["freq_V130"]) + (((data["D8"]) * 2.0)))))) * 2.0)) * 2.0)))) * ((((data["freq_D3"]) <= (((data["freq_V310"]) + (data["V310"]))))*1.)))))) +

                    0.096400*np.tanh((((((((((data["freq_V312"]) <= (((((data["V312"]) + ((((data["D10"]) + (data["V315"]))/2.0)))) * (data["dist1"]))))*1.)) - ((((data["D10"]) <= (((((np.tanh((data["card5"]))) - (((data["M6"]) * (data["freq_V312"]))))) / 2.0)))*1.)))) * 2.0)) * 2.0)) +

                    0.067619*np.tanh((((((((((data["C10"]) <= (np.tanh((data["V317"]))))*1.)) - ((((data["V45"]) > (data["R_emaildomain"]))*1.)))) - ((((data["V45"]) > ((((((data["R_emaildomain"]) * 2.0)) + (data["D15"]))/2.0)))*1.)))) - ((((((data["C13"]) > (((data["freq_dist2"]) / 2.0)))*1.)) * 2.0)))))



def GPII(data):

    return Output(  -3.317076 +

                    0.074000*np.tanh((((((((((((((-1.0*((((((data["freq_V317"]) * (((data["freq_C12"]) - ((((((data["freq_C10"]) / 2.0)) <= (data["C13"]))*1.)))))) - ((((((data["freq_C1"]) <= (3.0))*1.)) / 2.0))))))) * 2.0)) * 2.0)) * 2.0)) - (2.0))) * 2.0)) - (data["freq_C10"]))) +

                    0.094000*np.tanh((((6.83369398117065430)) * (((((((((((((data["freq_C5"]) + (((((data["card2"]) - (data["freq_C14"]))) - (((data["freq_C10"]) * ((((data["freq_V283"]) > (((((data["card1"]) * 2.0)) * 2.0)))*1.)))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - (data["freq_V283"]))))) +

                    0.076400*np.tanh((((((((((-1.0*(((((((data["freq_C8"]) > ((((data["freq_C10"]) > (data["freq_C10"]))*1.)))*1.)) / 2.0))))) + ((((((data["C13"]) * 2.0)) > (((((data["freq_V294"]) / 2.0)) * ((((data["freq_C2"]) + (data["freq_C8"]))/2.0)))))*1.)))) * 2.0)) * 2.0)) * 2.0)) +

                    0.093640*np.tanh(((((((((((data["freq_C5"]) - (((data["freq_V317"]) - (((((((data["card1"]) * 2.0)) * 2.0)) * 2.0)))))) * 2.0)) - (((data["freq_V312"]) - ((((((data["freq_C8"]) <= (((((data["C13"]) + (data["C14"]))) + (data["card1"]))))*1.)) * 2.0)))))) * 2.0)) * 2.0)) +

                    0.093973*np.tanh(((data["year"]) * ((((((((((((((data["freq_C8"]) <= (((((data["C13"]) + (data["D5"]))) + (data["C13"]))))*1.)) + (data["card1"]))) * 2.0)) - (data["card4"]))) - (((data["freq_V312"]) * (((data["C14"]) + (((data["V127"]) + (data["freq_V45"]))))))))) * 2.0)))) +

                    0.002016*np.tanh(((((((((((((np.tanh((data["C14"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) + (((((((((((((((np.tanh((data["V313"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - ((((((data["freq_C10"]) > (data["card2"]))*1.)) * (data["freq_V133"]))))) * 2.0)) * 2.0)))) * 2.0)) +

                    0.000026*np.tanh(((((((((((((((data["card2"]) + (((data["card2"]) + (((((data["card2"]) + (((data["card2"]) + (np.tanh(((((((data["freq_M6"]) > (data["freq_R_emaildomain"]))*1.)) - ((((data["freq_V308"]) + (data["freq_V283"]))/2.0)))))))))) / 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.093480*np.tanh(((((((((data["C2"]) + (((((((((((((data["C14"]) * ((14.45412063598632812)))) - (((data["freq_V133"]) - ((((data["freq_C8"]) <= (data["freq_P_emaildomain"]))*1.)))))) * 2.0)) - (((data["freq_D14"]) - (((data["card1"]) * ((14.45412063598632812)))))))) * 2.0)) * 2.0)))) * 2.0)) - (data["freq_D14"]))) * 2.0)) +

                    0.006432*np.tanh(((((((((((data["freq_P_emaildomain"]) - (((((((((data["freq_C10"]) <= (((data["card1"]) * 2.0)))*1.)) <= (data["freq_C8"]))*1.)) + (((((((data["freq_V317"]) * (((data["freq_C8"]) * 2.0)))) - (((((data["card1"]) * ((10.0)))) * 2.0)))) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.092288*np.tanh(((((((((data["freq_C5"]) + ((((((((((((((data["card1"]) > ((((((((data["freq_V283"]) / 2.0)) * (((np.tanh((data["freq_V45"]))) * (data["freq_V294"]))))) + (data["card4"]))/2.0)))*1.)) - (data["freq_D9"]))) * 2.0)) + (data["freq_V308"]))) * 2.0)) * 2.0)))) + (data["D15"]))) * 2.0)) * 2.0)) +

                    0.099994*np.tanh(((3.0) * ((((((data["freq_V308"]) <= (((((((data["C12"]) <= (np.tanh((data["freq_C13"]))))*1.)) + (data["card1"]))/2.0)))*1.)) + (((((data["freq_C5"]) - (((((((data["freq_C1"]) <= (np.tanh((data["freq_C13"]))))*1.)) <= ((-1.0*(((((data["C14"]) > (((data["M5"]) * 2.0)))*1.))))))*1.)))) * 2.0)))))) +

                    0.094266*np.tanh((((14.03874969482421875)) * ((((((((data["freq_V294"]) <= (((data["freq_V283"]) - (data["freq_C2"]))))*1.)) + (data["freq_D2"]))) - (((((((-3.0) * (data["C14"]))) + (((data["freq_V283"]) - ((((data["freq_D2"]) > ((((((data["freq_C2"]) * 2.0)) + (data["freq_V283"]))/2.0)))*1.)))))) * 2.0)))))) +

                    0.096192*np.tanh(((data["year"]) * (np.tanh((((((data["card2"]) * (((data["TransactionAmt"]) - (((((data["V283"]) + (((((((data["freq_V294"]) * (data["freq_C12"]))) * ((((data["freq_M4"]) + (data["freq_C8"]))/2.0)))) * (((data["year"]) / 2.0)))))) / 2.0)))))) - (data["freq_C8"]))))))) +

                    0.096760*np.tanh(((((((data["year"]) - (data["TransactionAmt"]))) * (((((((data["card1"]) * ((((data["card1"]) + (((data["V315"]) * ((((((data["D1"]) * (((data["TransactionAmt"]) * (data["TransactionAmt"]))))) + (data["year"]))/2.0)))))/2.0)))) - (data["freq_C1"]))) - (((data["freq_V45"]) - (data["D1"]))))))) * 2.0)) +

                    0.098800*np.tanh(((((data["card1"]) + (((((data["card1"]) - ((((14.48271274566650391)) * (((((data["card4"]) - ((((data["card4"]) <= ((((((((((data["D15"]) + (data["D3"]))/2.0)) + (data["M4"]))/2.0)) + ((((data["card1"]) + ((((data["C6"]) + (data["C13"]))/2.0)))/2.0)))/2.0)))*1.)))) * 2.0)))))) * 2.0)))) * 2.0)) +

                    0.097434*np.tanh(((((((((((((((data["card1"]) * 2.0)) * ((((14.33066368103027344)) + (((data["freq_C5"]) - (np.tanh(((14.33066368103027344)))))))))) + ((((((((np.tanh((data["freq_C5"]))) > (((data["freq_C10"]) / 2.0)))*1.)) - (((data["freq_C2"]) * 2.0)))) - (data["freq_dist2"]))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.099932*np.tanh(((-2.0) + (((((data["C14"]) + ((((((data["card1"]) + ((((data["freq_C13"]) > (data["freq_C1"]))*1.)))) > (data["freq_C6"]))*1.)))) * ((((data["freq_C5"]) + (((data["TransactionAmt"]) * (((((data["freq_C5"]) - (data["freq_C13"]))) * 2.0)))))/2.0)))))) +

                    0.091142*np.tanh((((((((-1.0*((data["freq_V45"])))) * ((12.01536273956298828)))) + (((((((data["card1"]) * ((12.01536273956298828)))) + (((data["D1"]) - (((((((data["freq_V314"]) + (data["freq_V45"]))/2.0)) + (data["freq_C11"]))/2.0)))))) * (((data["TransactionAmt"]) - (data["C1"]))))))) * 2.0)) +

                    0.093466*np.tanh(((((data["year"]) * ((((11.32336330413818359)) * (((((data["card1"]) - (((((data["freq_C1"]) * (data["freq_V315"]))) * ((((data["D3"]) <= ((((data["R_emaildomain"]) <= (data["freq_C14"]))*1.)))*1.)))))) + (((data["card1"]) + (((data["D3"]) + (data["freq_D15"]))))))))))) - (data["year"]))) +

                    0.099698*np.tanh(((((data["R_emaildomain"]) - (((((data["freq_C1"]) * ((((data["M5"]) > (data["C14"]))*1.)))) * ((((((data["card1"]) * (np.tanh((data["M5"]))))) <= (((((((data["freq_D11"]) + (data["R_emaildomain"]))) * (data["freq_V294"]))) / 2.0)))*1.)))))) * (((data["card5"]) * (data["year"]))))) +

                    0.093979*np.tanh((((((14.49952507019042969)) * (((((data["freq_D2"]) * (((((data["year"]) * (data["D15"]))) + (((((2.0) + ((((-1.0*(((14.49952507019042969))))) * 2.0)))) - (2.0))))))) - (((((data["freq_V133"]) - (data["freq_D3"]))) * (data["freq_V317"]))))))) * 2.0)) +

                    0.099999*np.tanh((((((-3.0) * 2.0)) + ((((((-1.0*((data["freq_C1"])))) + ((((((((data["freq_C5"]) > ((((((((data["freq_C1"]) / 2.0)) - (data["card2"]))) + (((data["freq_C1"]) + (data["freq_M5"]))))/2.0)))*1.)) * 2.0)) * (((data["D2"]) * (((data["year"]) * (data["C5"]))))))))) * 2.0)))/2.0)) +

                    0.097546*np.tanh(((-3.0) + ((((((((((data["freq_M4"]) <= ((((data["freq_P_emaildomain"]) + ((((((data["freq_C6"]) <= ((((data["freq_C6"]) <= (((((((((data["card1"]) > (((data["freq_C1"]) * (data["freq_V83"]))))*1.)) * 2.0)) + (data["freq_C1"]))/2.0)))*1.)))*1.)) * 2.0)))/2.0)))*1.)) * 2.0)) * 2.0)) * 2.0)))) +

                    0.099500*np.tanh((((((((((((((((data["C13"]) > (((data["M6"]) + (((data["freq_C5"]) * (((data["freq_C5"]) * ((-1.0*(((((data["freq_C1"]) <= (data["card1"]))*1.))))))))))))*1.)) * 2.0)) * 2.0)) - (1.0))) * 2.0)) * 2.0)) * 2.0)) +

                    0.097326*np.tanh(((data["M4"]) + ((((((((((((data["freq_D2"]) > (((data["M4"]) * ((((data["freq_D2"]) <= ((((((data["card1"]) - (data["freq_V308"]))) <= (data["M4"]))*1.)))*1.)))))*1.)) - (np.tanh((data["freq_C2"]))))) - (np.tanh((data["freq_D2"]))))) * (((data["year"]) - (data["freq_D2"]))))) * 2.0)))) +

                    0.095160*np.tanh(((((((((data["freq_C5"]) - ((((((data["freq_C6"]) > (((((((((((data["card1"]) > (data["freq_C1"]))*1.)) * 2.0)) * (data["freq_C5"]))) + (((((((np.tanh((data["card1"]))) / 2.0)) * (((data["freq_C5"]) * (data["TransactionAmt"]))))) / 2.0)))/2.0)))*1.)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

                    0.099723*np.tanh((((((((data["freq_C1"]) <= (((data["C14"]) * 2.0)))*1.)) - (((((data["freq_V310"]) - (np.tanh((np.tanh((data["freq_D2"]))))))) - (((np.tanh((data["card1"]))) - (np.tanh((data["M6"]))))))))) * (((((data["year"]) * 2.0)) * (data["freq_C1"]))))) +

                    0.099912*np.tanh(((((((((((((((3.0) * (np.tanh((data["V312"]))))) + ((((data["freq_C13"]) > (data["freq_C6"]))*1.)))) * 2.0)) * 2.0)) * 2.0)) - (3.0))) + (((((((((3.0) * (((data["C13"]) * 2.0)))) + ((((data["freq_C13"]) > (data["freq_C1"]))*1.)))) * 2.0)) * 2.0)))) +

                    0.099540*np.tanh(((data["TransactionAmt"]) * (((data["freq_C5"]) - ((((data["freq_C11"]) + (((((((((data["freq_C11"]) <= (data["freq_C1"]))*1.)) * ((((data["freq_M5"]) + ((((((data["freq_M5"]) + (data["month"]))/2.0)) / 2.0)))/2.0)))) + ((((data["freq_M5"]) + ((((((data["freq_M5"]) + (data["month"]))/2.0)) / 2.0)))/2.0)))/2.0)))/2.0)))))) +

                    0.097131*np.tanh(((data["year"]) * ((((data["D10"]) + (((((data["year"]) * (((((data["freq_dist1"]) - (np.tanh(((((data["freq_P_emaildomain"]) <= (data["freq_C1"]))*1.)))))) * ((((data["freq_V130"]) > (data["freq_C8"]))*1.)))))) - (data["freq_V283"]))))/2.0)))) +

                    0.099800*np.tanh((((((-1.0*(((((data["M5"]) <= (data["freq_P_emaildomain"]))*1.))))) + ((((((-1.0*((data["freq_V45"])))) + (((data["TransactionAmt"]) * (((np.tanh((((((data["freq_P_emaildomain"]) * 2.0)) + (data["V315"]))))) * (np.tanh(((((((data["card2"]) > (data["freq_V45"]))*1.)) + (data["card2"]))))))))))) * 2.0)))) * 2.0)) +

                    0.099936*np.tanh((-1.0*((((data["month"]) - (((((((data["D2"]) * ((((((data["D2"]) - (((data["freq_card1"]) - (((data["V313"]) * (data["TransactionAmt"]))))))) + (((((data["card2"]) / 2.0)) * (((data["year"]) - (data["D2"]))))))/2.0)))) * 2.0)) * 2.0))))))) +

                    0.099880*np.tanh(((data["TransactionAmt"]) + ((((((((((((((((data["V317"]) <= (data["P_emaildomain"]))*1.)) > (((((((data["V317"]) <= (data["D4"]))*1.)) <= (data["D4"]))*1.)))*1.)) + ((((data["card1"]) + (((((data["D2"]) * 2.0)) - (data["freq_M5"]))))/2.0)))/2.0)) * 2.0)) / 2.0)) * (((data["TransactionAmt"]) * (data["TransactionAmt"]))))))) +

                    0.098920*np.tanh(((((data["freq_D10"]) - ((((data["V313"]) <= ((((data["V315"]) <= (((((((((((data["freq_C13"]) > (data["freq_C11"]))*1.)) * 2.0)) * 2.0)) <= (((data["M6"]) - (data["C14"]))))*1.)))*1.)))*1.)))) * (((data["TransactionAmt"]) + (data["TransactionAmt"]))))) +

                    0.038640*np.tanh(((((((((((((data["freq_D5"]) - ((((((data["freq_P_emaildomain"]) <= (((data["freq_C11"]) - (data["V45"]))))*1.)) - ((((data["freq_C8"]) <= ((((((((data["freq_V308"]) + (data["V45"]))/2.0)) * 2.0)) / 2.0)))*1.)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) - (data["freq_V308"]))) +

                    0.099995*np.tanh((((((((((((((((data["freq_D10"]) * (data["freq_C1"]))) * 2.0)) <= (data["D4"]))*1.)) + (np.tanh((((data["freq_D10"]) - ((((((data["C14"]) * (data["freq_D10"]))) <= (data["V283"]))*1.)))))))) * 2.0)) * 2.0)) * 2.0)) +

                    0.098440*np.tanh((((((((((((((((((((((data["freq_C13"]) > (((data["freq_C1"]) + (data["freq_C1"]))))*1.)) * 2.0)) - ((((data["M6"]) > (data["C13"]))*1.)))) * 2.0)) * 2.0)) + (data["freq_C1"]))) * 2.0)) + (data["freq_C13"]))) - ((((data["freq_D3"]) <= (((data["freq_C1"]) + (data["freq_C10"]))))*1.)))) * 2.0)) +

                    0.062342*np.tanh(((((((((((((data["P_emaildomain"]) * 2.0)) + (((((data["C13"]) - (((data["freq_C11"]) - (((((((((data["D15"]) > (data["P_emaildomain"]))*1.)) > ((((data["freq_D3"]) <= (((data["freq_C11"]) - (data["freq_P_emaildomain"]))))*1.)))*1.)) + (data["card1"]))))))) * 2.0)))) - (data["freq_M4"]))) * 2.0)) * 2.0)) * 2.0)) +

                    0.097400*np.tanh(((((((np.tanh((((np.tanh((data["card1"]))) + ((((((((data["freq_C1"]) <= (data["C13"]))*1.)) * 2.0)) + ((((((((((data["R_emaildomain"]) * 2.0)) > (data["freq_V83"]))*1.)) * 2.0)) + (((data["freq_D15"]) - (data["freq_V130"]))))))))))) * 2.0)) * 2.0)) * 2.0)) +

                    0.094280*np.tanh(((((data["V283"]) + ((((((((((data["card1"]) - (data["freq_C11"]))) > ((-1.0*(((((9.0)) * (((data["freq_D5"]) * (((data["month"]) * ((((data["V283"]) <= (data["dist1"]))*1.))))))))))))*1.)) * 2.0)) * 2.0)))) - (((((data["month"]) * (data["freq_V45"]))) + (data["freq_C11"]))))) +

                    0.099900*np.tanh(((data["TransactionAmt"]) * (((data["TransactionAmt"]) * (((((np.tanh((((data["freq_C13"]) * (((((((data["C9"]) <= ((((data["R_emaildomain"]) + (data["V314"]))/2.0)))*1.)) + ((((data["V133"]) <= ((((data["freq_C1"]) <= ((((data["freq_C1"]) <= (data["C14"]))*1.)))*1.)))*1.)))/2.0)))))) * 2.0)) - (data["V133"]))))))) +

                    0.099885*np.tanh(((((((((((((((data["card2"]) + (((((((data["freq_C14"]) + ((((((data["freq_V45"]) <= (data["card2"]))*1.)) * 2.0)))) * 2.0)) - (((data["freq_V83"]) - (((data["C6"]) * 2.0)))))))) * 2.0)) * 2.0)) * 2.0)) - ((((data["freq_C1"]) > (((data["freq_C10"]) * 2.0)))*1.)))) * 2.0)) * 2.0)) +

                    0.095117*np.tanh((((((((-1.0*(((((data["card1"]) <= (data["freq_D2"]))*1.))))) + (((((data["freq_C10"]) - (((((data["freq_C9"]) - ((((((data["C14"]) > (((((((data["freq_C13"]) <= (data["freq_C1"]))*1.)) + (data["freq_C1"]))/2.0)))*1.)) * 2.0)))) * 2.0)))) - ((((data["freq_C9"]) > (data["freq_C1"]))*1.)))))) * 2.0)) * 2.0)) +

                    0.100000*np.tanh(((((((((((((((data["V45"]) + (((data["freq_D3"]) + (data["card1"]))))) * 2.0)) - (((((((data["C12"]) <= (data["dist1"]))*1.)) <= (data["C12"]))*1.)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.099950*np.tanh(((((((((((((((((data["D8"]) + (data["V45"]))) - (data["freq_M5"]))) * 2.0)) + (data["card2"]))) - (((data["freq_C10"]) - ((((data["freq_C10"]) > (((data["freq_C1"]) - (data["C14"]))))*1.)))))) * 2.0)) - (((data["freq_M5"]) - ((((data["V310"]) > (data["V45"]))*1.)))))) * 2.0)) +

                    0.098000*np.tanh(((((((((((((((data["D9"]) <= (((data["freq_card1"]) + (((data["freq_card1"]) + (((data["card1"]) - (data["D9"]))))))))*1.)) + ((((((data["R_emaildomain"]) > (data["M5"]))*1.)) * 2.0)))) > (data["M5"]))*1.)) * 2.0)) - ((((data["R_emaildomain"]) <= (data["M5"]))*1.)))) * 2.0)) +

                    0.096408*np.tanh((((((((((((((data["freq_C1"]) > (np.tanh((((data["C6"]) * (((((data["freq_C14"]) * 2.0)) * 2.0)))))))*1.)) + (((((data["C14"]) * 2.0)) * (((data["freq_C14"]) * 2.0)))))/2.0)) <= (((((((data["V315"]) * 2.0)) + (((((data["C14"]) * 2.0)) * 2.0)))) * 2.0)))*1.)) * 2.0)) * 2.0)) +

                    0.094120*np.tanh(((((2.0) * (((data["freq_C1"]) + (((((((((((((((data["C14"]) > (data["freq_C1"]))*1.)) > ((((data["freq_C8"]) > (((data["freq_dist1"]) * (data["freq_V310"]))))*1.)))*1.)) * 2.0)) - (data["freq_V130"]))) * 2.0)) - (data["freq_V310"]))))))) * 2.0)) +

                    0.098406*np.tanh(((((((((data["C1"]) + ((-1.0*(((((data["card4"]) > (data["freq_C14"]))*1.))))))) * 2.0)) + ((((((data["freq_C9"]) <= (((((((data["C1"]) + (((data["freq_D3"]) * (data["freq_D5"]))))) * 2.0)) * 2.0)))*1.)) + ((-1.0*((data["freq_C11"])))))))) * 2.0)) +

                    0.099908*np.tanh(((((-3.0) * (((data["freq_D14"]) - (((((((data["freq_dist2"]) * 2.0)) + (data["V45"]))) * ((((data["D9"]) > (((data["freq_C2"]) - ((((((data["freq_D14"]) <= ((((data["freq_dist2"]) <= ((((data["V133"]) <= (data["M4"]))*1.)))*1.)))*1.)) + (data["D9"]))))))*1.)))))))) * 2.0)) +

                    0.099960*np.tanh(((((((((data["card1"]) * 2.0)) - ((((((data["freq_D14"]) / 2.0)) + (((data["freq_V127"]) * (((data["year"]) * (((data["C12"]) - (((data["dist1"]) + ((((np.tanh((data["freq_D14"]))) <= (((data["V45"]) * (((((data["card1"]) * 2.0)) * 2.0)))))*1.)))))))))))/2.0)))) * 2.0)) * 2.0)) +

                    0.100000*np.tanh(((((data["freq_card5"]) - (((data["freq_V310"]) * (((((((data["freq_V310"]) * (((((data["freq_D14"]) - (((((((((((data["freq_C14"]) * 2.0)) * ((((data["freq_card5"]) > ((((data["freq_C11"]) > (data["D9"]))*1.)))*1.)))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)))))) * 2.0)) +

                    0.098600*np.tanh(((((((((((data["card1"]) - (((((((data["card1"]) <= ((((((((((data["freq_C9"]) + ((-1.0*(((((data["C14"]) > (data["C10"]))*1.))))))/2.0)) > (data["freq_C13"]))*1.)) > (data["card2"]))*1.)))*1.)) > (data["card2"]))*1.)))) * 2.0)) * 2.0)) + (((data["freq_V317"]) - (data["freq_C2"]))))) * 2.0)) +

                    0.099648*np.tanh(((((data["V45"]) - ((((data["V313"]) <= ((((data["freq_C14"]) <= (data["freq_TransactionAmt"]))*1.)))*1.)))) + (((((data["V45"]) + (((data["V45"]) + ((((((data["freq_C13"]) > (data["freq_C6"]))*1.)) * 2.0)))))) - ((((data["card2"]) <= ((((data["V45"]) + ((((data["C14"]) <= (data["M6"]))*1.)))/2.0)))*1.)))))) +

                    0.100000*np.tanh(((data["freq_C1"]) + (((((((((data["freq_V45"]) - (data["D8"]))) / 2.0)) - (((data["freq_V133"]) - (((((((data["freq_C1"]) - (data["freq_dist1"]))) + ((((((data["D9"]) > (((((((data["freq_C2"]) + (data["M6"]))/2.0)) + (((data["freq_V45"]) / 2.0)))/2.0)))*1.)) * 2.0)))) * 2.0)))))) * 2.0)))) +

                    0.050880*np.tanh(((((((((((((data["D3"]) <= (data["D11"]))*1.)) > (data["freq_C8"]))*1.)) - (((data["freq_C5"]) - (((((((np.tanh((data["card1"]))) - ((((data["R_emaildomain"]) > (((((data["R_emaildomain"]) * ((11.48718547821044922)))) - ((((data["freq_addr1"]) > (data["R_emaildomain"]))*1.)))))*1.)))) * 2.0)) * 2.0)))))) * 2.0)) * 2.0)) +

                    0.099051*np.tanh(((((((data["freq_C14"]) + ((((((((data["C12"]) <= ((((((((data["card1"]) - (((data["card2"]) / 2.0)))) + ((((((data["C12"]) <= (((np.tanh((data["R_emaildomain"]))) / 2.0)))*1.)) * 2.0)))/2.0)) - (data["C12"]))))*1.)) * 2.0)) - ((((data["M6"]) <= (data["R_emaildomain"]))*1.)))))) * 2.0)) * 2.0)) +

                    0.099552*np.tanh(((((((((((data["freq_C8"]) - ((((data["V308"]) > (((data["dist1"]) * 2.0)))*1.)))) + (((data["card2"]) - ((((data["freq_D1"]) > ((((((((data["card1"]) - ((((data["V317"]) > (((data["card2"]) * 2.0)))*1.)))) + (data["freq_V308"]))/2.0)) * 2.0)))*1.)))))) * 2.0)) * 2.0)) * 2.0)) +

                    0.094800*np.tanh(((((((data["V315"]) - ((((((((data["V307"]) > (data["freq_C14"]))*1.)) + (data["freq_C11"]))) - (((((((data["C14"]) > (data["C1"]))*1.)) + (((data["V45"]) + (((data["card5"]) - ((((data["P_emaildomain"]) > (data["freq_C14"]))*1.)))))))/2.0)))))) + ((((data["C6"]) > (data["C5"]))*1.)))) * 2.0)) +

                    0.069152*np.tanh((((((((((((data["V313"]) > (((((-1.0*((data["addr1"])))) + ((((data["freq_M5"]) + ((((data["freq_M5"]) <= (data["freq_card1"]))*1.)))/2.0)))/2.0)))*1.)) * 2.0)) + ((((((data["V294"]) <= (data["C2"]))*1.)) - (((data["freq_M5"]) + ((((data["D10"]) <= (data["C12"]))*1.)))))))) * 2.0)) * 2.0)) +

                    0.077152*np.tanh(((data["freq_card1"]) + ((((((((((((data["R_emaildomain"]) > (data["freq_V83"]))*1.)) + ((((((data["freq_card1"]) > (data["freq_V127"]))*1.)) - ((((data["R_emaildomain"]) <= (data["V283"]))*1.)))))) + ((((data["freq_C13"]) > (data["freq_C11"]))*1.)))) * 2.0)) - ((((data["card2"]) <= (data["R_emaildomain"]))*1.)))))) +

                    0.090880*np.tanh(((data["V315"]) + (((((data["V45"]) + (((((((((data["C6"]) * 2.0)) + (data["card2"]))) + ((((((data["C8"]) > (data["freq_D9"]))*1.)) - ((((((data["C1"]) * 2.0)) > (np.tanh((((data["C14"]) * 2.0)))))*1.)))))) * 2.0)))) + ((((data["V283"]) <= (data["D1"]))*1.)))))) +

                    0.099112*np.tanh((((((((-1.0*((((data["freq_C1"]) - ((((data["M5"]) > (((((data["C8"]) - ((((((data["card2"]) + (((data["V313"]) + (data["D8"]))))/2.0)) * ((((((data["V310"]) + (data["D8"]))) + (((data["V310"]) + (data["card2"]))))/2.0)))))) * 2.0)))*1.))))))) * 2.0)) * 2.0)) * 2.0)) +

                    0.097000*np.tanh(((((((data["TransactionAmt"]) * (((data["card1"]) * 2.0)))) * ((((((data["C8"]) + (data["R_emaildomain"]))/2.0)) + ((((((data["card1"]) <= (data["dist2"]))*1.)) + ((((np.tanh((data["freq_C9"]))) <= (data["dist2"]))*1.)))))))) + (((data["dist2"]) - (((data["freq_D1"]) + (1.0))))))) +

                    0.099901*np.tanh(((data["freq_card2"]) + (((data["TransactionAmt"]) * ((((-1.0) + (((((((((((data["C6"]) * 2.0)) + ((((((data["freq_V317"]) * (((data["freq_C13"]) * 2.0)))) + (((data["C13"]) * 2.0)))/2.0)))) + (((data["V312"]) + (np.tanh((data["V313"]))))))) * 2.0)) * 2.0)))/2.0)))))) +

                    0.099900*np.tanh(((((((((data["C11"]) + ((-1.0*(((((-1.0*((((((data["D1"]) + (data["C11"]))) + (((((((data["D14"]) * 2.0)) - ((((data["freq_D2"]) + ((((data["D1"]) <= (((data["card5"]) - (data["card4"]))))*1.)))/2.0)))) + (data["D8"])))))))) * 2.0))))))) * 2.0)) * 2.0)) * 2.0)) +

                    0.090819*np.tanh(((((data["freq_card4"]) - ((((data["freq_D3"]) <= (((((((data["freq_card4"]) + ((((((7.0)) - (data["hour"]))) * ((-1.0*((data["freq_D1"])))))))/2.0)) + (((data["freq_V130"]) * (((data["V310"]) - ((((data["freq_D1"]) > (np.tanh((((data["C11"]) - (data["V310"]))))))*1.)))))))/2.0)))*1.)))) * 2.0)) +

                    0.058016*np.tanh(((((data["V45"]) - ((((np.tanh((data["P_emaildomain"]))) <= (data["P_emaildomain"]))*1.)))) + (((((((data["V45"]) - ((((data["C1"]) <= (data["P_emaildomain"]))*1.)))) + ((((((data["V45"]) * 2.0)) > ((((data["addr1"]) <= (((data["V45"]) - (data["freq_D15"]))))*1.)))*1.)))) * 2.0)))) +

                    0.099824*np.tanh((((((((((((data["freq_C8"]) <= (np.tanh(((((data["freq_C8"]) <= (data["V283"]))*1.)))))*1.)) - ((((((data["freq_C14"]) <= (((data["freq_V83"]) * (data["V283"]))))*1.)) * 2.0)))) + ((((data["V283"]) <= (data["D2"]))*1.)))) - ((((data["D2"]) <= (data["freq_V127"]))*1.)))) * 2.0)) +

                    0.070640*np.tanh((((((((data["freq_C8"]) <= (((data["freq_V294"]) * (data["freq_dist2"]))))*1.)) - (data["freq_V294"]))) + ((((((((data["freq_C8"]) <= (((data["freq_V294"]) * (data["freq_dist2"]))))*1.)) - ((((data["addr1"]) <= (data["D8"]))*1.)))) + ((((((data["freq_V294"]) <= (((data["card1"]) * (data["V313"]))))*1.)) * 2.0)))))) +

                    0.098829*np.tanh((((((((((((data["C8"]) > (data["freq_V83"]))*1.)) * 2.0)) * 2.0)) - ((((((data["V45"]) * 2.0)) <= ((((data["card4"]) <= ((((data["card2"]) + ((((((data["V45"]) + ((((data["freq_C14"]) <= (data["V313"]))*1.)))/2.0)) * 2.0)))/2.0)))*1.)))*1.)))) + ((((data["V313"]) <= (data["M4"]))*1.)))) +

                    0.034000*np.tanh((((((((((data["freq_C9"]) <= (data["V315"]))*1.)) + (((data["freq_V315"]) * ((((-1.0*(((((data["freq_C1"]) > (((data["freq_R_emaildomain"]) / 2.0)))*1.))))) + ((((data["freq_TransactionAmt"]) > ((((((data["freq_C1"]) * (data["C10"]))) + (((data["V315"]) * (data["M4"]))))/2.0)))*1.)))))))) * 2.0)) * 2.0)) +

                    0.099900*np.tanh(((((((((((data["D9"]) + ((((data["card1"]) > ((((data["V294"]) <= (((data["freq_M5"]) - (data["card1"]))))*1.)))*1.)))) - ((((data["hour"]) <= (data["card1"]))*1.)))) - ((((data["C12"]) > (((((data["freq_C13"]) * 2.0)) + (((((data["card1"]) / 2.0)) / 2.0)))))*1.)))) * 2.0)) * 2.0)) +

                    0.041920*np.tanh((((data["freq_C5"]) <= ((((((data["C12"]) <= (((data["dist1"]) - ((((((-1.0) * (data["D14"]))) > ((((data["V294"]) + (((data["D8"]) - ((((data["freq_C2"]) + (((data["freq_dist1"]) / 2.0)))/2.0)))))/2.0)))*1.)))))*1.)) + (((data["card2"]) + ((((data["D14"]) + (data["freq_C5"]))/2.0)))))))*1.)) +

                    0.099600*np.tanh((((((((((data["freq_card1"]) > (data["freq_V130"]))*1.)) - ((-1.0*((((((((data["V315"]) - ((-1.0*((((data["freq_V130"]) - ((((data["freq_addr1"]) > ((((data["freq_C14"]) + (data["freq_card2"]))/2.0)))*1.))))))))) - ((((data["dist1"]) <= (data["V45"]))*1.)))) - (data["freq_dist1"])))))))) * 2.0)) - (data["freq_card1"]))) +

                    0.041370*np.tanh(((((((((data["freq_V314"]) - ((((((((((((((((data["card2"]) > ((((data["V315"]) <= (data["freq_C13"]))*1.)))*1.)) + (data["D4"]))) > ((((data["card2"]) <= (data["freq_C13"]))*1.)))*1.)) + (data["D4"]))) + (data["D4"]))) <= (data["freq_C1"]))*1.)))) * 2.0)) * 2.0)) * 2.0)) +

                    0.099300*np.tanh(((data["hour"]) * (((data["hour"]) * ((((((np.tanh((data["freq_V317"]))) <= (data["freq_V83"]))*1.)) * (((((((data["card2"]) + (((((data["freq_V312"]) + (data["V315"]))) - (((data["freq_M5"]) - (np.tanh((data["freq_V317"]))))))))/2.0)) + (data["card1"]))/2.0)))))))) +

                    0.099158*np.tanh(((((((((((data["C6"]) + (((data["V313"]) - (((((((data["freq_TransactionAmt"]) + ((((data["V313"]) + ((((data["freq_C1"]) > (data["freq_V294"]))*1.)))/2.0)))/2.0)) > (((3.0) * (data["freq_D8"]))))*1.)))))) * 2.0)) - ((((data["C14"]) > (((3.0) * (data["C5"]))))*1.)))) * 2.0)) * 2.0)) +

                    0.099200*np.tanh(((data["freq_P_emaildomain"]) + ((((((((data["freq_V308"]) <= (((data["card2"]) - (data["V283"]))))*1.)) + (((((data["D9"]) * 2.0)) - ((((((data["D1"]) <= (((data["M6"]) / 2.0)))*1.)) + (data["V283"]))))))) - ((((((data["D1"]) <= (((data["V283"]) / 2.0)))*1.)) + (data["freq_C5"]))))))) +

                    0.010157*np.tanh(((((((data["R_emaildomain"]) + ((((data["D14"]) > ((((((data["D5"]) + (data["freq_C5"]))/2.0)) - (((data["V315"]) * 2.0)))))*1.)))) - ((((data["V315"]) <= (((data["freq_C9"]) * ((((((data["addr1"]) + (data["freq_dist1"]))/2.0)) * ((((data["D5"]) > (data["R_emaildomain"]))*1.)))))))*1.)))) * 2.0)) +

                    0.034576*np.tanh((((((((((-1.0*((((data["freq_C5"]) + ((((-1.0*(((((data["freq_TransactionAmt"]) <= ((((data["freq_V283"]) > (data["C13"]))*1.)))*1.))))) / 2.0))))))) + (((((((data["R_emaildomain"]) > (data["freq_V283"]))*1.)) > (((data["C2"]) + ((((-1.0*((data["C13"])))) / 2.0)))))*1.)))) * 2.0)) * 2.0)) * 2.0)) +

                    0.008000*np.tanh(((((((((((data["freq_D5"]) <= (data["freq_C1"]))*1.)) + (data["freq_C2"]))) <= ((((((((np.tanh((data["addr1"]))) > (data["M4"]))*1.)) / 2.0)) - ((((data["freq_C2"]) + (data["C13"]))/2.0)))))*1.)) - ((((data["D10"]) > ((-1.0*(((((-1.0*((data["freq_D5"])))) * 2.0))))))*1.)))) +

                    0.099980*np.tanh((((((((((((((((((((data["freq_addr1"]) <= (((data["R_emaildomain"]) * ((((data["M4"]) <= (data["M5"]))*1.)))))*1.)) * 2.0)) * 2.0)) - ((((data["freq_addr1"]) <= (data["M5"]))*1.)))) - (data["C9"]))) * 2.0)) - ((((data["freq_addr1"]) <= (data["V127"]))*1.)))) - (data["C9"]))) + (data["M4"]))) +

                    0.099993*np.tanh(((((data["freq_R_emaildomain"]) - (((((((data["freq_D2"]) <= (data["freq_M5"]))*1.)) > ((((data["freq_V315"]) > ((((data["freq_card2"]) <= (((((data["D9"]) * 2.0)) + (data["freq_C13"]))))*1.)))*1.)))*1.)))) + (((data["freq_M5"]) * ((((((data["freq_V307"]) <= (np.tanh(((((data["freq_V313"]) <= (data["freq_V315"]))*1.)))))*1.)) * 2.0)))))) +

                    0.053920*np.tanh((((((((data["freq_C8"]) <= (data["freq_D5"]))*1.)) + ((((data["freq_C9"]) <= (((data["dist1"]) / 2.0)))*1.)))) - ((((((((data["P_emaildomain"]) <= ((((((data["D11"]) <= ((((data["V45"]) <= ((((((-1.0*((((data["freq_C14"]) - (data["V283"])))))) * 2.0)) * 2.0)))*1.)))*1.)) / 2.0)))*1.)) * 2.0)) * 2.0)))) +

                    0.099699*np.tanh((((((((((data["freq_C11"]) > (((data["freq_C9"]) * ((((6.76347398757934570)) * ((((((((data["freq_card5"]) > (data["freq_card1"]))*1.)) - ((((data["card2"]) > (data["freq_D9"]))*1.)))) - (data["freq_card1"]))))))))*1.)) + ((((data["V313"]) > (((((data["freq_D9"]) - (data["card2"]))) * 2.0)))*1.)))) * 2.0)) * 2.0)) +

                    0.082432*np.tanh(((((((((data["card2"]) - ((((data["C14"]) <= (data["C10"]))*1.)))) + ((((((data["C10"]) <= (((data["dist1"]) - ((((data["V314"]) > (data["M4"]))*1.)))))*1.)) - ((((data["addr1"]) <= (data["M4"]))*1.)))))) * 2.0)) + (((data["card2"]) - ((((data["C14"]) <= (data["C10"]))*1.)))))) +

                    0.051024*np.tanh((((((data["V315"]) > ((((data["C1"]) + (data["freq_C9"]))/2.0)))*1.)) + (((data["C11"]) + ((((((((data["freq_C11"]) <= (((data["freq_C13"]) / 2.0)))*1.)) + ((((data["V315"]) > ((((data["C1"]) + (((data["freq_dist1"]) / 2.0)))/2.0)))*1.)))) + ((-1.0*((data["freq_dist1"])))))))))) +

                    0.045443*np.tanh((((((data["C13"]) <= (data["V83"]))*1.)) + ((((((data["freq_C12"]) <= (data["freq_C10"]))*1.)) - (((((data["freq_D15"]) - (((((((((data["freq_C10"]) - (data["freq_V127"]))) > (data["D15"]))*1.)) <= (data["C8"]))*1.)))) + ((((data["P_emaildomain"]) <= ((((data["freq_P_emaildomain"]) + (data["M4"]))/2.0)))*1.)))))))) +

                    0.099960*np.tanh((((data["freq_D1"]) <= ((((data["freq_D15"]) + (((((((data["freq_D15"]) + (((((-1.0*(((-1.0*((data["V45"]))))))) > (data["freq_D15"]))*1.)))/2.0)) > ((((data["freq_C1"]) + ((((data["V45"]) > ((((data["C13"]) + ((((data["C13"]) <= (((data["V315"]) * (data["freq_M4"]))))*1.)))/2.0)))*1.)))/2.0)))*1.)))/2.0)))*1.)) +

                    0.058240*np.tanh((((((((((data["C8"]) - (data["freq_card1"]))) <= ((((((data["C14"]) + (data["C14"]))) <= ((((data["C1"]) + ((((data["C14"]) + (data["V315"]))/2.0)))/2.0)))*1.)))*1.)) * 2.0)) - ((((data["C9"]) <= (((data["C1"]) + ((((((data["C14"]) <= (data["C8"]))*1.)) - (data["freq_card1"]))))))*1.)))) +

                    0.002520*np.tanh((((((((((data["V315"]) * 2.0)) * 2.0)) > ((((((((data["dist2"]) * 2.0)) <= ((((((((((data["D14"]) * 2.0)) * 2.0)) <= ((((data["V315"]) > ((((((data["card2"]) * 2.0)) > (((((data["freq_V127"]) + (data["freq_C5"]))) / 2.0)))*1.)))*1.)))*1.)) / 2.0)))*1.)) / 2.0)))*1.)) * 2.0)) +

                    0.086051*np.tanh(((((((data["V45"]) - ((((data["V45"]) > ((((data["freq_D5"]) > (((data["freq_C14"]) * (((data["freq_D10"]) - (data["C10"]))))))*1.)))*1.)))) + (((data["freq_C14"]) * ((((data["V45"]) <= ((((data["C10"]) + ((((data["M5"]) + (data["card5"]))/2.0)))/2.0)))*1.)))))) * 2.0)) +

                    0.090800*np.tanh(((((data["freq_M4"]) + (data["freq_M4"]))) - (((data["freq_C14"]) + ((((((((data["freq_C14"]) <= (((data["C2"]) - (((data["freq_D2"]) + ((((((((data["freq_C14"]) <= (np.tanh((data["freq_C2"]))))*1.)) * 2.0)) * ((-1.0*((data["freq_D2"])))))))))))*1.)) * 2.0)) * 2.0)))))) +

                    0.098800*np.tanh((((((data["V127"]) <= (((((((((((data["C13"]) + (data["V312"]))) / 2.0)) - ((((data["V312"]) > ((((data["freq_V283"]) + (data["freq_R_emaildomain"]))/2.0)))*1.)))) / 2.0)) - ((((data["freq_R_emaildomain"]) > ((((((data["freq_V307"]) + (data["V130"]))/2.0)) + ((((data["freq_C13"]) <= (data["V130"]))*1.)))))*1.)))))*1.)) * 2.0)) +

                    0.097765*np.tanh((((((((((data["D4"]) + ((((data["card2"]) + ((((-1.0*(((((data["C5"]) > (((data["V307"]) + ((((data["C11"]) <= (((data["freq_V130"]) + ((((data["freq_D10"]) + (((data["freq_D14"]) * ((((data["C11"]) <= (data["C9"]))*1.)))))/2.0)))))*1.)))))*1.))))) * 2.0)))/2.0)))/2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.068880*np.tanh((((((((((data["freq_V312"]) <= ((((((((((data["D15"]) + (((data["M5"]) * (data["card1"]))))/2.0)) / 2.0)) / 2.0)) / 2.0)))*1.)) * 2.0)) * 2.0)) - ((((data["card1"]) <= ((((data["V312"]) <= (((data["card1"]) - ((((data["freq_V133"]) <= (((data["V312"]) * (data["freq_V312"]))))*1.)))))*1.)))*1.)))) +

                    0.070400*np.tanh((((((((((data["C2"]) <= (((data["C6"]) / 2.0)))*1.)) + ((((data["freq_dist1"]) <= ((((data["C8"]) <= (((data["C10"]) / 2.0)))*1.)))*1.)))) * 2.0)) + (((((data["C8"]) - ((((data["C6"]) > (np.tanh((data["V315"]))))*1.)))) + ((((data["freq_C9"]) <= (data["C8"]))*1.)))))) +

                    0.039523*np.tanh((((((((((data["card5"]) <= ((((((data["V83"]) + ((((data["freq_V133"]) <= ((((((data["freq_V133"]) <= (data["freq_dist1"]))*1.)) - ((((data["V83"]) > (data["card5"]))*1.)))))*1.)))/2.0)) / 2.0)))*1.)) * 2.0)) + ((((data["freq_dist1"]) <= (((((data["freq_V315"]) * (((data["D14"]) * 2.0)))) * 2.0)))*1.)))) * 2.0)) +

                    0.000200*np.tanh(((((((data["freq_C12"]) + (((data["V307"]) * ((-1.0*((data["freq_V83"])))))))/2.0)) <= (((data["V313"]) - (((((data["V133"]) - (((data["freq_TransactionAmt"]) + (((data["freq_TransactionAmt"]) + ((((data["freq_dist1"]) <= (((data["freq_card1"]) * 2.0)))*1.)))))))) * 2.0)))))*1.)) +

                    0.097208*np.tanh((((((((((-1.0*(((((data["V313"]) <= ((((data["card1"]) <= ((((data["P_emaildomain"]) <= (((data["V83"]) / 2.0)))*1.)))*1.)))*1.))))) * 2.0)) - ((((((np.tanh((((data["card5"]) / 2.0)))) / 2.0)) > (data["D15"]))*1.)))) * 2.0)) + ((((data["freq_card2"]) > (data["freq_C1"]))*1.)))) +

                    0.099840*np.tanh((((((((((data["C2"]) / 2.0)) <= (((data["freq_D1"]) * ((((((data["freq_D1"]) * 2.0)) <= ((((((data["D14"]) > ((((data["freq_D15"]) + (data["C8"]))/2.0)))*1.)) * 2.0)))*1.)))))*1.)) + ((((data["freq_V283"]) <= (((((data["card1"]) - (data["freq_D1"]))) - (data["freq_D1"]))))*1.)))) * 2.0)) +

                    0.099900*np.tanh(((data["card2"]) - ((((((data["freq_V310"]) > (((data["card2"]) * 2.0)))*1.)) - ((-1.0*((((data["freq_V283"]) * ((((((((np.tanh((np.tanh((data["card2"]))))) * 2.0)) > (((data["V313"]) * 2.0)))*1.)) - (((data["freq_V310"]) * (((data["TransactionAmt"]) * (data["V294"])))))))))))))))) +

                    0.100000*np.tanh((((((((data["freq_C8"]) <= (((data["freq_D9"]) * (data["freq_D3"]))))*1.)) - ((((data["freq_V294"]) <= (((((data["freq_C11"]) / 2.0)) + ((-1.0*(((((data["V315"]) > (data["freq_D3"]))*1.))))))))*1.)))) - ((((data["D4"]) <= (((data["M6"]) + ((-1.0*((((data["D4"]) * (3.0)))))))))*1.)))) +

                    0.039248*np.tanh(((((((((data["D9"]) + ((((((((((data["freq_C2"]) / 2.0)) <= ((((data["D9"]) <= (((data["dist1"]) / 2.0)))*1.)))*1.)) + ((((((data["freq_D9"]) <= (((((data["freq_D3"]) / 2.0)) + ((-1.0*((((data["freq_C2"]) / 2.0))))))))*1.)) - (data["C8"]))))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) +

                    0.099760*np.tanh(((((((((((data["freq_V314"]) * ((((((data["freq_V317"]) * (((data["freq_V317"]) * (data["card1"]))))) <= (data["M4"]))*1.)))) * 2.0)) * 2.0)) * 2.0)) + (((data["V313"]) - ((((((data["freq_dist1"]) * (data["freq_V310"]))) > (((data["V313"]) * (data["D10"]))))*1.)))))) +

                    0.099930*np.tanh(((data["card1"]) + (((((((((data["V315"]) - ((((data["D10"]) <= (((((((data["card2"]) > (((data["freq_V83"]) * (data["C12"]))))*1.)) <= ((((data["freq_V83"]) > (((((((data["card2"]) > (((data["freq_V83"]) * (data["D10"]))))*1.)) > (data["P_emaildomain"]))*1.)))*1.)))*1.)))*1.)))) * 2.0)) * 2.0)) * 2.0)))) +

                    0.099477*np.tanh(((((data["D14"]) - (((((((data["freq_C8"]) > (data["freq_dist2"]))*1.)) > ((-1.0*(((((data["V313"]) <= ((((((data["freq_addr1"]) * 2.0)) > (((((((data["freq_M4"]) + (data["dist2"]))/2.0)) > ((((((data["C11"]) + (data["freq_C9"]))/2.0)) / 2.0)))*1.)))*1.)))*1.))))))*1.)))) + (data["dist2"]))) +

                    0.099718*np.tanh((((((((((((data["freq_V130"]) <= (((data["freq_D15"]) * (data["freq_C6"]))))*1.)) + (data["D14"]))) + ((-1.0*(((((data["card1"]) <= ((((data["freq_C6"]) <= (((data["D3"]) - ((((((data["D3"]) + ((((data["freq_card5"]) + (data["card1"]))/2.0)))/2.0)) / 2.0)))))*1.)))*1.))))))) * 2.0)) * 2.0)) +

                    0.099500*np.tanh(((data["V45"]) + ((-1.0*(((((np.tanh((((data["D11"]) - ((((data["D3"]) <= ((((data["addr1"]) + (((data["M4"]) - ((((data["freq_V312"]) > (data["D3"]))*1.)))))/2.0)))*1.)))))) <= ((((((data["V45"]) + (((data["C13"]) - ((((data["M4"]) > (data["D3"]))*1.)))))/2.0)) / 2.0)))*1.))))))) +

                    0.099400*np.tanh((((((data["freq_C8"]) <= ((((data["freq_V45"]) > (((((data["freq_M5"]) + (data["D3"]))) + (((data["freq_card4"]) - (((np.tanh(((((np.tanh((data["card2"]))) <= (((data["card1"]) * (((data["freq_M5"]) + (((data["D14"]) + (data["C10"]))))))))*1.)))) * 2.0)))))))*1.)))*1.)) * 2.0)) +

                    0.099088*np.tanh((-1.0*(((((((data["V130"]) <= (((((((((data["P_emaildomain"]) > (data["V130"]))*1.)) * (data["freq_card5"]))) + ((((data["P_emaildomain"]) <= (((((((data["freq_D8"]) <= (data["freq_card1"]))*1.)) + (((data["V314"]) * ((((data["P_emaildomain"]) <= ((((data["V314"]) <= (data["V130"]))*1.)))*1.)))))/2.0)))*1.)))/2.0)))*1.)) * 2.0))))) +

                    0.099899*np.tanh((((((data["dist1"]) > ((((((data["freq_C9"]) + (((data["freq_addr1"]) - (((((((((((((((data["M4"]) > (data["freq_addr1"]))*1.)) + (data["freq_D2"]))/2.0)) <= (((data["freq_D11"]) / 2.0)))*1.)) * (data["freq_C9"]))) > (data["C2"]))*1.)))))/2.0)) * 2.0)))*1.)) * (((3.0) + (3.0))))) +

                    0.098842*np.tanh(((data["dist2"]) + ((((((-1.0*(((((data["V83"]) <= (data["V294"]))*1.))))) * (((data["freq_C12"]) - ((((data["freq_card5"]) > (((data["freq_C12"]) + (data["freq_card2"]))))*1.)))))) - (np.tanh(((((((((data["freq_card2"]) > (data["freq_C12"]))*1.)) * 2.0)) - ((((data["freq_card5"]) > (data["freq_C12"]))*1.)))))))))) +

                    0.099386*np.tanh((((((((((((data["freq_V127"]) > (np.tanh(((((data["C8"]) <= ((((((data["C1"]) * 2.0)) > (data["C6"]))*1.)))*1.)))))*1.)) - (((((data["C8"]) - ((((((data["freq_V294"]) + (data["freq_addr1"]))) <= (data["C8"]))*1.)))) + (data["C1"]))))) * 2.0)) - (data["freq_addr1"]))) * 2.0)) +

                    0.087798*np.tanh(((((-1.0*(((((((((data["P_emaildomain"]) * 2.0)) * 2.0)) + (data["dist2"]))/2.0))))) <= (((data["V315"]) - ((((((((data["freq_C12"]) > (((data["freq_R_emaildomain"]) - (((((((data["freq_dist1"]) / 2.0)) / 2.0)) / 2.0)))))*1.)) / 2.0)) / 2.0)))))*1.)) +

                    0.067312*np.tanh((((((((((((((data["C2"]) <= (((data["M4"]) * (((((((data["freq_V133"]) <= (data["freq_P_emaildomain"]))*1.)) + (data["freq_C1"]))/2.0)))))*1.)) * 2.0)) * 2.0)) * 2.0)) + (((data["card2"]) - ((((data["D3"]) > (data["freq_V133"]))*1.)))))) + (((data["card2"]) - ((((data["card2"]) > (data["C2"]))*1.)))))) +

                    0.095800*np.tanh(((((((((((((((data["freq_D1"]) + (data["card2"]))) > (data["C13"]))*1.)) > (((((((data["M6"]) > (data["freq_addr1"]))*1.)) <= (((((((data["month"]) + (data["V294"]))/2.0)) + (data["V317"]))/2.0)))*1.)))*1.)) * 2.0)) - ((((data["M6"]) > (((data["card2"]) + (data["freq_D1"]))))*1.)))) * 2.0)) +

                    0.076800*np.tanh(((data["freq_card2"]) + (((data["freq_TransactionAmt"]) * (((data["TransactionAmt"]) * (((data["freq_V127"]) - ((((((data["D10"]) > ((((data["freq_card2"]) <= (((data["D10"]) * (data["V283"]))))*1.)))*1.)) - (((data["freq_V45"]) + ((((data["freq_V317"]) > (((data["freq_C2"]) * 2.0)))*1.)))))))))))))) +

                    0.099325*np.tanh((((((((((((data["M5"]) > (data["dist2"]))*1.)) * 2.0)) * ((((data["freq_M5"]) <= (data["freq_P_emaildomain"]))*1.)))) - (((((((data["freq_C2"]) > (((data["freq_C11"]) + (data["C5"]))))*1.)) + (np.tanh((((((((data["freq_P_emaildomain"]) <= (data["freq_C11"]))*1.)) > (((data["freq_C11"]) + (data["freq_M5"]))))*1.)))))/2.0)))) * 2.0)) +

                    0.086464*np.tanh((((((data["freq_P_emaildomain"]) <= ((((data["freq_card5"]) + (data["V312"]))/2.0)))*1.)) - (((data["freq_card5"]) - ((((((data["freq_P_emaildomain"]) <= ((((data["freq_card5"]) + (data["V312"]))/2.0)))*1.)) + ((((((((((data["V312"]) + ((-1.0*(((((data["M5"]) > (((data["P_emaildomain"]) * 2.0)))*1.))))))/2.0)) * 2.0)) * 2.0)) * 2.0)))))))) +

                    0.096075*np.tanh(((((((data["card2"]) - ((((data["V313"]) <= ((((data["card2"]) <= ((((data["freq_C1"]) > (data["freq_dist2"]))*1.)))*1.)))*1.)))) - ((((data["D11"]) <= ((((data["V310"]) > (((((((data["D2"]) <= ((((data["R_emaildomain"]) <= ((((data["freq_C1"]) > (data["freq_dist2"]))*1.)))*1.)))*1.)) <= (data["freq_D1"]))*1.)))*1.)))*1.)))) * 2.0)) +

                    0.099996*np.tanh(((((data["P_emaildomain"]) + (((((data["C6"]) - ((((data["R_emaildomain"]) <= (((data["D3"]) * ((((data["freq_C6"]) <= (((data["freq_D4"]) + ((((data["C8"]) > (((data["P_emaildomain"]) * 2.0)))*1.)))))*1.)))))*1.)))) * 2.0)))) * 2.0)) +

                    0.099420*np.tanh((((((data["freq_P_emaildomain"]) <= (((data["freq_C6"]) * (data["freq_D5"]))))*1.)) + ((-1.0*(((((((data["dist2"]) <= (data["card2"]))*1.)) - (((((((((data["freq_C13"]) <= (data["card2"]))*1.)) - (((((((data["freq_C6"]) > (((data["freq_D5"]) * 2.0)))*1.)) + (data["D3"]))/2.0)))) + (data["freq_D5"]))/2.0))))))))) +

                    0.096954*np.tanh(((((((((((((((data["dist2"]) - ((((((((data["freq_C2"]) + (data["D1"]))) - (data["M5"]))) <= (((data["M5"]) / 2.0)))*1.)))) * 2.0)) - ((((data["M6"]) <= (((data["M5"]) / 2.0)))*1.)))) - ((((data["P_emaildomain"]) <= (((data["M5"]) / 2.0)))*1.)))) * 2.0)) * 2.0)) * 2.0)) +

                    0.0*np.tanh(((data["C10"]) + (((((((data["C10"]) + (((data["C10"]) + (((((((((data["freq_C5"]) <= (((((((data["freq_D14"]) * ((((data["freq_V83"]) > (data["C10"]))*1.)))) - (data["freq_V307"]))) / 2.0)))*1.)) + ((((data["D4"]) > (data["freq_D14"]))*1.)))/2.0)) - (data["freq_D15"]))))))) * 2.0)) * 2.0)))) +

                    0.0*np.tanh(((data["freq_D15"]) * ((((((((((((((data["freq_D11"]) <= ((((data["freq_C11"]) <= (data["C1"]))*1.)))*1.)) - (((((data["freq_V310"]) - ((((((((data["freq_C1"]) + (data["C1"]))) <= (data["C14"]))*1.)) * 2.0)))) * ((((data["freq_D11"]) > (data["card2"]))*1.)))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +

                    0.0*np.tanh(((((((((((((data["freq_C9"]) > ((((((data["freq_C6"]) <= (data["freq_P_emaildomain"]))*1.)) * ((((((data["freq_C11"]) + (data["freq_C9"]))/2.0)) * (data["freq_C11"]))))))*1.)) <= ((((((data["freq_C11"]) <= (data["freq_TransactionAmt"]))*1.)) * ((((data["freq_C14"]) > (data["freq_C2"]))*1.)))))*1.)) * 2.0)) * 2.0)) - (data["freq_D15"]))))
roc_auc_score(trainfreq.isFraud,GPI(trainfreq))
roc_auc_score(trainfreq.isFraud,GPII(trainfreq))
roc_auc_score(trainfreq.isFraud,np.sqrt(GPI(trainfreq)*GPII(trainfreq)))
test_predictions = pd.DataFrame()

test_predictions['TransactionID'] = test.TransactionID.values

test_predictions['isFraud'] = GPI(testfreq).values

test_predictions[['TransactionID','isFraud']].to_csv('gpsubmissionI.csv', index=False)
test_predictions = pd.DataFrame()

test_predictions['TransactionID'] = test.TransactionID.values

test_predictions['isFraud'] = GPII(testfreq).values

test_predictions[['TransactionID','isFraud']].to_csv('gpsubmissionII.csv', index=False)
test_predictions = pd.DataFrame()

test_predictions['TransactionID'] = test.TransactionID.values

test_predictions['isFraud'] = np.sqrt(GPI(testfreq).values*GPII(testfreq).values)

test_predictions[['TransactionID','isFraud']].to_csv('gpsubmissionmother.csv', index=False)
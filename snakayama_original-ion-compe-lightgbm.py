import numpy as np 

import pandas as pd

import math

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import gc

warnings.filterwarnings('ignore')

import lightgbm as lgb

from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split,StratifiedShuffleSplit

from sklearn import metrics

from tqdm import tqdm

from scipy import signal

pd.set_option('display.max_columns', 10000)

pd.set_option('display.max_rows', 10000)

import os
def read_data():

    print('Reading training, testing and submission data...')

    train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv')

    test = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv')

    submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time':str})

    print('Train set has {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Test set has {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    return train, test, submission



def get_batch(train, test):

    # concatenate data

    batch = 50

    total_batches = 14

    train['set'] = 'train'

    test['set'] = 'test'

    data = pd.concat([train, test])

    for i in range(int(total_batches)):

        data.loc[(data['time'] > i * batch) & (data['time'] <= (i + 1) * batch), 'batch'] = i + 1

    train = data[data['set'] == 'train']

    test = data[data['set'] == 'test']

    train.drop(['set'], inplace = True, axis = 1)

    test.drop(['set'], inplace = True, axis = 1)

    del data

    return train, test
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        if col!='open_channels':

            col_type = df[col].dtypes

            if col_type in numerics:

                c_min = df[col].min()

                c_max = df[col].max()

                if str(col_type)[:3] == 'int':

                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)  

                else:

                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                        df[col] = df[col].astype(np.float16)

                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                        df[col] = df[col].astype(np.float32)

                    else:

                        df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

def calc_gradients(s, n_grads = 4):

    '''

    Calculate gradients for a pandas series. Returns the same number of samples

    '''

    grads = pd.DataFrame()

    

    g = s.values

    for i in range(n_grads):

        g = np.gradient(g)

        grads['grad_' + str(i+1)] = g

        

    return grads
def calc_low_pass(s, n_filts=10):

    '''

    Applies low pass filters to the signal. Left delayed and no delayed

    '''

    wns = np.logspace(-2, -0.3, n_filts)

    

    low_pass = pd.DataFrame()

    x = s.values

    for wn in wns:

        b, a = signal.butter(1, Wn=wn, btype='low')

        zi = signal.lfilter_zi(b, a)

        low_pass['lowpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]

        low_pass['lowpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)

        

    return low_pass



def calc_high_pass(s, n_filts=10):

    '''

    Applies high pass filters to the signal. Left delayed and no delayed

    '''

    wns = np.logspace(-2, -0.1, n_filts)

    

    high_pass = pd.DataFrame()

    x = s.values

    for wn in wns:

        b, a = signal.butter(1, Wn=wn, btype='high')

        zi = signal.lfilter_zi(b, a)

        high_pass['highpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]

        high_pass['highpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)

        

    return high_pass



def calc_ewm(s, windows=[10, 50, 100, 500, 1000]):

    '''

    Calculates exponential weighted functions

    '''

    ewm = pd.DataFrame()

    for w in windows:

        ewm['ewm_mean_' + str(w)] = s.ewm(span=w, min_periods=1).mean()

        ewm['ewm_std_' + str(w)] = s.ewm(span=w, min_periods=1).std()

        

    # add zeros when na values (std)

    ewm = ewm.fillna(value=0)

        

    return ewm





def add_features(s):

    '''

    All calculations together

    '''

    

    gradients = calc_gradients(s)

    low_pass = calc_low_pass(s)

    high_pass = calc_high_pass(s)

    ewm = calc_ewm(s)

    

    return pd.concat([s, gradients, low_pass, high_pass, ewm], axis=1)





def divide_and_add_features(s, signal_size=500000):

    '''

    Divide the signal in bags of "signal_size".

    Normalize the data dividing it by 15.0

    '''

    # normalize

    s = s / 15.0

    

    ls = []

    for i in tqdm(range(int(s.shape[0]/signal_size))):

        sig = s[i*signal_size:(i+1)*signal_size].copy().reset_index(drop=True)

        sig_featured = add_features(sig)

        ls.append(sig_featured)

    

    return pd.concat(ls, axis=0)



def rolling_features(train, test):

    

    pre_train = train.copy()

    pre_test = test.copy()

    

        

    for df in [pre_train, pre_test]:

        

        df['lag_t1'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(1))

        df['lag_t2'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(2))

        df['lag_t3'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(3))

        

        df['lead_t1'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-1))

        df['lead_t2'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-2))

        df['lead_t3'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-3))

        df['lead_t4'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-4))

        df['lead_t5'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-5))

        df['lead_t6'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-6))

        df['lead_t7'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-7))

        df['lead_t8'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-8))

        df['lead_t9'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-9))

        df['lead_t10'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-10))

        df['lead_t11'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-11))

        df['lead_t12'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-12))

        df['lead_t13'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-13))

        df['lead_t14'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-14))

        df['lead_t15'] = df.groupby('batch')['signal'].transform(lambda x: x.shift(-15))

        

        df['sg_lead_t1'] = df.groupby('batch')['signal_sg'].transform(lambda x: x.shift(-1))

        df['sg_lead_t1'] = df.groupby('batch')['signal_sg'].transform(lambda x: x.shift(-2))

        df['sg_lead_t1'] = df.groupby('batch')['signal_sg'].transform(lambda x: x.shift(-3))

        

        

        

                

        for window in [1000, 5000]:

            

            # roll backwards

            df['signalmean_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).mean())

            df['signalstd_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).std())

            df['signalvar_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).var())

            df['signalmin_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).min())

            df['signalmax_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).max())

            min_max = (df['signal'] - df['signalmin_t' + str(window)]) / (df['signalmax_t' + str(window)] - df['signalmin_t' + str(window)])

            df['norm_t' + str(window)] = min_max * (np.floor(df['signalmax_t' + str(window)]) - np.ceil(df['signalmin_t' + str(window)]))

            

            # roll forward

            df['signalmean_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).mean())

            df['signalstd_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).std())

            df['signalvar_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).var())

            df['signalmin_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).min())

            df['signalmax_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).max())   

            min_max = (df['signal'] - df['signalmin_t' + str(window) + '_lead']) / (df['signalmax_t' + str(window) + '_lead'] - df['signalmin_t' + str(window) + '_lead'])

            df['norm_t' + str(window) + '_lead'] = min_max * (np.floor(df['signalmax_t' + str(window) + '_lead']) - np.ceil(df['signalmin_t' + str(window) + '_lead']))

            

    del train, test, min_max

    

    return pre_train, pre_test



def static_batch_features(df, n):

    

    df = df.copy()

    df.drop('batch', inplace = True, axis = 1)

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10000) - 1).values

    df['batch_' + str(n)] = df.index // n

    df['batch_index_' + str(n)] = df.index  - (df['batch_' + str(n)] * n)

    df['batch_slices_' + str(n)] = df['batch_index_' + str(n)]  // (n / 10)

    df['batch_slices2_' + str(n)] = df.apply(lambda r: '_'.join([str(r['batch_' + str(n)]).zfill(3), str(r['batch_slices_' + str(n)]).zfill(3)]), axis=1)



    for c in ['batch_' + str(n), 'batch_slices2_' + str(n)]:

        d = {}

        # -----------------------------------------------

        d['mean' + c] = df.groupby([c])['signal'].mean()

        d['median' + c] = df.groupby([c])['signal'].median()

        d['max' + c] = df.groupby([c])['signal'].max()

        d['min' + c] = df.groupby([c])['signal'].min()

        d['std' + c] = df.groupby([c])['signal'].std()

        d['p10' + c] = df.groupby([c])['signal'].apply(lambda x: np.percentile(x, 10))

        d['p25' + c] = df.groupby([c])['signal'].apply(lambda x: np.percentile(x, 25))

        d['p75' + c] = df.groupby([c])['signal'].apply(lambda x: np.percentile(x, 75))

        d['p90' + c] = df.groupby([c])['signal'].apply(lambda x: np.percentile(x, 90))

        d['skew' + c] = df.groupby([c])['signal'].apply(lambda x: pd.Series(x).skew())

        d['kurtosis' + c] = df.groupby([c])['signal'].apply(lambda x: pd.Series(x).kurtosis())

        min_max = (d['mean' + c] - d['min' + c]) / (d['max' + c] - d['min' + c])

        d['norm' + c] = min_max * (np.floor(d['max' + c]) - np.ceil(d['min' + c]))

        d['mean_abs_chg' + c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max' + c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min' + c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))

        d['range' + c] = d['max' + c] - d['min' + c]

        d['maxtomin' + c] = d['max' + c] / d['min' + c]

        d['abs_avg' + c] = (d['abs_min' + c] + d['abs_max' + c]) / 2

        # -----------------------------------------------

        for v in d:

            df[v] = df[c].map(d[v].to_dict())



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_' + str(n), 

                                                    'batch_index_' + str(n), 'batch_slices_' + str(n), 

                                                    'batch_slices2_' + str(n)]]:

        df[c + '_msignal'] = df[c] - df['signal']

        

    df.reset_index(drop = True, inplace = True)

        

    return df

def run_lgb(pre_train, pre_test, features, params):

    

#     feature_importance_df = pd.DataFrame()

#     kf = GroupKFold(n_splits = 5)

#     kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    kf = StratifiedShuffleSplit(n_splits = 1,random_state = 20)

    target = 'open_channels'

    oof_pred = np.zeros(len(pre_train))

    y_pred = np.zeros(len(pre_test))

     

    for fold, (tr_ind, val_ind) in enumerate(kf.split(pre_train, y=pre_train[target])):

        x_train, x_val = pre_train[features].iloc[tr_ind], pre_train[features].iloc[val_ind]

        y_train, y_val = pre_train[target][tr_ind], pre_train[target][val_ind]

        train_set = lgb.Dataset(x_train, y_train)

        val_set = lgb.Dataset(x_val, y_val)

        

        model = lgb.train(params, train_set, num_boost_round = 10000, early_stopping_rounds = 50, 

                         valid_sets = [train_set, val_set], verbose_eval = 100)

        

        oof_pred[val_ind] = model.predict(x_val)

        

        y_pred += model.predict(pre_test[features]) / kf.n_splits

        

#         fold_importance_df = pd.DataFrame()

#         fold_importance_df["feature"] = features

#         print(model.feature_importance(importance_type='gain').shape)

#         fold_importance_df["importance"] = model.feature_importance(importance_type='gain')

        

#         fold_importance_df["fold"] = fold + 1

#         feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        

    rmse_score = np.sqrt(metrics.mean_squared_error(pre_train[target], oof_pred))

    # want to clip and then round predictions (you can get a better performance using optimization to found the best cuts)

    oof_pred = np.round(np.clip(oof_pred, 0, 10)).astype(int)

    round_y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)

    f1 = metrics.f1_score(pre_train[target], oof_pred, average = 'macro')

    

    

    print(f'Our oof rmse score is {rmse_score}')

    print(f'Our oof macro f1 score is {f1}')

    return round_y_pred,feature_importance_df

train, test, submission = read_data()
from scipy.signal import savgol_filter as sg
w = 11

p = 2

train['signal_sg']  = sg(train.signal, w, p)

test['signal_sg']  = sg(test.signal, w, p)

train['signal_sg_{}'.format(1)]  = sg(train.signal, w,p, 1)

test['signal_sg_{}'.format(1)]  = sg(test.signal, w,p, 1)

train['signal_sg_{}'.format(2)]  = sg(train.signal, w,p, 2)

test['signal_sg_{}'.format(2)]  = sg(test.signal, w,p, 2)
train['signal_bool']=0

test['signal_bool']=0

train.loc[train['signal']>=0,'signal_bool'] = 1

test.loc[test['signal']>=0,'signal_bool'] = 1
import math



def sturges_rule(n):

    u"""

    スタージェスの公式

    """

    return round(1 + math.log2(n))
train['signal_bin'] = pd.cut(train['signal'], 23, labels=False)

test['signal_bin'] = pd.cut(test['signal'], 23, labels=False)
# data augmentation: add white noise

def add_white_noise(x, rate=0.002):

    return x + rate*np.random.randn(len(x))



# data augmentation: shift sound in timeframe

def shift_sound(x, rate=2):

    return np.roll(x, int(len(x)//rate))



# data augmentation: stretch sound

def stretch_sound(x, rate=1.1):

    input_length = len(x)

    x = librosa.effects.time_stretch(x, rate)

    if len(x)>input_length:

        return x[:input_length]

    else:

        return np.pad(x, (0, max(0, input_length - len(x))), "constant")

    

def calculate_melsp(x, n_fft=1024, hop_length=128):

    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2

    log_stft = librosa.power_to_db(stft)

    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)

    return melsp
import librosa

import librosa.display



train['signal_white_noise'] = add_white_noise(train['signal'].values)

test['signal_white_noise'] = add_white_noise(test['signal'].values)



train['signal_shift'] = shift_sound(train['signal'].values)

test['signal_shift'] = shift_sound(test['signal'].values)



train['signal_stretch'] = stretch_sound(train['signal'].values)

test['signal_stretch'] = stretch_sound(test['signal'].values)
BATCH_SIZE =4000

train['group'] = train.groupby(train.index//BATCH_SIZE, sort=False)['signal'].agg(['ngroup']).values

train['group'] = train['group'].astype(np.uint16)
def calc_low_pass(s, n_filts=10):

    '''

    Applies low pass filters to the signal. Left delayed and no delayed

    '''

    wns = np.logspace(-2, -0.3, n_filts)

    

    low_pass = pd.DataFrame()

    x = s.values

    for wn in wns:

        b, a = signal.butter(1, Wn=wn, btype='low')

        zi = signal.lfilter_zi(b, a)

        low_pass['lowpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]

        low_pass['lowpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)

        

    return low_pass



def calc_high_pass(s, n_filts=10):

    '''

    Applies high pass filters to the signal. Left delayed and no delayed

    '''

    wns = np.logspace(-2, -0.1, n_filts)

    

    high_pass = pd.DataFrame()

    x = s.values

    for wn in wns:

        b, a = signal.butter(1, Wn=wn, btype='high')

        zi = signal.lfilter_zi(b, a)

        high_pass['highpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]

        high_pass['highpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)

        

    return high_pass


def add_features(s):

    '''

    All calculations together

    '''

    

    low_pass = calc_low_pass(s)

    high_pass = calc_high_pass(s)

    

    return pd.concat([s, low_pass, high_pass], axis=1)





def divide_and_add_features(s, signal_size=500000):

    '''

    Divide the signal in bags of "signal_size".

    Normalize the data dividing it by 15.0

    '''

    # normalize

    s = s/15.0

    

    ls = []

    for i in tqdm(range(int(s.shape[0]/signal_size))):

        sig = s[i*signal_size:(i+1)*signal_size].copy().reset_index(drop=True)

        sig_featured = add_features(sig)

        ls.append(sig_featured)

    

    return pd.concat(ls, axis=0)
pre_train4 = divide_and_add_features(train['signal'])

pre_test4 = divide_and_add_features(test['signal'])



pre_train4.drop(['signal'], inplace = True, axis = 1)

pre_test4.drop(['signal'], inplace = True, axis = 1)



pre_train4.reset_index(inplace = True, drop = True)

pre_test4.reset_index(inplace = True, drop = True)



pre_train4 = reduce_mem_usage(pre_train4)

pre_test4 = reduce_mem_usage(pre_test4)



train, test = get_batch(train, test)

pre_train1, pre_test1 = rolling_features(train, test)

pre_train1 = reduce_mem_usage(pre_train1)

pre_test1 = reduce_mem_usage(pre_test1)

pre_train2 = static_batch_features(train, 25000)

pre_train2 = reduce_mem_usage(pre_train2)

pre_test2 = static_batch_features(test, 25000)

pre_test2 = reduce_mem_usage(pre_test2)



del train, test

gc.collect()



feat2 = [col for col in pre_train2.columns if col not in ['open_channels', 'signal', 'time', 'batch_25000', 

                                                          'batch_index_25000', 'batch_slices_25000', 'batch_slices2_25000']]

pre_train = pd.concat([pre_train1, pre_train2[feat2], pre_train4], axis = 1)

pre_test = pd.concat([pre_test1, pre_test2[feat2], pre_test4], axis = 1)

del pre_train1, pre_train2, pre_train4, pre_test1, pre_test2, pre_test4
features = [col for col in pre_train.columns if col not in ['open_channels', 'time', 'batch','group']]

features =features[0:10]

print('Training with {} features'.format(len(features)))



params = {'boosting_type': 'gbdt',

          'metric': 'rmse',

          'objective': 'regression',

          'n_jobs': -1,

          'seed': 236,

          'num_leaves': 280,

          'learning_rate': 0.026623466966581126,

          'max_depth': 80,

          'lambda_l1': 2.959759088169741,

          'lambda_l2': 1.331172832164913,

          'bagging_fraction': 0.9655406551472153,

          'bagging_freq': 9,

          'colsample_bytree': 0.6867118652742716}



round_y_pred,feature_importance = run_lgb(pre_train, pre_test, features, params)

submission['open_channels'] = round_y_pred

submission.to_csv('submission.csv', index = False)
feature_importance[["feature", "importance"]].groupby("feature", as_index=False).mean().sort_values(by="importance", ascending=False).head(100)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

print(os.listdir("../input"))

import nltk

from nltk.corpus import stopwords



from sklearn.metrics import f1_score

from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd 
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.shape
train_text = train_data['question_text']

test_text = test_data['question_text']

train_target = train_data['target']

all_text = train_text.append(test_text)
tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(all_text)



count_vectorizer = CountVectorizer()

count_vectorizer.fit(all_text)



train_text_features_cv = count_vectorizer.transform(train_text)

test_text_features_cv = count_vectorizer.transform(test_text)



train_text_features_tf = tfidf_vectorizer.transform(train_text)

test_text_features_tf = tfidf_vectorizer.transform(test_text)
kfold = KFold(n_splits = 5, shuffle = True, random_state = 2018)

test_preds = 0

oof_preds = np.zeros([train_data.shape[0],])



for i, (train_idx,valid_idx) in enumerate(kfold.split(train_data)):

    x_train, x_valid = train_text_features_tf[train_idx,:], train_text_features_tf[valid_idx,:]

    y_train, y_valid = train_target[train_idx], train_target[valid_idx]

    classifier = LogisticRegression()

    print('fitting.......')

    classifier.fit(x_train,y_train)

    print('predicting......')

    print('\n')

    oof_preds[valid_idx] = classifier.predict_proba(x_valid)[:,1]

    test_preds += 0.2*classifier.predict_proba(test_text_features_tf)[:,1]
pred_train = (oof_preds > .25).astype(np.int)

f1_score(train_target, pred_train)
submission1 = pd.DataFrame.from_dict({'qid': test_data['qid']})

submission1['prediction'] = (test_preds>0.25).astype(np.int)

# submission1.to_csv('logistic_submission.csv', index=False)

submission1['prediction'] = (test_preds>0.25)
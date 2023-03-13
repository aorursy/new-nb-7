import eli5
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
import re,string
train = pd.read_csv('../input/train.csv')
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()
ys = train[label_cols+['none']]
ys.head(n=3)
train['comment_text_char'] = train.comment_text.values
train.head(n=3)
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

# we need a custom pre-processor to extract correct field,
# but want to also use default scikit-learn preprocessing (e.g. lowercasing)
default_preprocessor = CountVectorizer().build_preprocessor()
def build_preprocessor(field):
    field_idx = list(train.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])
    
vectorizer = FeatureUnion([
    ('comment_text', TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1, preprocessor=build_preprocessor('comment_text'))),
    ('comment_text_char', TfidfVectorizer(sublinear_tf=True,  strip_accents='unicode',
               analyzer='char', stop_words='english', ngram_range=(2, 6), 
               max_features=20000, preprocessor=build_preprocessor('comment_text_char')))
])
X_train = vectorizer.fit_transform(train.values)
X_train
classifier = RidgeClassifier(solver='sag')

y = ys['toxic'].values

kf = KFold(n_splits=5, shuffle=True, random_state=239)
for train_index, test_index in kf.split(X_train):
    classifier = RidgeClassifier(solver='sag')
    classifier.fit(X_train[train_index], y[train_index])
    predict = classifier.decision_function(X_train[test_index])
    cv_score = roc_auc_score(y[test_index], predict)
    print(cv_score)
    break
eli5.show_weights(classifier, vec=vectorizer)
train[COMMENT].values[6]
eli5.show_prediction(classifier, doc=train.values[6], vec=vectorizer)
eli5.show_weights(classifier, vec=vectorizer, top=100, feature_filter=lambda x: x != '<BIAS>')
classifier = RidgeClassifier(solver='sag')

y = ys['identity_hate'].values

kf = KFold(n_splits=5, shuffle=True, random_state=239)
for train_index, test_index in kf.split(X_train):
    classifier = RidgeClassifier(solver='sag')
    classifier.fit(X_train[train_index], y[train_index])
    predict = classifier.decision_function(X_train[test_index])
    cv_score = roc_auc_score(y[test_index], predict)
    print(cv_score)
    break
eli5.show_weights(classifier, vec=vectorizer, top=100, feature_filter=lambda x: x != '<BIAS>')
train[COMMENT].values[42]
eli5.show_prediction(classifier, doc=train.values[42], vec=vectorizer)
#from langdetect import detect
#def dl(s):
#    try: return detect(s)
#    except: return 'en'
#train['lang'] = train.comment_text.apply(dl)
#train.lang.unique()
train[COMMENT].values[156771]
eli5.show_prediction(classifier, doc=train.values[156771], vec=vectorizer)
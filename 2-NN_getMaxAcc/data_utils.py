from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def load_data(preprocess):
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    # pprint(newsgroups_train.data[0])

    num_train = len(newsgroups_train.data)
    num_test = len(newsgroups_test.data)

    vectorizer = TfidfVectorizer(max_features=1024)

    X = vectorizer.fit_transform(newsgroups_train.data + newsgroups_test.data)
    X_train = X[0:num_train, :]
    X_test = X[num_train:num_train + num_test, :]

    Y_train = newsgroups_train.target
    Y_test = newsgroups_test.target

    # print('train X shape:' + X_train.shape, 'train Y shape:' + Y_train.shape)
    # print('test X shape:' + X_test.shape, 'test Y shape:' + Y_test.shape)
    if preprocess == "mean":
        X_train = np.mean(X_train, axis=0)
        X_test = np.mean(X_test, axis=0)
    elif preprocess == "std":
        X_train /= np.std(X_train, axis=0)
        X_test /= np.std(X_test, axis=0)

    print(X_train.shape, X_test.shape)
    return X_train, Y_train, X_test, Y_test

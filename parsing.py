import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gensim
import re
from tqdm import tqdm
# no emoji at the CS computers
import emoji
from pandas.api.types import CategoricalDtype
from pandas.plotting import scatter_matrix
from plotnine import *
import seaborn as sns
from feature_extraction import add_features
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

paths = ["data/ConanOBrien_tweets.csv", "data/cristiano_tweets.csv", "data/donaldTrump_tweets.csv",
         "data/ellenShow_tweets.csv",
         "data/jimmykimmel_tweets.csv", "data/joeBiden_tweets.csv", "data/KimKardashian_tweets.csv",
         "data/labronJames_tweets.csv",
         "data/ladygaga_tweets.csv", "data/Schwarzenegger_tweets.csv"]

names = ["Donald Trump", "Joe Biden", "Conan O'brien", "Ellen Degeneres",
         "Kim Kardashian", "Lebron James", "Lady Gaga", "Cristiano Ronaldo",
         "Jimmy kimmel", "Arnold schwarzenegger"]

_tweets = []


def get_tweets(path):
    return pd.read_csv(path)


def extract_hashtags(str):
    match = re.findall(r"#[a-zA-Z0-9]*", str)
    #  removes the # sign
    for i, tag in enumerate(match):
        match[i] = match[i][1:]
    return match


def extract_tags(str):
    match = re.findall(r"@[a-zA-Z0-9]*", str)
    #  removes the @ sign
    for i, tag in enumerate(match):
        match[i] = match[i][1:]
    return match


def extract_len(tweet):
    tweet = tweet.split(' ')
    return tweet


def split_tweet(tweets_list):
    empty = []
    for tweet in tweets_list:
        tweet = tweet.replace("[", "")
        tweet = tweet.replace("]", "")
        tweet = tweet.replace("\\", "")
        tweet = tweet.replace(",", "")
        tweet = tweet[1:-1]
        empty.append(tweet.split(' '))
    return empty


def get_tweet_len(tweets_list):
    empty = []
    for tweet in tweets_list:
        empty.append(len(tweet))
    return empty


def length(str):
    return len(str)


def counting(tweets):
    num_of_words = []
    for tweet in tweets:
        t, num, l, s = extract_len(tweet)
        num_of_words.append(num)
    return np.mean(num_of_words), np.var(num_of_words)


def count_emoji(tweet):
    return len(''.join(c for c in tweet if c in emoji.UNICODE_EMOJI))


def is_it_spanish(tweet):
    s = re.findall(r"(á|õ)", tweet)
    if len(s) > 0:
        return 1
    return 0


def count_link(tweet):
    s = re.findall(r"https:", tweet)
    return len(s)


def main():
    all_tweets = []
    num_of_words_var = []
    num_of_words_mean = []
    for path in paths:
        tweets = get_tweets(path)
        tweets = np.array(tweets['tweet'])
        mean, var = counting(tweets)
        num_of_words_mean.append(mean)
        num_of_words_var.append(var)
        all_tweets.append(tweets)
    print_graph(num_of_words_mean, "mean of numbers of words")
    print_graph(num_of_words_var, "var of numbers of words")


def get_longest_word(tweets_list):
    empty = []
    for tweet in tweets_list:
        longest = max(tweet, key=length)
        empty.append(len(longest))
    return empty


def get_shortest_word(tweets_list):
    empty = []
    for tweet in tweets_list:
        shortest = min(tweet, key=length)
        empty.append(len(shortest))
    return empty


def build_lang_model(sentences):
    model = gensim.models.Word2Vec(sentences)
    model.save('model.bin')
    words = list(model.wv.vocab)


def check_tweet(tweet):
    lst = []
    lst.append(len(extract_tags(tweet)))
    lst.append(len(extract_hashtags(tweet)))
    lst.append(count_emoji(tweet))
    lst.append(is_it_spanish(tweet))
    lst.append(count_link(tweet))
    lst.append(len(extract_len(tweet)))
    return lst


def build_data(pre_data):
    columns = ['tags', 'hashtags', 'emoji', 'spanish', 'links', 'length', 'label']
    rows = []
    pre_data = np.array(pre_data)
    for d in pre_data:
        l = d[0]
        f = check_tweet(d[1])
        f.append(l)
        rows.append(f)
    return pd.DataFrame(rows, columns=columns)


# build_data(get_tweets("data/ConanOBrien_tweets.csv"))

if __name__ == "__main__":
    # frames = [get_tweets(f) for f in paths]
    training_set = pd.read_csv("raw_data/train.csv")
    test_set = pd.read_csv("raw_data/test.csv")
    all_tweets_np = training_set.to_numpy()
    splat = split_tweet(all_tweets_np[:, 1])
    # all_tweets['broken_to_words'] = splat
    # all_tweets['number_of_words'] = get_tweet_len(all_tweets_np[:, 1])
    # all_tweets['longest_word_length'] = get_longest_word(all_tweets['broken_to_words'])
    # all_tweets['shortest_word_length'] = get_shortest_word(all_tweets['broken_to_words'])
    # all_tweets.to_csv()
    # all_tweets = add_features(all_tweets)
    train_features = build_data(training_set)
    test_features = build_data(test_set)
    # build_lang_model(splat)
    models = {
        # "bagging": BaggingClassifier(n_estimators=100, random_state=0),
        # "extraTree": ExtraTreesClassifier(n_estimators=100, random_state=0),
        "tree": DecisionTreeClassifier,
        "svc": SVC(gamma='auto'),
        # "Naive Bayes": GaussianNB()
    }
    labels = train_features["label"]
    for name, model in models.items():
        scores = []
        for i in tqdm(range(1, 11), desc=name):
            booster = AdaBoostClassifier(base_estimator=model(random_state=0, max_depth=i), n_estimators=i, learning_rate=1, algorithm='SAMME.R')
            booster.fit(train_features.drop("label", axis=1), train_features["label"])
            scores.append(booster.score(test_features.drop("label", axis=1), test_features["label"]))
        plt.figure()
        plt.title(name)
        plt.plot(range(1, 11), scores)
        plt.show()
    # sns.set(style="ticks")
    # sns.pairplot(features, hue="label")
    #
    # plt.show()

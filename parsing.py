import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pathes = ["ConanOBrien_tweets.csv", "cristiano_tweets.csv", "donaldTrump_tweets.csv", "ellenShow_tweets.csv",
          "jimmykimmel_tweets.csv", "joeBiden_tweets.csv", "KimKardashian_tweets.csv", "labronJames_tweets.csv",
          "ladygaga_tweets.csv", "Schwarzenegger_tweets.csv"]

names = ["Donald Trump",
         "Joe Biden",
         "Conan O'brien",
         "Ellen Degeneres",
         "Kim Kardashian",
         "Lebron James",
         "Lady Gaga",
         "Cristiano Ronaldo",
         "Jimmy kimmel",
         "Arnold schwarzenegger"]

_tweets = []


def get_tweets(path):
    path = "data/" + path

    return pd.read_csv(path)


def extract_hashtags():
    pass


def extract_tags():
    pass


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


def get_longest_word(tweets_list):
    empty = []
    for tweet in tweets_list:
        longest = max(tweet, key = length)
        empty.append(len(longest))
    return empty


def get_shortest_word(tweets_list):
    empty = []
    for tweet in tweets_list:
        shortest = min(tweet, key = length)
        empty.append(len(shortest))
    return empty


if __name__ == "__main__":
    frames = [get_tweets(f) for f in pathes]
    all_tweets = pd.concat(frames)
    all_tweets_np = all_tweets.to_numpy()
    all_tweets['broken_to_words'] = split_tweet(all_tweets_np[:, 1])
    all_tweets['number_of_words'] = get_tweet_len(all_tweets_np[:, 1])
    all_tweets['longest_word_length'] = get_longest_word(all_tweets['broken_to_words'])
    all_tweets['shortest_word_length'] = get_shortest_word(all_tweets['broken_to_words'])

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

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
    number_of_words = len(tweet)
    shortest_word_length = min(tweet, key=len)
    longest_word_length = max(tweet, key=len)
    return tweet, number_of_words, longest_word_length, shortest_word_length


def main():
    for path in pathes:
        tweets = get_tweets(path)

extract_tags()
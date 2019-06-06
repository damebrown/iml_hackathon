import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
# no emoji at the CS computers
import emoji

pathes = ["data/ConanOBrien_tweets.csv", "data/cristiano_tweets.csv", "data/donaldTrump_tweets.csv",
          "data/ellenShow_tweets.csv",
          "data/jimmykimmel_tweets.csv", "data/joeBiden_tweets.csv", "data/KimKardashian_tweets.csv",
          "data/labronJames_tweets.csv",
          "data/ladygaga_tweets.csv", "data/Schwarzenegger_tweets.csv"]

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


def print_graph(y_asix, header):
    plt.plot(names, y_asix)
    plt.title(header)
    plt.xlabel("tweeters users")
    plt.xticks(rotation = 90)
    plt.legend()
    plt.savefig(header)
    plt.show()


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
    return len(s) > 0


def main():
    all_tweets = []
    num_of_words_var = []
    num_of_words_mean = []
    for path in pathes:
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

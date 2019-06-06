import pandas as pd
import numpy as np
import re


def calc_avg_sentence_length(row):
    tweet = row["tweet"]
    sentences = re.split(r'[.!?\n]', tweet)
    return np.average([len(sentence.split()) for sentence in sentences])


def word_count(row):
    words, hashtags, ats = 0, 0, 0
    for word in row['tweet'].split():
        if '@' in word:
            ats += 1
        elif '#' in word:
            hashtags += 1
        else:
            words += 1
    return pd.Series([words, hashtags, ats])


def add_features(df):
    df["avg_sentence_len"] = df.apply(calc_avg_sentence_length, axis=1)
    new_vals = df.apply(word_count, axis=1)
    df["word_count"], df["hashtag_count"], df["ats_count"] = new_vals[0].values, new_vals[1].values, new_vals[2].values
    return df


tweet = "this is a long tweet. is it composed of many words! and even\nenters! woohoo"

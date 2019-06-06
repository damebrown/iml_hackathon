import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os


def read_all_files(dir):
    result = []
    for filename in tqdm(os.listdir(dir), desc="reading"):
        if filename.endswith("tweets.csv"):
            result.append(pd.read_csv(os.path.join(dir, filename)))
    return result


def split_train_test(*args, ratio=0.75):
    train = None
    test = None
    for data in tqdm(args, desc="spliting"):
        new_train = data.sample(frac=ratio)
        new_test = data.drop(new_train.index)
        if train is None:
            train = new_train
        else:
            train = pd.concat([train, new_train], sort=False)
        if test is None:
            test = new_test
        else:
            test = pd.concat([test, new_test], sort=False)
    return train, test


def data_scrambler():
    data_file_dir = "data"
    data_dest_dir = "raw_data"
    data_array = read_all_files(data_file_dir)
    train_data, test_data = split_train_test(*data_array, ratio=0.75)
    train_data.to_csv(os.path.join(data_dest_dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(data_dest_dir, "test.csv"), index=False)


def read_raw_data():
    return pd.read_csv(os.path.join("raw_data", "train.csv")), pd.read_csv(os.path.join("raw_data", "test.csv"))


def plot_all_features():
    pass


if __name__ == '__main__':
    data_scrambler()

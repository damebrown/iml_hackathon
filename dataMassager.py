import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os


def read_all_files(dir):
    result = []
    for filename in tqdm(os.listdir(dir)):
        if filename.endswith(".csv") and "tweets" in filename:
            result.append(pd.read_csv(os.path.join(dir, filename)))
    return result


def split_train_test(*args, ratio=0.75):
    train = None
    test = None
    for data in args:
        # data = data.reindex(np.random.permutation(data.index))
        new_train = data.sample(frac=ratio)
        new_test = data.drop(new_train.index)
        if not train:
            train = new_train
        else:
            train = pd.concat(train, new_train)
        if not test:
            test = new_test
        else:
            train = pd.concat(test, new_test)
    return train, test


if __name__ == '__main__':
    data_file_dir = sys.argv[1]
    data_dest_dir = sys.argv[2]

    data_array = read_all_files(data_file_dir)

    train_data, test_data = split_train_test(*data_array, ratio=0.75)


import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self, config):
        self.config = config
        self.cfg_preprocess = config["preprocess"]

        self.n_sample = self.config["n_sample"]

        self.train_data = None

    def __preprocessing(self, data):
        self.config["n_users"] = data["user_id"].nunique()
        self.config["n_items"] = data["item_id"].nunique()

        items = data["item_id"].unique()
        data = data.groupby("user_id")["item_id"].apply(list).reset_index(name='positive')
        data["negative"] = data["positive"].apply(lambda x: list(set(items) - set(x)))
        data = data.explode("positive")
        data["negative"] = data["negative"].apply(lambda x: np.random.choice(x, self.n_sample))
        data = data.explode("negative")
        # data = data.sample(frac=1).reset_index(drop=True)

        return data

    def load_data_from_file(self):
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_csv("./data/u.data", delimiter='\t', names=names)
        data = data.drop(["rating", "timestamp"], axis=1)

        return data

    def load_train_data(self):
        self.train_data = self.load_data_from_file()
        self.train_data = self.__preprocessing(self.train_data)

        return self.train_data

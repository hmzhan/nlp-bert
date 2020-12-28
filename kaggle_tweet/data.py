import pandas as pd


class TweetData:
    def __init__(self):
        self.train = None
        self.test = None
        self.submission = None

    def load_data(self):
        """
        Load tweeter data
        :return: None
        """
        self.train = pd.read_csv('/kaggle_data/train.csv')
        self.test = pd.read_csv('/kaggle_data/test.csv')
        self.submission = pd.read_csv('/kaggle_data/sample_submission.csv')

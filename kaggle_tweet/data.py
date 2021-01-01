import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
        self.train = pd.read_csv('kaggle_data/train.csv')
        self.test = pd.read_csv('kaggle_data/test.csv')
        self.submission = pd.read_csv('kaggle_data/sample_submission.csv')

    def class_dist_eda(self):
        """
        Plot distribution of two classes
        :return: None
        """
        pos = self.train[self.train['target'] == 1].shape[0]
        neg = self.train[self.train['target'] == 0].shape[0]
        plt.rcParams['figure.figsize'] = (7, 5)
        plt.bar(10, pos, 3, label='Real', color='blue')
        plt.bar(15, neg, 3, label='Not', color='red')
        plt.legend()
        plt.ylabel('Number of examples')
        plt.title('Distribution of real and not real disaster tweets')
        plt.show()

    def num_char_eda(self):
        """
        Plot distribution of number of characters
        :return: None
        """
        pos_len = self.train[self.train['target'] == 1]['text'].apply(lambda x: len(x))
        neg_len = self.train[self.train['target'] == 0]['text'].apply(lambda x: len(x))
        plt.rcParams['figure.figsize'] = (18, 6)
        bins = 150
        plt.hist(pos_len, alpha=0.6, bins=bins, label='Not Real')
        plt.hist(neg_len, alpha=0.8, bins=bins, label='Real')
        plt.xlabel('length')
        plt.ylabel('numbers')
        plt.legend(loc='upper right')
        plt.xlim(0, 150)
        plt.grid()
        plt.show()

    def num_char_eda2(self):
        """
        Plot two distributions of number of characters
        :return: None
        """
        fig, (ax1, ax2) = plt.subplot(1, 2, figsize=(10, 5))
        pos_len = self.train[self.train['target'] == 1]['text'].str.len()
        neg_len = self.train[self.train['target'] == 0]['text'].str.len()
        ax1.hist(pos_len, color='blue')
        ax1.set_title('Disaster tweets')
        ax2.hist(neg_len, color='red')
        ax2.set_title('Not disaster tweets')
        fig.suptitle('Number of char in tweets')
        plt.show()

    def num_word_eda(self):
        """
        Plot two distributions of number of words
        :return: None
        """
        fig, (ax1, ax2) = plt.subplot(1, 2, figsize=(10, 5))
        pos_len = self.train[self.train['target'] == 1]['text'].apply(lambda x: len(x.split()))
        neg_len = self.train[self.train['target'] == 0]['text'].apply(lambda x: len(x.split()))
        ax1.hist(pos_len, color='blue')
        ax1.set_title('Disaster tweets')
        ax2.hist(neg_len, color='red')
        ax2.set_title('Not disaster tweets')
        fig.suptitle('Number of words in tweets')
        plt.show()

    def word_length_eda(self):
        """
        Plot two distributions of average word length
        :return: None
        """
        fig, (ax1, ax2) = plt.subplot(1, 2, figsize=(10, 5))
        pos_word = self.train[self.train['target'] == 1]['text'].apply(lambda x: np.mean(len(w) for w in x.split()))
        neg_word = self.train[self.train['target'] == 0]['text'].apply(lambda x: np.mean(len(w) for w in x.split()))
        sns.distplot(pos_word, ax1, color='blue')
        sns.displot(neg_word, ax2, color='red')
        ax1.set_title('Disaster')
        ax2.set_title('Not disaster')
        fig.suptitle('Average word length in each tweet')



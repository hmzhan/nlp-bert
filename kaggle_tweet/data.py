import re
import string
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from wordcloud import WordCloud


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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        pos_word = self.train[self.train['target'] == 1]['text'].apply(lambda x: np.mean([len(w) for w in x.split()]))
        neg_word = self.train[self.train['target'] == 0]['text'].apply(lambda x: np.mean([len(w) for w in x.split()]))
        sns.distplot(pos_word, ax1, color='blue')
        sns.displot(neg_word, ax2, color='red')
        ax1.set_title('Disaster')
        ax2.set_title('Not disaster')
        fig.suptitle('Average word length in each tweet')

    def create_corpus(self, target):
        """
        Create corpus from training or test data
        :param target: target label
        :return: list of words
        """
        corpus = []
        for x in self.train[self.train['target'] == target]['text'].str.split():
            for i in x:
                corpus.append(i)
        return corpus

    @staticmethod
    def stop_word_eda(corpus):
        """
        Plot distribution of top stop words
        :param corpus: corpus
        :return: None
        """
        stop = set(stopwords.words('english'))
        d = defaultdict(int)
        for word in corpus:
            if word in stop:
                d[word] += 1
        top = sorted(d.items(), key=lambda x: x[1], reverse=True)[:10]
        plt.rcParams['figure.figsize'] = (18, 6)
        x, y = zip(*top)
        plt.bar(x, y)

    @staticmethod
    def punctuation_eda(corpus):
        """
        Plot distribution of top punctuations
        :param corpus: corpus
        :return: None
        """
        d = defaultdict()
        special = string.punctuation
        for x in corpus:
            if x in special:
                d[x] += 1
        x, y = zip(*d.items())
        plt.bar(x, y, color='green')

    @staticmethod
    def common_word_eda(corpus):
        """
        Plot distribution of most common words
        :param corpus: corpus
        :return: None
        """
        counter = Counter(corpus)
        most_common = counter.most_common(40)
        x = []
        y = []
        for word, count in most_common:
            if word not in set(stopwords.words('english')):
                x.append(word)
                y.append(count)
        plt.figure(figsize=(16, 5))
        sns.barplot(x=y, y=x)

    @staticmethod
    def top_tweet_bigrams(corpus, n=None):
        """
        N-gram analysis
        :param corpus: corpus
        :param n: top n tweet
        :return: None
        """
        vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        plt.figure(figsize=(16, 5))
        x, y = map(list, zip(*words_freq[:10]))
        sns.barplot(x=y, y=x)

    @staticmethod
    def remove_url(text):
        """
        Remove URL from text
        :param text: text input
        :return: cleaned text
        """
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)

    @staticmethod
    def remove_html(text):
        """
        Remove html from text
        :param text: text input
        :return: cleaned text
        """
        html = re.compile(r'<.*?>')
        return html.sub(r'', text)

    @staticmethod
    def remove_emoji(text):
        """
        Remove emoji from text
        :param text: text input
        :return: cleaned text
        """
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  
            u"\U0001F300-\U0001F5FF"  
            u"\U0001F680-\U0001F6FF"  
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def remove_punct(text):
        """
        Remove punctuations from text
        :param text: text input
        :return: cleaned text
        """
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

    def text_clean(self):
        """
        Clean text input
        :return: cleaned text
        """
        df = pd.concat([self.train, self.test])
        df['text'] = df['text'].apply(lambda x: self.remove_url(x))
        df['text'] = df['text'].apply(lambda x: self.remove_html(x))
        df['text'] = df['text'].apply(lambda x: self.remove_emoji(x))
        df['text'] = df['text'].apply(lambda x: self.remove_punct(x))
        return df

    @staticmethod
    def word_cloud_eda(corpus):
        """
        Plot word cloud from corpus
        :param corpus: corpus
        :return: None
        """
        word_cloud = WordCloud(
            background_color='black',
            max_font_size=80
        ).generate(''.join(corpus[:50]))
        plt.figure(figsize=(12, 8))
        plt.imshow(word_cloud)
        plt.axis('off')
        plt.show()

    @staticmethod
    def cv(data):
        """
        Count Vectorizer
        :param data: input text
        :return: embeddings, counter vectorizer
        """
        count_vec = CountVectorizer()
        embeddings = count_vec.fit_transform(data)
        return embeddings, count_vec

    @staticmethod
    def tfidf(data):
        """
        TFIDF transformation
        :param data: input text
        :return: embeddings, tfidf vectorizer
        """
        tfidf_vec = TfidfVectorizer()
        embeddings = tfidf_vec.fit_transform(data)
        return embeddings, tfidf_vec

    @staticmethod
    def plot_lsa(data, labels, plot=True):
        """
        Visualizing the embeddings
        :param data: embeddings
        :param labels: label data
        :param plot: True or False
        :return: None
        """
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(data)
        lsa_scores = lsa.transform(data)
        color_mapper = {label: idx for idx, label in enumerate(set(labels))}
        color_column = [color_mapper[label] for label in labels]
        colors = ['orange', 'blue']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:, 1], s=8, alpha=.8,
                        c=labels, cmap=matplotlib.colors.ListedColormap(colors))
            orange_patch = mpatches.Patch(color='orange', label='Not')
            blue_patch = mpatches.Patch(color='blue', label='Real')
            plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})





















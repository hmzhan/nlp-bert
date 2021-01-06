from tqdm import tqdm
from nltk.tokenize import word_tokenize
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout
from keras.initializers import Constant
from keras.optimizers import Adam

class GloveTweeterModel:
    def __init__(self):
        pass

    def create_corpus_new(self, df):
        """
        Create corpus
        :return: corpus
        """
        corpus = []
        for tweet in tqdm(df['text']):
            words = [word.lower() for word in word_tokenize(tweet)]
            corpus.append(words)
        return corpus

    def load_glove_embeddings(self):
        """
        Load and clean glove embeddings
        :return: embeddings
        """
        embedding_dict = {}
        with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt', 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vectors = np.asarray(values[1:], 'float32')
                embedding_dict[word] = vectors
        f.close()
        return embedding_dict

    def create_emb_matrix(self):
        """
        Create embedding matrix
        :return: embedding matrix
        """
        MAX_LEN = 50
        tokenizer_obj = Tokenizer()



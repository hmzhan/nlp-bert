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

    @staticmethod
    def load_glove_embeddings():
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

    @staticmethod
    def create_emb_matrix(corpus, embedding_dict):
        """
        Create embedding matrix
        :return: embedding matrix
        """
        MAX_LEN = 50
        tokenizer_obj = Tokenizer()
        tokenizer_obj.fit_on_texts(corpus)
        sequences = tokenizer_obj.texts_to_sequences(corpus)
        tweet_pad = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
        word_index = tokenizer_obj.word_index
        num_words = len(word_index) + 1
        embedding_matrix = np.zeros((num_words, 100))
        for word, i in tqdm(word_index.items()):
            if i < num_words:
                embedding_vec = embedding_dict.get(word)
                if embedding_vec is not None:
                    embedding_matrix[i] = embedding_vec
        return embedding_matrix

    @staticmethod
    def build_model(num_words, embedding_matrix, MAX_LEN):
        """
        Build Glove model based on Keras
        :return:
        """
        embedding = Embedding(
            num_words, 100,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=MAX_LEN,
            trainable=False
        )
        model = Sequential()
        model.add(embedding)
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate=3e-4)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model



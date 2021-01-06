import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp.bert import tokenization


class BertTweeterModel:
    def __init__(self, url):
        self.bert_layer = hub.KerasLayer(url, trainable=True)
        self.tokenizer = None
        self.model = None

    def create_tokenizer(self):
        """
        Create tokenizer
        :return: None
        """
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def bert_encode(self, texts, max_len=512):
        """
        Generate tokens, masks, and segments
        :param texts: input text data
        :param max_len: maximum length of input text sequence
        :return: tokens, masks, and segments
        """
        all_tokens = []
        all_masks = []
        all_segments = []

        for text in texts:
            text = self.tokenizer.tokenize(text)
            text = text[:max_len-2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = max_len - len(input_sequence)

            tokens = self.tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len   # token id
            pad_masks = [1] * len(input_sequence) + [0] * pad_len  # mask id
            segment_ids = [0] * max_len   # segment id

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    def build_model(self, max_len=512):
        """
        Build BERT classification model
        :param max_len: maximum length of input text
        :return: model
        """
        input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_mask')
        segment_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='segment_ids')

        _, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        out = tf.keras.layers.Dense(1, activation='sigmoid')(clf_output)

        self.model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        self.model.compile(tf.keras.optimizers.Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])






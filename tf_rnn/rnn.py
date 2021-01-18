
import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()


class RNNTweetModel:
    def __init__(self):
        self.BUFFFER_SIZE = 10000
        self.BATCH_SIZE = 64
        self.train_dataset = None
        self.test_dataset = None
        self.encoder = None
        self.model = None

    def load_data(self):
        """
        Load IMDB movie review data
        :return: None
        """
        dataset, info = tfds.load('imdb_reviews',
                                  with_info=True,
                                  as_supervised=True)
        train = dataset['train']
        test = dataset['test']
        self.train_dataset = train.shuffle(self.BUFFFER_SIZE).batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        self.test_dataset = test.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def create_encoder(self):
        """
        Create encoder
        :return: None
        """
        VOCAB_SIZE = 1000
        self.encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=VOCAB_SIZE
        )
        self.encoder.adapt(self.train_dataset.map(lambda text, label: text))

    def build_model(self):
        """
        Build RNN model
        :return: None
        """
        self.model = tf.keras.Sequential([
            self.encoder,
            tf.keras.layers.Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=64,
                mask_zero=True
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=['accuracy']
        )


if __name__ == '__main__':
    rnn_model = RNNTweetModel()
    rnn_model.load_data()
    rnn_model.create_encoder()
    rnn_model.build_model()

    history = rnn_model.model.fit(
        rnn_model.train_dataset,
        epochs=10,
        validation_data=rnn_model.test_dataset,
        validation_steps=30
    )











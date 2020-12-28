import os
import shutil
import tensorflow as tf


class ImdbData:
    """
    Load and process IMDB data
    """
    def __init__(self, url, batch_size, split_ratio, seed):
        self.url = url
        self.batch_size = batch_size
        self.seed = seed
        self.split_ratio = split_ratio
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.class_names = None

    def load_data(self):
        """
        Download and extract the IMDB dataset
        :return: None
        """
        data_set = tf.keras.utils.get_file(
            'aclImdb_v1.tar.gz', self.url,
            untar=True, cache_dir='.',
            cache_subdir=''
        )
        dataset_dir = os.path.join(os.path.dirname(data_set), 'aclImdb')
        train_dir = os.path.join(dataset_dir, 'train')
        remove_dir = os.path.join(train_dir, 'unsup')
        shutil.rmtree(remove_dir)

    def split_data(self):
        """
        Split data into training, validation, and test data
        :return: None
        """
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            'aclImdb/train',
            batch_size=self.batch_size,
            validation_split=self.split_ratio,
            subset='training',
            seed=self.seed
        )
        self.class_names = raw_train_ds.class_names
        self.train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

        val_ds = tf.keras.preprocessing.text_dataset_from_directory(
            'aclImdb/train',
            batch_size=self.batch_size,
            validation_split=self.split_ratio,
            subset='validation',
            seed=self.seed
        )
        self.val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        test_ds = tf.keras.preprocessing.text_dataset_from_directory(
            'aclImdb/test',
            batch_size=self.batch_size
        )
        self.test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


if __name__ == '__main__':
    URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    BATCH_SIZE = 32
    VAL_RATIO = 0.2
    SEED = 42
    imdb = ImdbData(URL, BATCH_SIZE, VAL_RATIO, SEED)
    imdb.load_data()
    imdb.split_data()

    # print a few examples
    for text_batch, label_batch in imdb.train_ds.take(1):
        for i in range(3):
            print(f'Review: {text_batch.numpy()[i]}')
            label = label_batch.numpy()[i]
            print(f'Label : {label} ({imdb.class_names[label]})')





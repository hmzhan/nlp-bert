
from tf_bert.data import ImdbData
from tf_bert.model import BertModel

URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
BATCH_SIZE = 32
VAL_RATIO = 0.2
SEED = 42

BERT_MODEL_NAME = 'small_bert/bert_en_uncased_L-4_H-512_A-8'

if __name__ == '__main__':
    # DATA
    imdb = ImdbData(URL, BATCH_SIZE, VAL_RATIO, SEED)
    imdb.load_data()
    imdb.split_data()

    # MODEL
    bert_model = BertModel()
    bert_model.select_bert(BERT_MODEL_NAME)
    bert_model.build_classifier_model()
    bert_model.compile_classifier_model(imdb.train_ds)

    train_hist = bert_model.model.fit(
        x=imdb.train_ds,
        validation_data=imdb.val_ds,
        epochs=5
    )

    loss, accuracy = bert_model.model.evaluate(imdb.test_ds)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')





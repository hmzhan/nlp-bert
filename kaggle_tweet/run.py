from kaggle_tweet.data import TweetData
from kaggle_tweet.model import BertTweeterModel

URL = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1'

if __name__ == '__main__':
    # DATA
    tweet = TweetData()
    tweet.load_data()

    # MODEL
    bert_model = BertTweeterModel(URL)
    bert_model.create_tokenizer()

    train_input = bert_model.bert_encode(tweet.train['text'].values, max_len=160)
    train_label = tweet.train['target'].values
    test_input = bert_model.bert_encode(tweet.test['text'].values, max_len=160)

    train_hist = bert_model.model.fit(
        train_input,
        train_label,
        validation_split=0.2,
        epochs=3,
        batch_size=16
    )

    # PREDICTIONS
    pred = bert_model.model.predict(test_input)
    tweet.submission['target'] = pred.round().astype(int)
    tweet.submission.to_csv('submission.csv', index=False)

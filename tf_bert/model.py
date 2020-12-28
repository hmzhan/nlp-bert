import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
from tf_bert.bert_settings import map_name_to_handle, map_model_to_preprocess


class BertModel:
    """
    Build BERT models
    """
    def __init__(self):
        self.tfhub_handle_encoder = None
        self.tfhub_handle_preprocess = None
        self.model = None

    def select_bert(self, model_name):
        """
        Select BERT model from tf hub
        :param model_name: BERT model name
        :return: None
        """
        self.tfhub_handle_encoder = map_name_to_handle[model_name]
        self.tfhub_handle_preprocess = map_model_to_preprocess[model_name]
        print(f'BERT model selected: {self.tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {self.tfhub_handle_preprocess}')

    def build_classifier_model(self):
        """
        Build BERT model
        :return: None
        """
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        self.model = tf.keras.Model(inputs=text_input, outputs=net)

    def compile_classifier_model(self, train_ds):
        """
        Compile BERT classification model
        :return:
        """
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()
        epochs = 5
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1 * num_train_steps)
        init_lr = 3e-5
        optimizer = optimization.create_optimizer(
            init_lr=init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type='adamw'
        )
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

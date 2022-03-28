import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import TextVectorization, Dense, LSTM, Bidirectional, Add
from keras.layers.embeddings import Embedding
from keras.models import Model
from tensorflow.keras import initializers

from dialogi import utils

F1_SCORE_AT_THRESHOLD = []
for threshold in np.arange(0.5, 0.96, 0.01):
    F1_SCORE_AT_THRESHOLD.append(tfa.metrics.F1Score(1, 'macro', threshold, name=f'f1_score_{threshold}'))


class BaseClassifier:
    def __init__(self, **kwargs):
        self.textvector_layer = TextVectorization(**kwargs)
        self.basepath = utils.get_data_path('embedded')/'glove_wiki_embeddings'

        self.mltdim_embed_index = {}
        for glove_dim in [50, 100, 200, 300]:
            self.mltdim_embed_index[glove_dim] = {}

            glove_path = self.basepath/f'glove.6B.{glove_dim}d.txt'
            with open(glove_path) as f_glove:
                for line in f_glove:
                    word, vec = line.split(maxsplit=1)
                    vec = np.fromstring(vec, dtype=float, sep=' ')
                    self.mltdim_embed_index[glove_dim][word] = vec

        self.embedding_layer = None
        self.is_initialised = False

    def adapt_vector_layer(self, proc_texts=None, dim=None):
        if proc_texts is not None:
            # Run this on the CPU as, sometimes, copying to GPU throws an error.
            with tf.device("/CPU:0"):
                self.textvector_layer = TextVectorization(standardize=None, ngrams=1, output_mode='int')
                self.textvector_layer.adapt(proc_texts)

                # The model's called once before starting training. No texts have been provided when this happens.
                # This is here to handle this case and return a dummy embedding layer, so that no error is raised.
                if not self.is_initialised:
                    self.is_initialised = True

        if dim is not None:
            if not self.is_initialised:
                self.embedding_layer = Embedding(1, 1)     # Return a dummy embedding layer, when not initialised.
                return

            vocab = self.textvector_layer.get_vocabulary()
            word_index = dict(zip(vocab, range(len(vocab))))
            num_tokens = len(vocab) + 2

            embed_index = self.mltdim_embed_index[dim]
            embed_matrix = np.zeros((num_tokens, dim))

            for word, idx in word_index.items():
                embed_vector = embed_index.get(word)
                if embed_vector is not None:
                    embed_matrix[idx] = embed_vector

            embed_tensor = initializers.Constant(embed_matrix)
            self.embedding_layer = Embedding(num_tokens, dim, embeddings_initializer=embed_tensor, trainable=True)


class LSTMClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(standardize=None, ngrams=1, output_mode='int')
        self.layer_shapes = [0, 0, 0]   # Used for the extended model: [embedding_dim, bi_lstm_units, dense_units]
        self.extended_learn_rate = 1e-2

    def set_shape(self, layer_shapes):
        self.layer_shapes = layer_shapes

    def get_model(self, hp):
        hp_dict = {
            'embedding_dim': hp.Choice('embedding_dim', values=[50, 100, 200, 300]),
            'bi_lstm_units': hp.Int('bi_lstm_units', min_value=32,  max_value=96,  step=16),
            'dense_units':   hp.Int('dense_units',   min_value=192, max_value=320, step=32),
            'learning_rate': hp.Choice('learning_rate', values=[1e-3, 5e-3, 7e-3, 1e-2]),
        }

        inputs = {
            'text': tf.keras.Input(shape=(1,),      dtype=tf.string ),
            'vecs': tf.keras.Input(shape=(1, 450,), dtype=tf.float32),
        }

        self.adapt_vector_layer(dim=hp_dict['embedding_dim'])
        text_vector = self.textvector_layer(inputs['text'])
        embedding_layer = self.embedding_layer(text_vector)

        uni_lstm_units = int(hp_dict['bi_lstm_units'] * 0.5)
        lstm_layer = Bidirectional(LSTM(units=uni_lstm_units))(embedding_layer)

        dense_layer = Dense(units=hp_dict['dense_units'], activation='relu')(lstm_layer)
        output = Dense(1, activation='sigmoid')(dense_layer)

        model = Model(list(inputs.values()), output)
        model.compile(
            loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=hp_dict['learning_rate']),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), *F1_SCORE_AT_THRESHOLD])

        return model

    def get_model_ef(self):
        inputs = {
            'text': tf.keras.Input(shape=(1,),       dtype=tf.string ),
            'vecs': tf.keras.Input(shape=(1,  450,), dtype=tf.float32),
        }

        self.adapt_vector_layer(dim=self.layer_shapes[0])
        text_vector = self.textvector_layer(inputs['text'])
        embedding_layer = self.embedding_layer(text_vector)

        uni_lstm_units = int(self.layer_shapes[1] * 0.5)
        lstm_layer = Bidirectional(LSTM(units=uni_lstm_units))(embedding_layer)
        dense_text = Dense(units=self.layer_shapes[2], activation='relu')(lstm_layer)

        vecs = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(inputs['vecs'])
        dense_vecs = Dense(units=self.layer_shapes[2], activation='relu')(vecs)

        paper_vector = Add()([dense_text, dense_vecs])
        output = Dense(1, activation='sigmoid')(paper_vector)

        model_ef = Model(list(inputs.values()), output)
        model_ef = self.compile_model_default(model_ef)

        return model_ef

    def compile_model_default(self, model):
        model.compile(
            loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=self.extended_learn_rate),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), *F1_SCORE_AT_THRESHOLD])

        return model


if __name__ == '__main__':
    pass

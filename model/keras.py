import numpy as np
import pandas as pd
import tensorflow as tf


class MF:
    def __init__(self, data_path):
        self.data = np.genfromtxt(data_path,
                                  delimiter='\t',
                                  skip_header=True,
                                  dtype=np.int32)
        self.model = None
        self.result = None

    def train(self,
              dim=2,
              alpha=1e-3,
              beta=1e-4,
              epoch=300,
              batch=1000,
              verbose=2):
        self.build(dim, alpha, beta)
        self._train(epoch,
                    batch,
                    verbose)
        self.predict()

    def predict(self):
        predicted = self.model.predict(
            [self.data[:, 0], self.data[:, 1]]
        ).reshape(-1)
        pd.DataFrame({'actual': self.data[:, 2],
                      'predicted': predicted})\
            .to_csv('result/predict_keras.csv', index=False)

    def recommend(self, num_rec_items):
        """
        do recommend num_rec_items items excluding the observed items.
        """
        user_mtx, item_mtx = self.model.get_weights()
        predicted = np.inner(user_mtx, item_mtx) * -1
        predicted[self.data[:, 0], self.data[:, 1]] *= 0
        self.result = pd.DataFrame(predicted.argsort()[:, :num_rec_items],
                                   columns=['top%s' % i
                                            for i in range(1, num_rec_items + 1)],
                                   index=np.arange(len(user_mtx)))
        self.result.to_csv('result/recommend_keras.csv')

    def build(self, dim, alpha, beta):
        users, items, _ = self.data.max(axis=0)

        user_input = tf.keras.layers.Input((1, ), name='user')
        user_vec = self.embedding(user_input, users, dim, beta, 'user_vec')
        item_input = tf.keras.layers.Input((1, ), name='item')
        item_vec = self.embedding(item_input, items, dim, beta, 'item_vec')
        outputs = tf.keras.layers.Dot(axes=1)([user_vec, item_vec])

        model = tf.keras.models.Model([user_input, item_input], outputs)
        adam = tf.keras.optimizers.Adam(alpha)
        model.compile(adam, 'mse')

        model.summary()
        self.model = model

    def _train(self, epoch, batch, verbose=2):
        self.model.fit([self.data[:, 0], self.data[:, 1]],
                       self.data[:, 2],
                       epochs=epoch,
                       verbose=verbose,
                       batch_size=batch,
                       shuffle=False)

    def embedding(self,
                  last_layer,
                  input_dim,
                  latent_dim,
                  beta,
                  name):
        input_length = 1
        regularizer = tf.keras.regularizers.l2(beta)
        initializer = tf.keras\
            .initializers\
            .RandomNormal()
        embedding = tf.keras.layers.Embedding(
                input_dim+1,
                latent_dim,
                input_length=input_length,
                embeddings_initializer=initializer,
                embeddings_regularizer=regularizer)(last_layer)
        return tf.keras.layers.Flatten(name=name)(embedding)

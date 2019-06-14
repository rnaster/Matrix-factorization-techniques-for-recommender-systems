import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


class KerasVanillaMF:
    def __init__(self, data: pd.DataFrame):
        """

        :param data: column = [user, item, rating]
        """
        self.data = data
        self.model = None
        self.result = None

    def train(self,
              latent_dim=2,
              learning_rate=1e-3,
              l2_rate=1e-4,
              epoch=300,
              batch_size=100,
              verbose=2,
              save=False):
        self.build(latent_dim, learning_rate, l2_rate)
        self._train(epoch,
                    batch_size,
                    verbose)
        self.plot_cost(save)
        if latent_dim == 2: self.plot_embedded_vector(save)
        self.predict()

    def predict(self):
        user, item = self.data['user'], self.data['item']
        self.data['predicted'] = self.model.predict([user, item])

    def recommend(self, num_rec_items):
        users = max(self.data['user'].unique())
        items = max(self.data['item'].unique())
        columns = np.arange(items+1)
        user_vectors, item_vectors = self.model.get_weights()
        predicted: np.ndarray = user_vectors @ item_vectors.T * -1
        predicted[self.data['user'], self.data['item']] *= 0
        result = predicted.argpartition(num_rec_items)[:, :num_rec_items]
        self.result = pd.DataFrame(columns[result.reshape(-1)].reshape(-1, num_rec_items),
                                   columns=['top%s' % i for i in range(1, num_rec_items + 1)],
                                   index=np.arange(users+1))
        self.result.to_csv('result.csv')

    def plot_embedded_vector(self, save=True):
        user_vectors, item_vectors = self.model.get_weights()
        user_vectors = user_vectors[self.data['user'].unique()]
        item_vectors = item_vectors[self.data['item'].unique()]
        plt.plot(*user_vectors.T,
                 marker='o',
                 ls='',
                 c='b',
                 markersize=1,
                 label='user')
        plt.plot(*item_vectors.T,
                 marker='o',
                 ls='',
                 c='r',
                 markersize=1,
                 label='item')
        plt.legend(loc='upper right',
                   bbox_to_anchor=(1, 1.1),
                   ncol=2,
                   handletextpad=0.01)
        plt.title('Latent Vector Space')
        plt.axhline(0, color='g', ls='-')
        plt.axvline(0, color='g', ls='-')
        if save: plt.savefig('latent_space_keras.jpg')
        plt.show()
        plt.close()

    def plot_cost(self, save=True):
        pd.DataFrame(self.model.history.history).plot(legend=False)
        plt.xlabel("Epoch")
        plt.ylabel("cost")
        if save: plt.savefig('plot_cost_keras.jpg')
        plt.show()
        plt.close()

    def build(self, latent_dim, learning_rate, l2_rate):
        users = max(self.data['user'].unique())
        items = max(self.data['item'].unique())

        user_input = tf.keras.layers.Input((1, ), name='user')
        user_vec = self.embedding(user_input, users, latent_dim, l2_rate)
        item_input = tf.keras.layers.Input((1, ), name='item')
        item_vec = self.embedding(item_input, items, latent_dim, l2_rate)
        outputs = tf.keras.layers.Dot(axes=1)([user_vec, item_vec])

        model = tf.keras.models.Model([user_input, item_input], outputs)
        adam = tf.keras.optimizers.Adam(learning_rate)
        model.compile(adam, 'mse')

        model.summary()
        self.model = model

    def _train(self, epoch, batch_size, verbose=2):
        user, item, rating \
            = self.data['user'], self.data['item'], self.data['rating']
        self.model.fit([user, item],
                       rating,
                       epochs=epoch,
                       verbose=verbose,
                       batch_size=batch_size,
                       shuffle=False)

    def embedding(self,
                  last_layer,
                  input_dim,
                  latent_dim,
                  l2_rate):
        input_length = 1
        regularizer = tf.keras.regularizers.l2(l2_rate)
        initializer = tf.keras\
            .initializers\
            .RandomNormal()
        embedding = tf.keras.layers.Embedding(
                input_dim+1,
                latent_dim,
                input_length=input_length,
                embeddings_initializer=initializer,
                embeddings_regularizer=regularizer)(last_layer)
        return tf.keras.layers.Flatten()(embedding)

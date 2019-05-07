# TODO: split train, test
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from src.pipeline import DataPipeline
from src.visual import Visualization


class VanillaMF(Visualization):
    def __init__(self, user_item_mtx: np.ndarray):
        """

        :param user_item_mtx: sparse matrix
        """
        super().__init__()
        self.user_item_matrix = user_item_mtx
        self.user_matrix = None
        self.item_matrix = None
        self.predict_matrix = None

    def init_user_matrix(self,
                         shape: tuple or list,
                         mean=0,
                         std=0.1):
        return np.random.normal(mean, std, shape)

    def init_item_matrix(self,
                         shape: tuple or list,
                         mean=0,
                         std=0.1):
        return np.random.normal(mean, std, shape)

    def fit(self,
            rank=2,
            alpha=1e-4,
            beta=1e-3,
            tolerance=1e-4,
            epoch=70_000,
            every_print=5000):
        """
        user, item matrix 학습
        q_i <- q_i + alpha * (e_ui * p_u - beta * q_i)
        p_u <- p_u + alpha * (e_ui * q_i - beta * p_u)

        학습이 더이상 진행되지 않으면(loss값의 변화가 없다면)
        tolerance를 기준으로 early-stopping

        :return: r_ui hat matrix(p_u * q_i)
        """
        users, items = self.user_item_matrix.shape
        user_matrix = self.init_user_matrix([users, rank])
        item_matrix = self.init_item_matrix([items, rank])
        mask_matrix = self.user_item_matrix > 0
        last_cost = np.Inf
        for i in range(1, epoch + 1):
            predicted_matrix = user_matrix @ item_matrix.T
            error_matrix = self.user_item_matrix - predicted_matrix * mask_matrix
            user_matrix, item_matrix \
                = user_matrix + alpha * (error_matrix @ item_matrix - beta * user_matrix), \
                  item_matrix + alpha * (error_matrix.T @ user_matrix - beta * item_matrix)
            if i % every_print == 0:
                cost = np.sum(error_matrix * error_matrix, axis=1).mean()
                cost = np.sqrt(cost)
                if abs(last_cost - cost) < tolerance:
                    print('early-stopping!')
                    break
                else:
                    last_cost = cost
                    self.cost = last_cost
                    print('%dth training, cost: %.6f' % (i, last_cost))
        self.user_matrix = user_matrix
        self.item_matrix = item_matrix
        self.predict_matrix = user_matrix @ item_matrix.T
        print('finish fitting')

    def predict(self, user, num_rec_items):
        """
        excluding the observed items,
         recommend num_rec_items items for specific user.

        :param user: index
        :param num_rec_items: the number of item to recommend
        :return: index
        """
        mask_matrix = self.user_item_matrix <= 0
        predicted_matrix = self.predict_matrix * mask_matrix * -1
        index = predicted_matrix[user].argsort()
        return index[:num_rec_items]


class KerasVanillaMF:
    def __init__(self,
                 data: pd.DataFrame,
                 seed=None):
        """

        :param data: narrow data, column = [user, item, rating]
        """
        self.data = data
        self.seed = seed
        self.model = None
        self.recommendation = pd.DataFrame(columns=['top1', 'top2', 'top3'])

    def run(self,
            latent_dim,
            l2_rate,
            epoch,
            batch_size,
            verbose=2,
            save=False):
        dataset = self.data_flow(batch_size)
        self.build(latent_dim, l2_rate)
        self.train(dataset.train_steps,
                   dataset.val_steps,
                   epoch,
                   verbose)
        self.evaluate_test(dataset.test_steps, dataset.pre_test_set)
        self.plot_cost(save)
        if latent_dim == 2: self.plot_embedded_vector(save)
        self.predict()

    def evaluate_test(self, test_steps, pre_test_set):
        _, _, test_set = tf.get_collection('dataset')[0]
        self.model.evaluate(test_set, steps=test_steps)
        self.data['predicted'] = 0
        predicted_test = self.model.predict(test_set, steps=test_steps).reshape(-1)
        self.data = self.data.set_index(['user', 'item'])
        for i in range(len(pre_test_set)):
            self.data.loc[(pre_test_set[i][0], pre_test_set[i][1]), 'predicted'] = predicted_test[i]

    def predict(self):
        self.data = self.data.reset_index()
        item_set = set(self.data['item'])
        for user in self.data['user'].unique():
            item2recommend = item_set - set(self.data[self.data['user'] == user]['item'])
            item2recommend = list(item2recommend)
            dataset = self.make_dataset(user, item2recommend)
            result = self.model.predict(dataset)
            idx = (-result).reshape(-1).argsort()[:3]
            self.recommendation.loc[user] = [item2recommend[i] for i in idx]

    def make_dataset(self, user, item_list):
        user = [user] * len(item_list)
        return user, item_list

    def plot_embedded_vector(self, save=False):
        user_vector, item_vector = self.model.get_weights()
        plt.plot(*user_vector.T,
                 marker='o',
                 ls='',
                 c='b',
                 markersize=1,
                 label='user')
        plt.plot(*item_vector.T,
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
        if save: plt.savefig('./book_crossing/latent_space.jpg')
        plt.show()
        plt.close()

    def plot_cost(self, save=False):
        pd.DataFrame(self.model.history.history).plot()
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        if save: plt.savefig('./book_crossing/plot_cost.jpg')
        plt.show()
        plt.close()

    def data_flow(self, batch_size):
        dataset = DataPipeline(self.data.values,
                               batch_size=batch_size,
                               seed=self.seed)
        train_set, val_set, test_set = dataset.dataset()
        tf.add_to_collection('dataset',
                             [train_set, val_set, test_set])
        return dataset

    def build(self, latent_dim, l2_rate=0):
        users = max(self.data.user.unique())
        items = max(self.data.item.unique())

        user_input = tf.keras.layers.Input(shape=(1,), name="user")
        user_vec = self.embedding(user_input, users, latent_dim, l2_rate)

        item_input = tf.keras.layers.Input(shape=(1,), name="item")
        item_vec = self.embedding(item_input, items, latent_dim, l2_rate)

        dot_product = tf.keras.layers.Dot(axes=1)([user_vec, item_vec])
        model = tf.keras.models.Model([user_input, item_input], dot_product)
        model.compile('adam', 'mse')

        model.summary()
        self.model = model

    def train(self, train_steps, val_steps, epoch, verbose=2):
        train_set, val_set, _ = tf.get_collection('dataset')[0]
        self.model.fit(train_set.make_one_shot_iterator(),
                       epochs=epoch,
                       validation_data=val_set.make_one_shot_iterator(),
                       validation_steps=val_steps,
                       steps_per_epoch=train_steps,
                       verbose=verbose,
                       shuffle=False)
        self.save_model('./book_crossing/weights')

    def save_model(self, path):
        pathlib.Path(path) \
            .mkdir(parents=True,
                   exist_ok=True)
        self.model.save(path + '/model')

    def embedding(self,
                  last_layer,
                  input_dim,
                  latent_dim,
                  l2_rate):
        input_length = 1
        if l2_rate > 0:
            regularizer = tf.keras.regularizers.l2(l2_rate)
        else:
            regularizer = None
        initializer = tf.keras\
            .initializers\
            .RandomUniform(seed=self.seed)
        embedding = tf.keras.layers.Embedding(
                input_dim+1,
                latent_dim,
                input_length=input_length,
                embeddings_initializer=initializer,
                embeddings_regularizer=regularizer)(last_layer)
        return tf.keras.layers.Flatten()(embedding)


if __name__ == '__main__':
    np.random.seed(1818)
    mtx = np.random.normal(0, 1, size=[20, 5])
    mtx = mtx * (mtx > 0)
    print(mtx, '\n')
    test = VanillaMF(mtx)
    test.fit()
    test.plot_cost()
    test.viz_latent_space()
# TODO: split train, test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Visualization:
    """
    draw training loss plot
    visualize latent user vector and item vector on 2D
    """
    def __init__(self):
        self.history = []

    def plot_cost(self, epoch, every_print, save=True):
        pd.DataFrame(self.history,
                     index=np.arange(0, epoch, step=every_print))\
            .plot(legend=False)
        plt.xlabel("epoch")
        plt.ylabel("cost")
        plt.title('Vanilla MF')
        if save: plt.savefig('plot_cost_numpy.jpg')
        plt.show()
        plt.close()

    def viz_latent_space(self,
                         user_matrix,
                         item_matrix,
                         save=False):
        plt.plot(*user_matrix.T,
                 marker='o',
                 ls='',
                 c='b',
                 markersize=1,
                 label='user')
        plt.plot(*item_matrix.T,
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
        if save: plt.savefig('latent_space_numpy.jpg')
        plt.show()
        plt.close()


class VanillaMF(Visualization):
    def __init__(self, data: pd.DataFrame):
        """
        :param data: columns = [user, item, rating]
        """
        super().__init__()
        self.data = data
        self.user_matrix = None
        self.item_matrix = None
        self.result = None

    def embedding(self,
                  latent_dim,
                  mean=0,
                  std=0.1):
        users, items = self.data \
            .pivot('user', 'item', 'rating') \
            .fillna(0) \
            .shape
        self.print_params(users, items, latent_dim)
        self.user_matrix = np.random.normal(mean, std, [users, latent_dim])
        self.item_matrix = np.random.normal(mean, std, [items, latent_dim])

    def train(self,
              latent_dim=2,
              learning_rate=1e-4,
              l2_rate=1e-3,
              epoch=4000,
              every_print=1000,
              verbose=2,
              save=True):
        """
        user, item matrix 학습
        q_i <- q_i + learning_rate * (e_ui * p_u - l2_rate * q_i)
        p_u <- p_u + learning_rate * (e_ui * q_i - l2_rate * p_u)
        """
        self.embedding(latent_dim)
        self._train(learning_rate,
                    l2_rate,
                    epoch,
                    every_print,
                    verbose)
        print('finish')
        self.plot_cost(epoch, every_print, save)
        if latent_dim == 2:
            self.viz_latent_space(self.user_matrix.values,
                                  self.item_matrix.values,
                                  save)
        self.predict()

    def _train(self,
               learning_rate,
               l2_rate,
               epoch,
               every_print,
               verbose):
        data = self.data\
            .pivot('user', 'item', 'rating')\
            .fillna(0)
        mask_matrix = data > 0

        for i in range(1, epoch + 1):
            predicted_matrix = self.user_matrix @ self.item_matrix.T
            error_matrix = data - predicted_matrix * mask_matrix
            self.user_matrix, self.item_matrix \
                = self.user_matrix + learning_rate * (error_matrix @ self.item_matrix
                                                      - l2_rate * self.user_matrix), \
                  self.item_matrix + learning_rate * (error_matrix.T @ self.user_matrix
                                                      - l2_rate * self.item_matrix)
            if i % every_print == 0:
                cost = np.sum(error_matrix * error_matrix, axis=1).mean()
                cost = np.sqrt(cost)
                self.history.append(cost)
                if verbose > 0:
                    print('%dth training, cost: %.6f' % (i, cost))

    def predict(self):
        data = self.data.values.T
        user_matrix = self.user_matrix.loc[data[0]].values
        item_matrix = self.item_matrix.loc[data[1]].values
        predicted = user_matrix * item_matrix
        self.data['predicted'] = predicted.sum(axis=1)
        self.data.to_csv('predicted_numpy.csv')

    def print_params(self, users, items, latent_dim):
        print('the number of trainable params: {:,}\n'
              .format(users * latent_dim + items * latent_dim))

    def recommend(self, num_rec_items):
        """
        do recommend num_rec_items items excluding the observed items.
        """
        mask_matrix = self.data.pivot('user', 'item', 'rating').fillna(0) == 0
        columns = np.sort(self.data['item'].unique())
        predicted: pd.DataFrame = self.user_matrix @ self.item_matrix.T * -1
        predicted *= mask_matrix
        result = predicted.values.argpartition(num_rec_items)[:, :num_rec_items]
        self.result = pd.DataFrame(columns[result.reshape(-1)].reshape(-1, num_rec_items),
                                   columns=['top%s' % i for i in range(1, num_rec_items+1)],
                                   index=np.sort(self.data['user'].unique()))
        self.result.to_csv('result_numpy.csv')

import numpy as np
import pandas as pd

from tqdm.auto import tqdm


class MF:
    def __init__(self, data_path):
        self.data = np.genfromtxt(data_path,
                                  delimiter='\t',
                                  skip_header=True,
                                  dtype=np.int32)
        self.user_mtx = None
        self.item_mtx = None
        self.result = None

    def embedding(self,
                  dim,
                  mean=0,
                  std=0.1):
        users, items, _ = self.data.max(axis=0)
        self.print_params(users, items, dim)
        self.user_mtx = np.random.normal(mean, std, [users + 1, dim])
        self.item_mtx = np.random.normal(mean, std, [items + 1, dim])

    def train(self,
              dim=2,
              alpha=1e-5,
              beta=1e-3,
              epoch=500,
              every_print=20,
              verbose=1):
        """
        user, item matrix 학습
        q_i <- q_i + learning_rate * (e_ui * p_u - l2_rate * q_i)
        p_u <- p_u + learning_rate * (e_ui * q_i - l2_rate * p_u)
        """
        self.embedding(dim)
        self._train(alpha,
                    beta,
                    epoch,
                    every_print,
                    verbose)
        print('finish')
        self.predict(self.data)

    def _train(self,
               alpha,
               beta,
               epoch,
               every_print,
               verbose):
        rating = self.data[:, 2]
        rating = rating.astype(np.float32)
        for ep in tqdm(range(1, epoch + 1)):
            for (user, item), rat in zip(self.data[:, :2], rating):
                error = rat - np.dot(self.user_mtx[user], self.item_mtx[item])
                self.user_mtx[user], self.item_mtx[item] \
                    = self.user_mtx[user] + alpha * (error * self.item_mtx[item]
                                                     - beta * self.user_mtx[user]), \
                      self.item_mtx[item] + alpha * (error * self.user_mtx[user]
                                                     - beta * self.item_mtx[item])

            if verbose and ep % every_print == 0:
                error = rating - np.sum(
                    self.user_mtx[self.data[:, 0]] * self.item_mtx[self.data[:, 1]],
                    axis=1
                )
                cost = np.sqrt(np.mean(error * error))
                print('%dth training, cost: %.6f' % (ep, cost))

    def predict(self, array: np.ndarray):
        """

        :param array: [user, item, rating]
        """
        user_mtx = self.user_mtx[array[:, 0]]
        item_mtx = self.item_mtx[array[:, 1]]
        predicted = np.sum(user_mtx * item_mtx, axis=1)
        pd.DataFrame({'actual': array[:, 2],
                      'predicted': predicted})\
            .to_csv('result/predict_np.csv', index=False)
        print('Finish predicting')

    def print_params(self, users, items, latent_dim):
        print('the number of trainable params: {:,}\n'
              .format(users * latent_dim + items * latent_dim))

    def recommend(self, num_rec_items):
        """
        do recommend num_rec_items items excluding the observed items.
        """
        predicted = np.inner(self.user_mtx, self.item_mtx) * -1
        predicted[self.data[:, 0], self.data[:, 1]] *= 0
        self.result = pd.DataFrame(predicted.argsort()[:, :num_rec_items],
                                   columns=['top%s' % i
                                            for i in range(1, num_rec_items + 1)],
                                   index=np.arange(len(self.user_mtx)))
        self.result.to_csv('result/recommend_np.csv')


if __name__ == '__main__':
    test = MF('../data/book_ratings.dat')
    test.train()
import numpy as np
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


if __name__ == '__main__':
    np.random.seed(1818)
    mtx = np.random.normal(0, 1, size=[20, 5])
    mtx = mtx * (mtx > 0)
    print(mtx, '\n')
    test = VanillaMF(mtx)
    test.fit()
    test.plot_cost()
    test.viz_latent_space()
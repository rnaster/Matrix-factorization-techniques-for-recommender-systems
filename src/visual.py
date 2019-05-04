import matplotlib.pyplot as plt


class Visualization:
    """
    draw training loss plot
    visualize latent user vector and item vector on 2D or 3D
    """
    def __init__(self):
        self.__cost = []
        self.user_matrix = None
        self.item_matrix = None

    @property
    def cost(self):
        return self.__cost

    @cost.setter
    def cost(self, val):
        self.__cost.append(val)

    def plot_cost(self, save=False):
        plt.plot(range(len(self.cost)), self.cost,
                 marker='o', ls='-', c='b')
        plt.xlabel("epoch")
        plt.xticks(range(len(self.cost)))
        plt.ylabel("cost")
        plt.title('training cost')
        if save: plt.savefig('./training_cost.jpg')
        plt.show()
        plt.close()

    def viz_latent_space(self, save=False):
        if self.user_matrix is None \
                or self.item_matrix is None: return
        plt.plot(*self.user_matrix.T,
                 marker='o',
                 ls='',
                 c='b',
                 markersize=1,
                 label='user')
        plt.plot(*self.item_matrix.T,
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
        if save: plt.savefig('./latent_space.jpg')
        plt.show()
        plt.close()


if __name__ == '__main__':
    test = Visualization()
    test.cost = 0.11
    test.cost = 0.123
    test.cost = 0.1657
    test.cost = 0.9874
    test.cost = 0.0034
    test.cost = 0.4523
    test.plot_cost()
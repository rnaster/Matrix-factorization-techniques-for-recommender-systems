import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataPipeline:
    def __init__(self,
                 data: np.ndarray,
                 val_rate=0.1,
                 test_rate=0.2,
                 batch_size=100,
                 seed=None):
        self.data = data
        self.pre_test_set = None
        self.val_rate = val_rate
        self.test_rate = test_rate
        self.batch_size = batch_size
        self.seed = seed
        self.train_steps = 0
        self.val_steps = 0
        self.test_steps = 0

    def set_steps(self, dataset: str, size):
        steps = size / self.batch_size
        steps = int(steps) + 1 if steps > int(steps) else int(steps)
        if dataset == 'train':
            self.train_steps = steps
        elif dataset == 'val':
            self.val_steps = steps
        elif dataset == 'test':
            self.test_steps = steps
        else:
            ValueError('incorrect name,'
                       ' choose one of ["train", "val", "test"]')

    def split_dataset(self):
        train_set, test_set = train_test_split(
            self.data,
            test_size=self.test_rate,
            shuffle=True,
            random_state=self.seed)
        self.pre_test_set = test_set
        train_set, val_set = train_test_split(
            train_set,
            test_size=self.val_rate,
            shuffle=True,
            random_state=self.seed)
        return train_set, val_set, test_set

    def dataset(self):
        train_set, val_set, test_set = self.split_dataset()
        self.set_steps('train', len(train_set))
        self.set_steps('val', len(val_set))
        self.set_steps('test', len(test_set))
        train_set = self._dataset(train_set, shuffle=True)
        val_set = self._dataset(val_set)
        test_set = self._dataset(test_set, repeat=False)
        return train_set, val_set, test_set

    def _dataset(self, data, repeat=True, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices(data).map(self._reshape, -1)
        if shuffle: dataset = dataset.shuffle(len(data),
                                              seed=self.seed)
        if repeat: dataset = dataset.repeat()
        return dataset.batch(self.batch_size)

    def _reshape(self, data):
        return (data[0], data[1]), data[2]


if __name__ == '__main__':
    # np.random.seed(1818)
    data = np.arange(33*3).reshape(-1, 3)
    test = DataPipeline(data, batch_size=2, seed=1818)
    train_set, _, _ = test.dataset()
    print('train_steps: %d' % test.train_steps)
    iterator = train_set.make_one_shot_iterator()
    batch = iterator.get_next()
    with tf.Session() as sess:
        for _ in range(12):
            print(sess.run(batch))
        print()
        iterator = train_set.make_one_shot_iterator()
        batch = iterator.get_next()
        for _ in range(12):
            print(sess.run(batch))
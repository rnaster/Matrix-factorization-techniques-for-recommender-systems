import os
import random
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd

from model.mf import VanillaMF
from model.mf_with_keras import KerasVanillaMF


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='data directory')
    parser.add_argument('--mode', default='gpu', help='execution on cpu/gpu', choices=['cpu', 'gpu'])
    parser.add_argument('--epoch', default=300, type=int, help='training epoch')
    parser.add_argument('--l2_rate', default=1e-5, type=float, help='L2 regularization rate')
    parser.add_argument('--batch', default=32, type=int, help='training batch size')
    parser.add_argument('--verbose', default=2, type=int, help='equivalent to keras verbose', choices=[0, 1, 2])
    parser.add_argument('--latent_dim', default=2, type=int, help='choose latent dimension')
    parser.add_argument('--num_rec_items', default=3, type=int, help='the number of items to recommend')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    return parser.parse_args()


def main():
    os.environ['PYTHONHASHSEED'] = '0'
    seed = 1818
    np.random.seed(seed)
    random.seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(),
                      config=session_conf)
    tf.keras.backend.set_session(sess)

    args = arg_parser()
    mode = args.mode
    data_dir = args.data_dir
    epoch = args.epoch
    l2_rate = args.l2_rate
    batch = args.batch
    verbose = args.verbose
    latent_dim = args.latent_dim
    num_rec_items = args.num_rec_items
    learning_rate = args.learning_rate

    data = pd.read_csv(data_dir, sep='\t')

    if mode == 'gpu':
        model = KerasVanillaMF(data)
        model.train(latent_dim, learning_rate, l2_rate, epoch, batch, verbose)
    else:
        data['rating'] *= 0.1  # convert rating range 1 ~ 10 to 0.1 ~ 1.0
        model = VanillaMF(data)
        model.train(latent_dim, learning_rate, l2_rate, epoch, verbose=verbose)
    model.recommend(num_rec_items)
    print(model.data.head())
    print(model.result.head())


if __name__ == '__main__':
    main()

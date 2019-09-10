import json

from model.mf import MF
from model.keras import MF as KerasMF


def main():
    with open('config.json', 'r') as f:
        args = json.load(f)

    mode = args['mode']
    data_path = args['data_path']
    dim = args['dim']
    alpha = args['alpha']
    beta = args['beta']
    epoch = args['epoch']
    num_rec_items = args['num_rec_items']
    verbose = args['verbose']

    if mode == 'gpu':
        mf = KerasMF(data_path)
    else:
        mf = MF(data_path)
    mf.train(dim, alpha, beta, epoch, verbose=verbose)
    mf.recommend(num_rec_items)


if __name__ == '__main__':
    main()
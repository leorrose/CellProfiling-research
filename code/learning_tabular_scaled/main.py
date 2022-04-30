from configuration.config import args

from data_layer.prepare_data import load_data

from time import time

s = time()
dataloaders = load_data(args)
e = time()
print(f'Done loading data in {e-s} seconds')

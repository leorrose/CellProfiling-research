from configuration.config import args

from data_layer.prepare_data import load_data

dataloaders = load_data(args)

print('Done loading data')

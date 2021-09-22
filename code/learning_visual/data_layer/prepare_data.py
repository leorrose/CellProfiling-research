import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data_layer import transforms, dataset
from data_layer.dataset import CovidDataset

# Maximum value of a pixel
VMAX = 65535

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def load_data(args):
    """

    :param args:
        metadata_path: path to image filenames
        plates_split: dict containing:
            train: plates numbers used for training
            test: plates numbers used for test
        split_ratio (float in [0,1]): train-val split param
        target_channel (int): channel to predict

    :return:
    """

    plates_meta_paths = [os.path.join(args.metadata_path, f'{p}.csv') for plates in args.plates_split for p in plates]
    dfs = [pd.read_csv(m) for m in plates_meta_paths]
    mt_df = pd.concat(dfs, ignore_index=True)

    partitions = split_by_plates(mt_df, args.plates_split[0], args.plates_split[1],
                                 args.test_samples_per_plate)
    partitions['train'], partitions['val'] = train_test_split(np.asarray(partitions['train']),
                                                              train_size=args.split_ratio,
                                                              shuffle=True)

    partitions = partitions_idx_to_dfs(mt_df, partitions)

    datasets = create_datasets(args.plates_split, partitions, args.images_path, args.target_channel,
                               args.input_size, args.device, args.num_input_channels)
    print_data_statistics(datasets)
    data_loaders = create_data_loaders(datasets, partitions, args.batch_size, args.num_data_workers)

    return data_loaders


def split_by_plates(df, train_plates, test_plates=None, test_samples_per_plate=None) -> dict:
    partitions = {
        'train': list(df[(df['Plate'].isin(train_plates)) & (df['Well_Role'] == 'mock')].index),
        'test': {}
    }

    if test_plates is None:
        test_plates = train_plates

    # divide test data into plates (mock, irradiated and active from test plates)
    for plate in test_plates:
        partitions['test'][str(plate)] = {}
        partitions['test'][str(plate)]['mock'] = list(
            df[(df['Plate'] == plate) & (df['Well_Role'] == 'mock')].index)[:test_samples_per_plate]
        partitions['test'][str(plate)]['treated'] = list(
            df[(df['Plate'] == plate) & (df['Well_Role'] == 'treated')].index)[:test_samples_per_plate]

    return partitions


def partitions_idx_to_dfs(mt_df, partitions):
    df_partitions = {
        'train': mt_df.iloc[partitions['train']],
        'val': mt_df.iloc[partitions['val']],
        'test': {}
    }

    for plate in list(partitions['test'].keys()):
        df_partitions['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            df_partitions['test'][plate][key] = mt_df.iloc[partitions['test'][plate][key]]

    return df_partitions


def create_datasets(plates_split, partitions, data_dir, target_channel, input_size, device,
                    num_input_channels):
    train_plates, test_plates = plates_split
    mean, std = get_data_stats(partitions['train'], train_plates, data_dir, target_channel, device)

    train_transforms = transforms.Compose([
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    datasets = {
        'train': CovidDataset(partitions['train'], target_channel, root_dir=data_dir, transform=train_transforms,
                              input_channels=num_input_channels),
        'val': CovidDataset(partitions['val'], target_channel, root_dir=data_dir, transform=train_transforms,
                            input_channels=num_input_channels),
        'val_for_test': CovidDataset(partitions['val'], target_channel, root_dir=data_dir,
                                     transform=test_transforms, is_test=True, input_channels=num_input_channels),
        'test': {}
    }

    for plate in list(partitions['test'].keys()):
        datasets['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            datasets['test'][plate][key] = \
                CovidDataset(partitions['test'][plate][key], target_channel, root_dir=data_dir,
                             transform=test_transforms, is_test=True, input_channels=num_input_channels)

    return datasets


def print_data_statistics(datasets):
    print('train set contains ' + str(len(datasets['train'])) + ' images')
    print('val set contains ' + str(len(datasets['val'])) + ' images')

    for plate in list(datasets['test'].keys()):
        for key in datasets['test'][plate].keys():
            print(' test set from plate ' + plate + ' of ' + key + ' contains ' + str(
                len(datasets['test'][plate][key])) + ' images')


def get_data_stats(train_mt_df, train_plates, data_dir, target_channel, device):
    # TODO: More appropriate way to disable recalculating
    # TODO: Replace with actual numbers from more plates
    train_plates = []
    if not train_plates:
        if target_channel.name == 'AGP':
            mean = [57.562225341796875, 32.24236297607422, 54.539520263671875, 57.95323944091797, 54.08218002319336]
            std = [102.41613006591797, 104.47589111328125, 104.00897979736328, 98.84741973876953, 97.49189758300781]
    else:
        logging.info('calculating mean and std...')
        mean, std = calc_mean_and_std(train_mt_df, data_dir, len(train_plates), device)

    return mean, std


def calc_mean_and_std(mt_df, data_dir, num_batches, device):
    train_data = dataset.CovidDataset(mt_df, root_dir=data_dir, target_channel=None,
                                      for_data_statistics_calc=True)
    batch_size = int(len(train_data) / num_batches)
    # TODO: Why this size?
    batch_size = 512
    train_loader = DataLoader(train_data, batch_size=batch_size)
    num_channels = len(dataset.Channels)

    mean = torch.zeros(num_channels).to(device)
    std = torch.zeros(num_channels).to(device)

    for images in train_loader:
        images = images.to(device)
        # TODO: Divide by maximum value was removed
        batch_mean, batch_std = torch.std_mean(images.float(), dim=(0, 1, 2))

        mean += batch_mean
        std += batch_std

    mean /= num_batches
    std /= num_batches
    print('mean of train data is ' + str(mean.tolist()))
    print('std of train data is ' + str(std.tolist()))

    return mean.tolist(), std.tolist()


def create_data_loaders(datasets, partitions, batch_size, num_workers=32) -> dict:
    data_loaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size,
                            shuffle=True, num_workers=num_workers),
        'val': DataLoader(datasets['val'], batch_size=batch_size,
                          shuffle=False, num_workers=num_workers),
        'val_for_test': DataLoader(datasets['val_for_test'], batch_size=1,
                                   shuffle=False, num_workers=num_workers),
        'test': {}
    }

    for plate in list(partitions['test'].keys()):
        data_loaders['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            data_loaders['test'][plate][key] = \
                DataLoader(datasets['test'][plate][key], batch_size=1,
                           shuffle=False, num_workers=num_workers)

    return data_loaders

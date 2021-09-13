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

    datasets = create_datasets(mt_df, args.plates_split, partitions, args.images_path, args.target_channel,
                               args.input_size, args.device, args.num_input_channels)
    print_data_statistics(datasets)
    dataloaders = create_dataloaders(datasets, partitions, args.batch_size)

    return dataloaders


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


def create_datasets(mt_df, plates_split, partitions, data_dir, target_channel, input_size, device,
                    num_input_channels):
    train_plates, test_plates = plates_split
    # mean, std = [0.5,0.4,0.3,0.2], [0.1,0.1,0.1,0.1]
    # mean, std = calc_mean_and_std(partitions['train'])

    # Y_mean, Y_std = mean[target_channel], std[target_channel]
    # X_mean, X_std = mean.remove(target_channel), std.remove(target_channel)
    train_plates = []  # TODO: More appropriate way to disable recalculating
    mean, std = get_data_stats(mt_df, partitions['train'], train_plates, data_dir, device)

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
        'train': CovidDataset(mt_df, partitions['train'], target_channel, root_dir=data_dir, transform=train_transforms,
                              input_channels=num_input_channels),
        'val': CovidDataset(mt_df, partitions['val'], target_channel, root_dir=data_dir, transform=train_transforms,
                            input_channels=num_input_channels),
        'val_for_test': CovidDataset(mt_df, partitions['val'], target_channel, root_dir=data_dir,
                                     transform=test_transforms, is_test=True, input_channels=num_input_channels),
        'test': {}
    }

    for plate in list(partitions['test'].keys()):
        datasets['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            datasets['test'][plate][key] = \
                CovidDataset(mt_df, partitions['test'][plate][key], target_channel, root_dir=data_dir,
                             transform=test_transforms, is_test=True, input_channels=num_input_channels)

    return datasets


def print_data_statistics(datasets):
    print('train set contains ' + str(len(datasets['train'])) + ' images')
    print('val set contains ' + str(len(datasets['val'])) + ' images')

    for plate in list(datasets['test'].keys()):
        for key in datasets['test'][plate].keys():
            print(' test set from plate ' + plate + ' of ' + key + ' contains ' + str(
                len(datasets['test'][plate][key])) + ' images')


def get_data_stats(mt_df, train_inds, train_plates, data_dir, device):
    if not train_plates:  # TODO: Replace with actual numbers from more plates
        mean = [0.011394727043807507, 0.00655326247215271, 0.011172938160598278, 0.011629479937255383,
                0.01122655812650919]
        std = [0.020888447761535645, 0.022583933547139168, 0.021113308146595955, 0.021329505369067192,
               0.020590465515851974]
    else:
        logging.info('calculating mean and std...')
        mean, std = calc_mean_and_std(mt_df, train_inds, data_dir, len(train_plates), device)

    return mean, std


def calc_mean_and_std(mt_df, inds, data_dir, num_batches, device):
    # calculate_mean

    train_data = dataset.CovidDataset(mt_df, inds, root_dir=data_dir, target_channel=None,
                                      for_data_statistics_calc=True)
    batch_size = int(len(train_data) / num_batches)
    # TODO: Why this size?
    batch_size = 512
    train_loader = DataLoader(train_data, batch_size=batch_size)
    num_channels = len(dataset.DEFAULT_CHANNELS)

    mean = torch.zeros(num_channels).to(device)
    std = torch.zeros(num_channels).to(device)

    for images in train_loader:
        images = images.to(device)
        batch_mean, batch_std = torch.std_mean(images.float().div(VMAX), dim=(0, 1, 2))

        mean += batch_mean
        std += batch_std

    mean /= num_batches
    std /= num_batches
    print('mean of train data is ' + str(mean.tolist()))
    print('std of train data is ' + str(std.tolist()))

    return mean.tolist(), std.tolist()


def create_dataloaders(datasets, partitions, batch_size, num_workers=32) -> dict:
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size,
                            shuffle=True, num_workers=num_workers),
        'val': DataLoader(datasets['val'], batch_size=batch_size,
                          shuffle=False, num_workers=num_workers),
        'val_for_test': DataLoader(datasets['val_for_test'], batch_size=1,
                                   shuffle=False, num_workers=num_workers),
        'test': {}
    }

    for plate in list(partitions['test'].keys()):
        dataloaders['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            dataloaders['test'][plate][key] = \
                DataLoader(datasets['test'][plate][key], batch_size=1,
                           shuffle=False, num_workers=num_workers)

    return dataloaders

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from data_layer.tabular_dataset import TabularDataset

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def load_data(args):
    """

    :param args:
        metadata_path: path to tabular metadata csv
        plates_path: path to folder of plates' csvs
        plates_split: dict containing:
            train: plates numbers used for training
            test: plates numbers used for test
        split_ratio (float in [0,1]): train-val split param
        target_channel (int): channel to predict

    :return:
    """

    all_plates = sum(args.plates_split, [])
    if args.metadata_path is None:
        mt_df = create_tabular_metadata(args.plates_path, all_plates, args.label_field)
    else:
        mt_df = pd.read_csv(args.metadata_path, dtype={'Plate': int, 'Count': int})
        mt_df['Indexes'] = mt_df['Indexes'].apply(lambda x: json.loads(x))
        plates = [p for p in all_plates if p not in mt_df['Plate'].unique()]
        if plates:
            add_df = create_tabular_metadata(args.plates_path, plates, args.label_field)
            mt_df = pd.concat([mt_df, add_df], ignore_index=True)
            mt_df.to_csv(args.metadata_path, index=False)

    partitions = split_by_plates(mt_df, args)

    partitions = partitions_idx_to_dfs(mt_df, partitions)

    datasets = create_datasets(args.plates_split, partitions, args.plates_path,
                               args.input_fields, args.target_fields,
                               args.device, args.norm_params_path)
    print_data_statistics(datasets)
    data_loaders = create_data_loaders(datasets, partitions, args.batch_size, args.num_data_workers)

    return data_loaders


def create_tabular_metadata(plates_path, plates, label_field):
    mt_dict = {'Plate': [], label_field: [], 'Indexes': [], 'Count': []}
    for plate in plates:
        plate_path = os.path.join(plates_path, f'{plate}.csv')
        df = pd.read_csv(plate_path)
        for (p, wr), c in df.groupby(['Plate', label_field]).count().iloc[:, 0].iteritems():
            mt_dict['Plate'].append(p)
            mt_dict[label_field].append(wr)
            mt_dict['Indexes'].append(list(df[df[label_field] == wr].index))
            mt_dict['Count'].append(c)
    return pd.DataFrame(mt_dict)


def split_by_plates(df, args) -> dict:
    train_plates, test_plates = args.plates_split
    train_plates, val_plates = train_test_split(train_plates, train_size=args.split_ratio, shuffle=True)

    # logging.info(f'Train Plates: {" ".join(str(t) for t in train_plates)}')
    # logging.info(f'Validation Plates: {" ".join(str(t) for t in val_plates)}')
    # logging.info(f'Test Plates: {" ".join(str(t) for t in test_plates)}' if test_plates else 'There are no test plates')
    print(f'Train Plates: {" ".join(str(t) for t in train_plates)}')
    print(f'Validation Plates: {" ".join(str(t) for t in val_plates)}')
    print(f'Test Plates: {" ".join(str(t) for t in test_plates)}' if test_plates else 'There are no test plates')

    partitions = {
        'train': list(df[(df['Plate'].isin(train_plates)) & (df[args.label_field].isin(args.train_labels))].index),
        'val': list(df[(df['Plate'].isin(val_plates)) & (df[args.label_field].isin(args.train_labels))].index),
        'test': {}
    }

    if test_plates is None:
        test_plates = train_plates

    # divide test data into plates (mock, irradiated and active from test plates)
    for plate in test_plates:
        partitions['test'][str(plate)] = {}
        for lbl in args.labels:
            partitions['test'][str(plate)][lbl] = list(
                df[(df['Plate'] == plate) & (df[args.label_field] == lbl)].index)[:args.test_samples_per_plate]

    return partitions


def partitions_idx_to_dfs(mt_df, partitions):
    df_partitions = {
        'train': mt_df.iloc[partitions['train']].copy(),
        'val': mt_df.iloc[partitions['val']].copy(),
        'test': {}
    }

    for plate in list(partitions['test'].keys()):
        df_partitions['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            df_partitions['test'][plate][key] = mt_df.iloc[partitions['test'][plate][key]].copy()

    return df_partitions


def create_datasets(plates_split, partitions, data_dir,
                    input_fields, target_fields,
                    device, norm_params_path):
    train_plates, test_plates = plates_split
    mean, std = get_data_stats(partitions['train'], train_plates, data_dir, device,
                               input_fields, target_fields, norm_params_path)

    train_transforms = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transforms = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    datasets = {
        'train': TabularDataset(partitions['train'], root_dir=data_dir, transform=train_transforms,
                                input_fields=input_fields, target_fields=target_fields),
        'val': TabularDataset(partitions['val'], root_dir=data_dir, transform=train_transforms,
                              input_fields=input_fields, target_fields=target_fields),
        'val_for_test': TabularDataset(partitions['val'], root_dir=data_dir, transform=test_transforms,
                                       input_fields=input_fields, target_fields=target_fields,
                                       is_test=True),
        'test': {}
    }

    for plate in list(partitions['test'].keys()):
        datasets['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            datasets['test'][plate][key] = \
                TabularDataset(partitions['test'][plate][key], root_dir=data_dir, transform=test_transforms,
                               input_fields=input_fields, target_fields=target_fields,
                               is_test=True)

    return datasets


def print_data_statistics(datasets):
    print('train set contains ' + str(len(datasets['train'])) + ' cells')
    print('val set contains ' + str(len(datasets['val'])) + ' cells')

    for plate in list(datasets['test'].keys()):
        for key in datasets['test'][plate].keys():
            print(' test set from plate ' + plate + ' of ' + key + ' contains ' + str(
                len(datasets['test'][plate][key])) + ' cells')


def get_data_stats(train_mt_df, train_plates, data_dir, device, input_fields, target_fields, norm_params_path):
    # TODO: Replace with actual numbers from more plates
    if os.path.exists(norm_params_path):
        mean, std = joblib.load(norm_params_path)
    else:
        logging.info('calculating mean and std...')
        mean, std = calc_mean_and_std(train_mt_df, data_dir, len(train_plates), device, input_fields, target_fields)
        joblib.dump((mean, std), norm_params_path)

    return mean, std


def calc_mean_and_std(mt_df, data_dir, num_batches, device, input_fields, target_fields):
    train_data = TabularDataset(mt_df, root_dir=data_dir,
                                input_fields=input_fields, target_fields=target_fields,
                                for_data_statistics_calc=True)
    batch_size = 256 #int(len(train_data) / num_batches)
    num_batches = 0
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    num_channels = len(input_fields) + len(target_fields)

    mean = torch.zeros(num_channels).to(device)
    std = torch.zeros(num_channels).to(device)
    max_p = 0
    min_p = 65535

    # from datetime import datetime
    # print("Before Time =", datetime.now().strftime("%H:%M:%S"))
    # itd = iter(train_loader)
    # print("Iter Time =", datetime.now().strftime("%H:%M:%S"))
    # batch = next(itd)
    # print("After Time =", datetime.now().strftime("%H:%M:%S"))

    for samples in train_loader:
        samples = samples.to(device)
        batch_mean, batch_std = torch.std_mean(samples.float(), dim=(0,))
        max_p = max(torch.max(samples.float()), max_p)
        min_p = min(torch.min(samples.float()), min_p)

        mean += batch_mean
        std += batch_std
        num_batches += 1

    mean /= num_batches
    std /= num_batches
    print('mean of train data is ' + str(mean.tolist()))
    print('std of train data is ' + str(std.tolist()))
    print('maximum of train data is ' + str(max_p.tolist()))
    print('minimum of train data is ' + str(min_p.tolist()))

    return mean.tolist(), std.tolist()


def create_data_loaders(datasets, partitions, batch_size, num_workers=32) -> dict:
    data_loaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size,
                            shuffle=True, num_workers=num_workers),
        'val': DataLoader(datasets['val'], batch_size=batch_size,
                          shuffle=False, num_workers=num_workers),
        'val_for_test': DataLoader(datasets['val_for_test'], batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers),
        'test': {}
    }

    for plate in list(partitions['test'].keys()):
        data_loaders['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            data_loaders['test'][plate][key] = \
                DataLoader(datasets['test'][plate][key], batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)

    return data_loaders

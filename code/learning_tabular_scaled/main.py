import logging
import os
import pickle
import sys
from time import time

import pandas as pd
import numpy as np

from configuration.config import parse_args
from configuration.model_config import Model_Config
from data_layer.prepare_data import load_data
from model_layer.TabularAE import unify_test_function

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch


def main(model, args, kwargs={}):
    print_exp_description(model, args, kwargs)

    logging.info('Preparing data...')
    s = time()
    dataloaders = load_data(args)
    e = time()
    logging.info(f'Done loading data in {e - s} seconds')

    model = model.model_class(**model.params, batch_size=args.batch_size)
    if args.mode == 'predict' and args.checkpoint is not None:
        logging.info('loading model from file...')
        model = model.load_from_checkpoint(args.checkpoint)
        model.to(args.device)
        logging.info('loading model from file finished')

        logging.info('testing model...')
        model.eval()

        res = test_by_partition(model, dataloaders['test'], args.exp_dir)
        logging.info('testing model finished...')

        save_results(res, args, kwargs)

    else:
        logging.info('training model...')
        model.to(args.device)
        logger = TensorBoardLogger(args.exp_dir,
                                   name='log_dir')
        trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=1, logger=logger,
                             gpus=int(torch.cuda.is_available()),
                             auto_scale_batch_size='binsearch', weights_summary='full')
        trainer.fit(model, dataloaders['train'], dataloaders['val_for_test'])
        logging.info('training model finished.')


def print_exp_description(Model, args, kwargs):
    description = 'Training Model ' + Model.name + ' with target ' + '-'.join(args.target_channels)
    for arg in kwargs:
        description += ', ' + arg + ': ' + str(kwargs[arg])
    print(description)

    print('Arguments:')
    col = 3
    i = 0
    for k, v in args.__dict__.items():
        print(f'\t{k}: {v}', end='')
        i = (i + 1) % col
        if not i:
            print()
    print()


def test_by_partition(model, test_dataloaders, exp_dir):
    result_path = os.path.join(exp_dir, 'results')
    os.makedirs(result_path, exist_ok=True)

    results = []
    for plate, plate_data in test_dataloaders.items():
        res_plate_path = os.path.join(result_path, f'{plate}.csv')
        if not os.path.exists(res_plate_path):
            plate_results = []
            for _, dataloader in test_dataloaders[plate].items():
                plate_res = test(model, dataloader)
                plate_results.append(plate_res)

            plate_res = pd.concat(plate_results)
            plate_res.to_csv(res_plate_path, index=False)
            del plate_res
            del plate_results
        # else:
        #     plate_res = pd.read_csv(res_plate_path)

        # results.append(plate_res)

    # res = pd.concat(results)
    return pd.DataFrame()


def test(model, data_loader):
    pred, mse, pcc = zip(*[unify_test_function(model, batch, mse_reduction='none') for batch in data_loader])
    mses = [i.cpu().numpy() for i in mse]
    mses = np.concatenate(mses)
    index = data_loader.dataset.get_index()
    rec = np.concatenate([index.to_numpy(), mses], axis=1)

    results = pd.DataFrame(rec, columns=data_loader.dataset.index_fields + data_loader.dataset.target_fields)
    return results


def save_results(res, args, kwargs={}):
    for arg in kwargs:
        res[arg] = kwargs[arg]

    res_dir = os.path.join(args.exp_dir, 'results.csv')
    if os.path.isfile(res_dir):
        prev_res = pd.read_csv(res_dir)
        res = pd.concat([prev_res, res])

    # save_to_pickle(res, os.path.join(args.exp_dir, 'results.pkl'))
    save_to_pickle(args, os.path.join(args.exp_dir, 'args.pkl'))
    res.to_csv(res_dir, index=False)


def save_to_pickle(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


if __name__ == '__main__':
    inp = int(sys.argv[1])
    lsd = 8
    channel_id = inp % 10
    plate_split_id = 0

    exp_num = (plate_split_id + 3) * 10000
    DEBUG = False

    exps = [(lr, batch_size)
            for batch_size in [512, 1024, 2048, 4096, 8196]
            for lr in [1.5e-4, 1.0e-4, 1.5e-3, 1.0e-3, 1.5e-2, 1.0e-2]
            ]
    exp_values = exps[27 - 1]

    lr, batch_size = exp_values
    exp_dict = {'lr': lr,
                'epochs': 20, 'latent_space_dim': lsd}

    channels_to_predict = [channel_id]

    model = Model_Config.TAE

    for target_channel in channels_to_predict:
        # torch.cuda.empty_cache()
        args = parse_args(channel_idx=target_channel, exp_num=exp_num)
        args.batch_size = batch_size
        args.epochs = exp_dict['epochs']

        exp_dict['input_size'] = len(args.input_fields)
        exp_dict['target_size'] = len(args.target_fields)
        model.update_custom_params(exp_dict)
        
        # args.mode = 'predict'

    
        all_plates = [
            'SQ00015211', 'SQ00015216', 'SQ00015229', 'SQ00015220',
            'SQ00015218'
        ]

        all_split = [all_plates, all_plates.copy()]

        args.plates_split = all_split
        args.split_ratio = 0.8
        args.test_samples_per_plate = None

        if DEBUG:
            args.test_samples_per_plate = 10
            args.epochs = 5

        main(model, args)

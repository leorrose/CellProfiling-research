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
            'SQ00015218', 'SQ00015041', 'SQ00015046', 'SQ00015048',
            'SQ00015219', 'SQ00015221', 'SQ00015217', 'SQ00015210',
            'SQ00015049', 'SQ00015047', 'SQ00015132', 'SQ00014817',
            'SQ00015135', 'SQ00014819', 'SQ00015103', 'SQ00015150',
            'SQ00015168', 'SQ00015157', 'SQ00015195', 'SQ00015159',
            'SQ00015166', 'SQ00015105', 'SQ00014820', 'SQ00015102',
            'SQ00014818', 'SQ00015134', 'SQ00015133', 'SQ00014816',
            'SQ00015158', 'SQ00015167', 'SQ00015160', 'SQ00015194',
            'SQ00015169', 'SQ00015156', 'SQ00015151', 'SQ00015173',
            'SQ00015142', 'SQ00015145', 'SQ00015111', 'SQ00015129',
            'SQ00015116', 'SQ00015120', 'SQ00015118', 'SQ00015127',
            'SQ00015144', 'SQ00015143', 'SQ00015172', 'SQ00015119',
            'SQ00015126', 'SQ00015121', 'SQ00015128', 'SQ00015117',
            'SQ00015110', 'SQ00015096', 'SQ00015053', 'SQ00015098',
            'SQ00015054', 'SQ00015232', 'SQ00015203', 'SQ00015204',
            'SQ00015055', 'SQ00015052', 'SQ00015099', 'SQ00015097',
            'SQ00015205', 'SQ00015202', 'SQ00015233', 'SQ00015045',
            'SQ00015042', 'SQ00015215', 'SQ00015212', 'SQ00015224',
            'SQ00015223', 'SQ00015043', 'SQ00015044', 'SQ00015222',
            'SQ00015214', 'SQ00015154', 'SQ00015198', 'SQ00015153',
            'SQ00015165', 'SQ00015196', 'SQ00015162', 'SQ00015136',
            'SQ00014813', 'SQ00015109', 'SQ00015131', 'SQ00014814',
            'SQ00015107', 'SQ00015138', 'SQ00015100', 'SQ00015163',
            'SQ00015197', 'SQ00015164', 'SQ00015199', 'SQ00015152',
            'SQ00015155', 'SQ00015101', 'SQ00015106', 'SQ00015139',
            'SQ00015130', 'SQ00014815', 'SQ00015137', 'SQ00015108',
            'SQ00014812', 'SQ00015112', 'SQ00015124', 'SQ00015123',
            'SQ00015148', 'SQ00015170', 'SQ00015146', 'SQ00015141',
            'SQ00015122', 'SQ00015125', 'SQ00015140', 'SQ00015147',
            'SQ00015171', 'SQ00015149', 'SQ00015209', 'SQ00015231',
            'SQ00015207', 'SQ00015200', 'SQ00015059', 'SQ00015057',
            'SQ00015050', 'SQ00015201', 'SQ00015206', 'SQ00015230',
            'SQ00015208', 'SQ00015051', 'SQ00015056', 'SQ00015058'
        ]

        all_split = [all_plates, all_plates.copy()]

        args.plates_split = all_split
        args.split_ratio = 0.8
        args.test_samples_per_plate = None

        if DEBUG:
            args.test_samples_per_plate = 10
            args.epochs = 5

        main(model, args)

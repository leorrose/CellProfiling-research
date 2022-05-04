import logging
import sys
from time import time

from configuration.config import parse_args
from configuration.model_config import Model_Config
from data_layer.prepare_data import load_data

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

        # TODO: TEST EVALUATION

        logging.info('testing model finished...')

        # TODO: SAVE RESULTS
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


if __name__ == '__main__':
    inp = int(sys.argv[1])
    # model_id = inp // 100
    channel_id = (inp // 100) - 1
    lsd = 8  # 2 ** (inp % 10)
    plate_id = (inp % 100) - 1

    exp_num = inp  # if None, new experiment directory is created with the next available number
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

        args.mode = 'train'
        # args.mode = 'predict'

        plates = [24792, 25912, 24509, 24633, 25987, 25680, 25422,
                  24517, 25664, 25575, 26674, 25945, 24687, 24752,
                  24311, 26622, 26641, 24594, 25676, 24774, 26562,
                  25997, 26640, 24562, 25938, 25708, 24321, 24735,
                  26786, 25571, 26666, 24294, 24640, 25985, 24661]
        # 24278

        args.plates_split = [
            [p for p in plates if p != plates[plate_id]],
            [plates[plate_id]]
        ]
        args.split_ratio = 0.8

        args.test_samples_per_plate = None

        if DEBUG:
            args.test_samples_per_plate = 10
            args.epochs = 5

        main(model, args)

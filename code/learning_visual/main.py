import logging
import os
import sys

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from configuration import config
from configuration.model_config import Model_Config
from data_layer.channels import Channels
from data_layer.prepare_data import load_data
from model_layer.UNET import unify_test_function
from util.files_operations import save_to_pickle, is_file_exist
from visuals.visualize import show_input_and_target


def test_by_partition(model, test_dataloaders, input_size, input_channels, exp_dir=None):
    result_path = os.path.join(exp_dir, 'results')
    os.makedirs(result_path, exist_ok=True)

    results = []
    for plate in list(test_dataloaders):
        res_plate_path = os.path.join(result_path, f'{plate}.csv')
        if not os.path.exists(res_plate_path):
            plate_results = []
            for key in test_dataloaders[plate].keys():
                set_name = 'plate ' + plate + ', population ' + key
                plate_res = test(model, test_dataloaders[plate][key], input_size, input_channels, set_name, exp_dir)
                plate_results.append(plate_res)

            plate_res = pd.concat(plate_results)
            plate_res.to_csv(res_plate_path, index=False)
        else:
            plate_res = pd.read_csv(res_plate_path)

        results.append(plate_res)

    res = pd.concat(results)
    return res


def test(model, data_loader, input_size, input_channels=4, title='', save_dir='', show_images=True):
    start = 0
    results = pd.DataFrame(
        columns=['Plate', 'Well', 'Site', 'ImageNumber', 'Well_Role', 'Broad_Sample', 'PCC', 'MSE'])

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        (inp, target, ind) = batch
        rec = data_loader.dataset.metadata_file.iloc[ind].drop([c.name for c in Channels], axis=1)
        pred, mse, pcc = unify_test_function(model, batch)
        results = results.append(rec, ignore_index=True)
        results.PCC[start] = pcc.cpu().numpy()
        results.MSE[start] = mse.cpu().numpy()

        if show_images and start == 0:
            pred = pred.cpu().detach().numpy()
            # TODO: Reverse transform (?)
            if input_channels == 5:
                show_input_and_target(inp.cpu().detach().numpy()[0, :, :, :],
                                      pred=pred[0, :, :, :], title=title, save_dir=save_dir)
                # TODO: Check of use target channel
            else:
                show_input_and_target(inp.cpu().detach().numpy()[0, :, :, :],
                                      target.cpu().detach().numpy()[0, :, :, :],
                                      pred[0, :, :, :], title, save_dir,
                                      target_channel=args.target_channel)
        start += 1

    return results


def save_results(res, args, kwargs={}):
    for arg in kwargs:
        res[arg] = kwargs[arg]

    res_dir = os.path.join(args.exp_dir, 'results.csv')
    if is_file_exist(res_dir):
        prev_res = pd.read_csv(res_dir)
        res = pd.concat([prev_res, res])

    # save_to_pickle(res, os.path.join(args.exp_dir, 'results.pkl'))
    save_to_pickle(args, os.path.join(args.exp_dir, 'args.pkl'))
    res.to_csv(res_dir, index=False)


def main(Model, args, kwargs={}):
    print_exp_description(Model, args, kwargs)

    logging.info('Preparing data...')
    dataloaders = load_data(args)
    logging.info('Preparing data finished.')

    # dataloaders['val'].dataset.__getitem__(0, True)
    # dataloaders['val_for_test'].dataset.__getitem__(0, True)
    # exit(42)

    model = Model.model_class(**Model.params, batch_size=args.batch_size)
    if args.mode == 'predict' and args.checkpoint is not None:
        logging.info('loading model from file...')
        model = model.load_from_checkpoint(args.checkpoint)
        model.to(args.device)
        logging.info('loading model from file finished')

        logging.info('testing model...')
        model.eval()

        # # # Check test process
        # t_plate = '24357'  # '24294'
        # inp0, target0, idx0 = dataloaders['test'][t_plate]['mock'].dataset.__getitem__(0)
        # inp1, target1, idx1 = dataloaders['test'][t_plate]['mock'].dataset.__getitem__(1)
        # batch = torch.stack([inp0, inp1]).to(args.device), torch.stack([target0, target1]).to(args.device)
        # res = unify_test_function(model, batch)

        res = test_by_partition(model, dataloaders['test'], args.input_size, args.num_input_channels, args.exp_dir)
        logging.info('testing model finished...')

        save_results(res, args, kwargs)

    else:
        logging.info('training model...')
        model.to(args.device)
        logger = TensorBoardLogger(args.log_dir,
                                   name='log_dir')  # Model.name + " on channel" + args.target_channel.name)
        trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=1, logger=logger, gpus=1,
                             auto_scale_batch_size='binsearch', weights_summary='full')
        trainer.fit(model, dataloaders['train'], dataloaders['val_for_test'])
        logging.info('training model finished.')


def print_exp_description(Model, args, kwargs):
    description = 'Training Model ' + Model.name + ' with target ' + str(args.target_channel)
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
    mf = 16  # 2 ** (inp % 10)
    plate_id = (inp % 100) - 1

    exp_num = inp  # if None, new experiment directory is created with the next available number
    DEBUG = False

    models = [
        # Model_Config.UNET1TO1,
        Model_Config.UNET4TO1,
        # Model_Config.UNET5TO5,
        # Model_Config.AUTO1T01,
        # Model_Config.AUTO4T01
    ]
    # models = [models[model_id]]

    exps = [(input_size, lr, batch_size)
            for input_size in [(128, 128), (130, 116), (260, 232), (256, 256)]
            for batch_size in [16, 32, 36, 64]
            for lr in [1.5e-4, 1.0e-4, 1.5e-3, 1.0e-3, 1.5e-2, 1.0e-2]
            ]
    exp_values = exps[49 - 1]

    input_size, lr, batch_size = exp_values
    exp_dict = {'input_size': input_size, 'lr': lr,
                'epochs': 20, 'minimize_net_factor': mf}

    # channels_to_predict = [Channels.AGP]
    # channels_to_predict = [Channels.DNA]
    # channels_to_predict = [Channels.ER]
    # channels_to_predict = [Channels.Mito]
    # channels_to_predict = [Channels.RNA]

    channels_to_predict = [list(Channels)[channel_id]]

    for model in models:
        model.update_custom_params(exp_dict)
        for target_channel in channels_to_predict:
            # torch.cuda.empty_cache()
            args = config.parse_args(model, target_channel, exp_num)
            args.batch_size = batch_size

            args.mode = 'train'
            args.mode = 'predict'
            # args.plates_split = [
            #     [
            #         24357, 24585, 24623, 24661, 24735, 24792, 25392, 25569, 25588, 25683, 25726, 25912, 25955, 25997,
            #         26207, 26576, 26640, 26674, 26745, 24300, 24509, 24594, 24631, 24683, 24752, 24796, 25406, 25570,
            #         25599, 25686, 25732, 25918, 26115, 26247, 26577, 26641, 26677, 26765, 24303, 24517, 24609, 24685,
            #         24756, 25372, 25422, 25571, 25663, 25692, 25742, 25937, 26124, 26271, 26600, 26662, 26683, 26771,
            #         26724, 26795],
            #     [24294, 24311, 25938, 25985, 25987, 24633,
            #      24309, 24523, 24611, 24634, 24687, 24759, 25374, 25432, 25573, 25664, 25695, 25854, 25989, 26133,
            #      26562, 26611, 26664, 26685, 26786, 24525, 24617, 24640, 24688, 24773, 25380, 25492, 25575, 25676,
            #      25708, 25857, 25944, 25991, 26174, 26563, 26622, 26666, 26695, 26794, 24321, 24562, 24619, 24654,
            #      24734, 24774, 25382, 25566, 25579, 25680, 25725, 25862, 25945, 25994, 26204, 26564, 26623, 26672
            #      ]
            # ]

            plates = [24792, 25912, 24509, 24633, 25987, 25680, 25422, 24517, 25664, 25575, 26674, 25945, 24687, 24752,
                      24311, 26622, 26641, 24594, 25676, 24774, 26562, 25997, 26640, 24562, 25938, 25708, 24321, 24735,
                      26786, 25571, 26666, 24294, 24640, 25985, 24661]

            args.plates_split = [
                [p for p in plates if p != plates[plate_id]],
                [plates[plate_id]]  # , 25862, 26672, 24734, 25991, 25382, 25566]
            ]

            args.test_samples_per_plate = None
            args.checkpoint = config.get_checkpoint(args.log_dir, model.name, args.target_channel)

            if DEBUG:
                args.test_samples_per_plate = 10
                args.epochs = 5

            main(model, args)

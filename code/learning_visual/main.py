import logging
import os

import pandas as pd
import pytorch_lightning as pl
import scipy
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from configuration import config
from configuration.model_config import Model_Config
from data_layer.prepare_data import load_data
from data_layer.dataset import Channels
from process_images import process_image
from util.files_operations import save_to_pickle, is_file_exist
from visuals.visualize import show_input_and_target


def test_by_partition(model, test_dataloaders, input_size, input_channels, exp_dir=None):
    # res = {}
    res = pd.DataFrame()
    for plate in list(test_dataloaders):
        # res[plate] = {}
        for key in test_dataloaders[plate].keys():
            # res[plate][key] = []
            set_name = 'plate ' + plate + ', population ' + key
            plate_res = test(model, test_dataloaders[plate][key], input_size, input_channels, set_name, exp_dir)
            # res[plate][key].append(results)
            res = pd.concat([res, plate_res])
    return res


def test(model, data_loader, input_size, input_channels=4, title='', save_dir='', show_images=True):
    start = 0
    results = pd.DataFrame(
        columns=['Plate', 'Well', 'Site', 'ImageNumber', 'Well_Role', 'Broad_Sample', 'PCC'])
    # results = {}
    pccs = []
    for i, (input, target, ind) in tqdm(enumerate(data_loader), total=len(data_loader)):

        # deformation to patches and reconstruction based on https://discuss.pytorch.org/t/creating-nonoverlapping-patches-from-3d-data-and-reshape-them-back-to-the-image/51210/6
        rec = data_loader.dataset.metadata_file.iloc[ind].drop([c.name for c in Channels], axis=1)
        pred = process_image(model, input, input_size, input_channels)
        pcc, p_value = scipy.stats.pearsonr(pred.flatten(), target.cpu().detach().numpy().flatten())
        results = results.append(rec, ignore_index=True)
        results.PCC[start] = pcc
        # pccs.append(pcc)

        if show_images and start == 0:
            if input_channels == 5:
                show_input_and_target(input.cpu().detach().numpy()[0, :, :, :],
                                      pred=pred, title=title, save_dir=save_dir)
            else:
                show_input_and_target(input.cpu().detach().numpy()[0, :, :, :],
                                      target.cpu().detach().numpy()[0, :, :, :], pred, title, save_dir)
        start += 1

    # results['pcc'] = pccs

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
    res.to_csv(res_dir)


def main(Model, args, kwargs={}):
    print_exp_description(Model, args, kwargs)

    logging.info('Preparing data...')
    dataloaders = load_data(args)
    logging.info('Preparing data finished.')

    model = Model.model_class(**Model.params)
    args.checkpoint = config.get_checkpoint(args.log_dir, Model.name, args.target_channel)
    if args.mode == 'predict' and args.checkpoint is not None:
        logging.info('loading model from file...')
        model = model.load_from_checkpoint(args.checkpoint)
        model.to(args.device)
        logging.info('loading model from file finished')

    else:
        logging.info('training model...')
        model.to(args.device)
        logger = TensorBoardLogger(args.log_dir, name=Model.name + " on channel" + args.target_channel.name)
        trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=1, logger=logger, gpus=1,
                             auto_scale_batch_size='binsearch', weights_summary='full')
        trainer.fit(model, dataloaders['train'], dataloaders['val'])
        logging.info('training model finished.')

    logging.info('testing model...')
    res = test_by_partition(model, dataloaders['test'], args.input_size, args.num_input_channels, args.exp_dir)
    logging.info('testing model finished...')

    save_results(res, args, kwargs)


def print_exp_description(Model, args, kwargs):
    description = 'Training Model ' + Model.name + ' with target ' + str(args.target_channel)
    for arg in kwargs:
        description += ', ' + arg + ': ' + str(kwargs[arg])
    print(description)


if __name__ == '__main__':
    exp_num = 2  # if None, new experiment directory is created with the next available number
    DEBUG = False

    models = [
        # Model_Config.UNET1TO1,
        Model_Config.UNET4TO1,
        # Model_Config.UNET5TO5
    ]

    # channels_to_predict = [Channels.AGP]  # <-- 159342
    # channels_to_predict = [Channels.DNA]  # <-- 159343
    # channels_to_predict = [Channels.ER]  # <-- 159344
    # channels_to_predict = [Channels.Mito]  # <-- 159345
    channels_to_predict = [Channels.RNA]  # <-- 159346
    description = f'Checking 4to1 prediction on {" ".join(c.name for c in channels_to_predict)}'
    print(description)

    for model in models:
        for target_channel in channels_to_predict:
            # torch.cuda.empty_cache()
            args = config.parse_args(exp_num, target_channel=target_channel, model_type=model.name)
            args.num_input_channels = model.value[2]['n_input_channels']

            args.mode = 'train'
            args.plates_split = [[24509, 24562, 24640, 24687, 24752, 25571, 25676, 25945, 26562, 26577, 26765, 24517, 24661, 24774, 25680, 25912, 26640, 26666, 26786, 24321, 24594, 24735, 24792, 25422, 25575, 25664, 25708, 25997, 26576, 26622, 26641, 26674],
                                 [24294, 24311, 25938, 25985, 25987, 24633]]

            args.test_samples_per_plate = None
            args.batch_size = 36
            args.input_size = 128

            if DEBUG:
                args.test_samples_per_plate = 5
                args.epochs = 3

            main(model, args)

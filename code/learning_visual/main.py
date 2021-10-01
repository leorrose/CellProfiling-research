import logging
import os

import pandas as pd
import pytorch_lightning as pl
import scipy
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from configuration import config
from configuration.model_config import Model_Config
from data_layer.dataset import Channels
from data_layer.prepare_data import load_data
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
        rec = data_loader.dataset.metadata_file.iloc[ind].drop([c.name for c in Channels], axis=1)
        pred = process_image(model, input, input_size, input_channels)
        pcc, p_value = scipy.stats.pearsonr(pred.flatten(), target.cpu().detach().numpy().flatten())
        results = results.append(rec, ignore_index=True)
        results.PCC[start] = pcc
        # pccs.append(pcc)

        if show_images and start == 0:
            # TODO: Reverse transform
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

    # # Check test process
    # inp, target, idx = dataloaders['test']['24294']['mock'].dataset.__getitem__(0)
    # pred = process_image(model, inp.unsqueeze(0), args.input_size, args.num_input_channels)
    # pcc, p_value = scipy.stats.pearsonr(pred.flatten(), target.cpu().detach().numpy().flatten())

    logging.info('testing model...')
    res = test_by_partition(model, dataloaders['test'], args.input_size, args.num_input_channels, args.exp_dir)
    logging.info('testing model finished...')

    save_results(res, args, kwargs)


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
    # 2 with old mean & std
    # 3 with new mean & std
    # 4 with div (159949-159953)
    # 5 no div, changed input to 256 (159954-159958)
    # 6 load images with div by 65535, no mean & std transform (160065-160069)
    # 7 like 6 with output sigmoid (160086, 160077-160080)
    # 8 with div, more plates and epochs (160145-160149)
    # --- Unet scale changed to 1
    # 9 with div fix(255 because of ToTensor)
    # 10 520x696 input
    # 11 10 with lr 1.5e-3
    exp_num = 11  # if None, new experiment directory is created with the next available number
    DEBUG = False

    models = [
        # Model_Config.UNET1TO1,
        Model_Config.UNET4TO1,
        # Model_Config.UNET5TO5
    ]

    import sys

    chan_id = int(sys.argv[1]) - 1

    # channels_to_predict = [Channels.AGP]
    # channels_to_predict = [Channels.DNA]
    # channels_to_predict = [Channels.ER]
    # channels_to_predict = [Channels.Mito]
    channels_to_predict = [list(Channels)[chan_id]]

    for model in models:
        for target_channel in channels_to_predict:
            # torch.cuda.empty_cache()
            args = config.parse_args(exp_num, target_channel=target_channel, model_type=model.name)
            args.num_input_channels = model.value[2]['n_input_channels']

            args.mode = 'train'
            args.mode = 'predict'
            args.plates_split = [
                [
                    24357, 24585, 24623, 24661, 24735, 24792, 25392, 25569, 25588, 25683, 25726, 25912, 25955, 25997,
                    26207, 26576, 26640, 26674, 26745, 24300, 24509, 24594, 24631, 24683, 24752, 24796, 25406, 25570,
                    25599, 25686, 25732, 25918, 26115, 26247, 26577, 26641, 26677, 26765, 24303, 24517, 24609, 24685,
                    24756, 25372, 25422, 25571, 25663, 25692, 25742, 25937, 26124, 26271, 26600, 26662, 26683, 26771,
                    24309, 24523, 24611, 24634, 24687, 24759, 25374, 25432, 25573, 25664, 25695, 25854, 25989, 26133,
                    26562, 26611, 26664, 26685, 26786, 24525, 24617, 24640, 24688, 24773, 25380, 25492, 25575, 25676,
                    # 25708, 25857, 25944, 25991, 26174, 26563, 26622, 26666, 26695, 26794, 24321, 24562, 24619, 24654,
                    # 24734, 24774, 25382, 25566, 25579, 25680, 25725, 25862, 25945, 25994, 26204, 26564, 26623, 26672,
                    26724, 26795],
                [24294, 24311, 25938, 25985, 25987, 24633]]

            args.test_samples_per_plate = None
            args.batch_size = 32
            args.input_size = model.params['input_size']
            args.checkpoint = config.get_checkpoint(args.log_dir, model.name, args.target_channel)

            if DEBUG:
                args.test_samples_per_plate = 5
                args.epochs = 3

            main(model, args)

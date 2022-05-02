import sys

from configuration.config import parse_args
from configuration.model_config import Model_Config
from data_layer.prepare_data import load_data

from time import time


def main(model, args, kwargs={}):
    print_exp_description(model, args, kwargs)

    s = time()
    dataloaders = load_data(args)
    e = time()
    print(f'Done loading data in {e - s} seconds')


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

    exps = [(input_size, lr, batch_size)
            for input_size in [(128, 128), (130, 116), (260, 232), (256, 256)]
            for batch_size in [16, 32, 36, 64]
            for lr in [1.5e-4, 1.0e-4, 1.5e-3, 1.0e-3, 1.5e-2, 1.0e-2]
            ]
    exp_values = exps[49 - 1]

    input_size, lr, batch_size = exp_values
    exp_dict = {'input_size': input_size, 'lr': lr,
                'epochs': 20, 'latent_space_dim': lsd}

    channels_to_predict = [1]

    model = Model_Config.TAE
    model.update_custom_params(exp_dict)
    for target_channel in channels_to_predict:
        # torch.cuda.empty_cache()
        args = parse_args(channel_idx=target_channel, exp_num=exp_num)
        args.batch_size = batch_size

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
        args.split_ratio = 0.1

        args.test_samples_per_plate = None

        if DEBUG:
            args.test_samples_per_plate = 10
            args.epochs = 5

        main(model, args)

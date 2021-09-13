from enum import Enum
from model_layer.UNET import Unet
from model_layer.UNETnoBypass import UnetnoBypass
input_size = 128
lr = 1.5e-4
epochs = 25
minimize_net_factor = 4


class Model_Config(Enum):

    UNET4TO1 = ('unet_4to1', Unet, {'n_input_channels':4, 'n_classes':1, 'input_size':input_size, 'lr': lr, 'epochs': epochs, 'minimize_net_factor': minimize_net_factor})
    UNET5TO5 = ('unet_5to5', Unet, {'n_input_channels':5, 'n_classes':5, 'input_size':input_size, 'lr': lr, 'epochs': epochs, 'minimize_net_factor': minimize_net_factor})
    UNET1TO1 = ('unet_1to1', Unet, {'n_input_channels':1, 'n_classes':1, 'input_size':input_size, 'lr': lr, 'epochs': epochs, 'minimize_net_factor': minimize_net_factor})
    AUTO4T01 = ('auto_4to1', UnetnoBypass,
                {'n_input_channels': 4, 'n_classes': 1, 'input_size': input_size, 'lr': lr, 'epochs': epochs,
                 'minimize_net_factor': minimize_net_factor})

    def __init__(self, model_name, model_class, params):
        self.model_name = model_name
        self.model_class = model_class
        self.params = params


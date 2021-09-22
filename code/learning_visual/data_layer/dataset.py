import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from visuals.visualize import show_input_and_target

from enum import Enum, auto

use_cuda = torch.cuda.is_available()
print("USE CUDA=" + str(use_cuda))
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


class Channels(Enum):
    AGP = auto()
    DNA = auto()
    ER = auto()
    Mito = auto()
    RNA = auto()


RGB_MAP = {
    1: {
        'rgb': np.array([19, 0, 249]),
        'range': [0, 51]
    },
    2: {
        'rgb': np.array([42, 255, 31]),
        'range': [0, 107]
    },
    3: {
        'rgb': np.array([255, 0, 25]),
        'range': [0, 64]
    },
    4: {
        'rgb': np.array([45, 255, 252]),
        'range': [0, 191]
    },
    5: {
        'rgb': np.array([250, 0, 253]),
        'range': [0, 89]
    },
    6: {
        'rgb': np.array([254, 255, 40]),
        'range': [0, 191]
    }
}


def convert_tensor_to_rgb(t, channels=Channels, vmax=65535, rgb_map=RGB_MAP):
    """
    Converts and returns the image data as RGB image
    Parameters
    ----------
    t : np.ndarray
        original image data
    channels : Enum
        channels to include
    vmax : int
        the max value used for scaling
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.
    Returns
    -------
    np.ndarray the image data of the site as RGB channels
    """
    colored_channels = []
    for i, channel in enumerate(channels):
        x = (t[:, :, i] / vmax) / \
            ((rgb_map[channel.value]['range'][1] - rgb_map[channel.value]['range'][0]) / 255) + \
            rgb_map[channel.value]['range'][0] / 255
        x = np.where(x > 1., 1., x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel.value]['rgb']).reshape(512, 512, 3),
            dtype=int)
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    return im


class CovidDataset(Dataset):

    def __init__(self,
                 metadata_df,
                 target_channel,
                 root_dir,
                 input_channels=4,
                 transform=None,
                 im_shape=(520, 696),
                 for_data_statistics_calc=False,
                 is_test=False):

        super(Dataset).__init__()
        self.metadata_file = metadata_df
        self.root_dir = root_dir
        self.target_channel = target_channel.value if target_channel else None
        self.input_channels = input_channels
        self.transform = transform
        self.im_shape = im_shape

        self.for_data_statistics_calc = for_data_statistics_calc
        self.is_test = is_test

    def __len__(self):
        return len(self.metadata_file)

    def load_images_as_tensor(self, image_paths, dtype=np.uint8):
        n_channels = len(image_paths)

        data = np.ndarray(shape=(*self.im_shape, n_channels), dtype=dtype)

        for ix, img_path in enumerate(image_paths):
            data[:, :, ix] = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)

        return data

    def image_path(self, channel, plate=None, well=None, site=None, index=None):
        """
        Returns the path of a channel image.
        Parameters
        ----------
        channel : str
            channel name
        plate : int
            plate number
        well : str
            well id
        site : int
            site number
        index : int
            well index in metadata table

        Returns
        -------
        str the path of image
        """
        if index is not None:
            metadata_row = self.metadata_file.iloc[index]
            plate = metadata_row.get('Plate')
            filename = metadata_row.get(channel)
        elif plate is not None and well is not None and site is not None:
            filename = \
                self.metadata_file.query(f'Plate == {plate} and Well == "{well}" and Site == {site}')[channel].iat[0]
        else:
            raise 'image_path, need to provide plate-well-site or index'

        if self.root_dir is None or plate is None or filename is None:
            print(self.root_dir, str(plate), filename)

        return os.path.join(self.root_dir, str(plate), filename)

    def load_site(self, plate=None, well=None, site=None, index=None):
        """
        Returns the image data of a site
        Parameters
        ----------
        plate : int
            plate number
        well : str
            well id
        site : int
            site number
        index : int
            well index in metadata table
        Returns
        -------
        np.ndarray the image data of the site
        """
        input_paths = [
            self.image_path(c.name, plate, well, site, index)
            for c in Channels
        ]
        return self.load_images_as_tensor(input_paths)

    def __getitem__(self, idx, show_sample=False):
        inp = self.load_site(index=idx)
        if not self.for_data_statistics_calc:
            if show_sample:
                trans_input = np.zeros((inp.shape[2], inp.shape[0], inp.shape[1]))
                for i in range(5):
                    trans_input[i, :, :] = inp[:, :, i]
                show_input_and_target(trans_input, title='before transforms')
            if self.transform:
                inp = self.transform(inp)
                if show_sample:
                    show_input_and_target(inp, title='after transforms')
            if self.input_channels == 4:
                inp, target = self.split_target_from_tensor(inp, show_sample)
            elif self.input_channels == 1:
                inp, target = inp[self.target_channel - 1:self.target_channel, :, :], \
                              inp[self.target_channel - 1:self.target_channel, :, :]
            elif self.input_channels == 5:
                target = inp
            else:
                raise ValueError('Number of input channels is not supported')
            if self.is_test:
                # rec = dict_fields_to_str(rec.to_frame().to_dict()[rec.name])
                return inp, target, idx
            else:
                return inp, target
        else:
            return inp

    def split_target_from_tensor(self, inp, show_sample=False):

        num_channels = inp.shape[0]

        if self.target_channel == 1:

            target, inp = torch.split(inp, [1, num_channels - 1])
        elif self.target_channel == num_channels:
            inp, target = torch.split(inp, [num_channels - 1, 1])
        else:
            after = num_channels - self.target_channel
            before = num_channels - after - 1
            a, target, c = torch.split(inp, [before, 1, after])
            inp = torch.cat((a, c), dim=0)

        if show_sample:
            show_input_and_target(inp.detach().numpy(), target.detach().numpy(),
                                  title='after split to train and target')

        return inp, target

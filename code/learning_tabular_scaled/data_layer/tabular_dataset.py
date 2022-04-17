import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()
print("USE CUDA=" + str(use_cuda))
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

# Pixel Attributes
BITS_PER_PIXEL = 16  # 16 for tiff, 8 for png
VMAX = 2 ** BITS_PER_PIXEL - 1
DTYPE = f'float32'


class TabularDataset(Dataset):
    def __init__(self,
                 metadata_df,
                 root_dir,
                 input_fields,
                 target_fields=None,
                 target_channel=None,
                 transform=None,
                 for_data_statistics_calc=False,
                 is_test=False):

        super(Dataset).__init__()
        self.metadata_df = metadata_df
        metadata_df['CumCount'] = metadata_df['Count'].cumsum()
        self.root_dir = root_dir
        self.target_channel = target_channel.value if target_channel else None
        self.target_channel_enum = target_channel
        self.input_fields = input_fields
        self.target_fields = target_fields
        self.transform = transform

        self.for_data_statistics_calc = for_data_statistics_calc
        self.is_test = is_test

    def __len__(self):
        return self.metadata_df['Count'].sum()

    def load_cell(self, index):
        """
        Returns the tabular data of an index
        Parameters
        ----------
        index : int
            cell index in metadata table
        Returns
        -------
        np.ndarray the tabular data of an index
        """

        p, lbl, idxs, c, cc = self.metadata_df[self.metadata_df['CumCount'] > index].iloc[0]
        index_idx = index - (cc - c)
        row_idx = idxs[index_idx] + 1
        plate_path = os.path.join(self.root_dir, f'{p}.csv')
        fields_names = pd.read_csv(plate_path, nrows=0).columns
        row = pd.read_csv(plate_path, nrows=1, skiprows=row_idx, names=fields_names)
        data = row[self.input_fields + self.target_fields]
        return data.to_numpy().squeeze()

    def __getitem__(self, idx, show_sample=False):
        inp = self.load_cell(idx)
        if not self.for_data_statistics_calc:
            # if show_sample:
            #     trans_input = np.zeros((inp.shape[2], inp.shape[0], inp.shape[1]))
            #     for i in range(5):
            #         trans_input[i, :, :] = inp[:, :, i]
            #     show_input_and_target(trans_input, title='before transforms')
            if self.transform:
                inp = self.transform(inp)
                # if show_sample:
                #     show_input_and_target(inp, title='after transforms')

            inp, target = inp[:len(self.input_fields)], inp[len(self.input_fields):]

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
                                  title='after split to train and target', target_channel=self.target_channel_enum)

        return inp, target

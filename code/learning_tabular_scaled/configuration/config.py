# In[1] Constants and params:
import json
import sys
from argparse import Namespace

import torch

args = {
    'channels': ['AGP', 'DNA', 'ER', 'Mito', 'RNA'],
    'families': ['Granularity', 'Intensity', 'Location', 'RadialDistribution', 'Texture'],
    'labels': ['mock', 'treated'],
    'train_labels': ['mock'],
    'label_field': 'Metadata_ASSAY_WELL_ROLE',
    'index_cols': ['Plate', 'Metadata_ASSAY_WELL_ROLE', 'Metadata_broad_sample', 'Image_Metadata_Well', 'ImageNumber', 'ObjectNumber'],
    'cols_file': r'/storage/users/g-and-n/plates/columns.txt',

    # Load Data
    'metadata_path': '/storage/users/g-and-n/plates/tabular_metadata.csv',
    'plates_path': '/storage/users/g-and-n/plates/csvs/',
    # 'plates_split': ([25732, 26564, 26572, 26575], [26569, 26574]),
    'plates_split': ([24792,25912,24509,24633,25987,25680,25422,24517,25664,25575,26674,25945,24687,24752,24311,26622,26641,24594,25676,24774,26562,25997,26640,24562,25938,25708,24321,24735,26786,25571,26666,24294,24640,25985,24661], [24278]),
    'split_ratio': 0.1,
    'input_channels': ['GENERAL', 'DNA', 'ER', 'Mito', 'RNA'],
    'target_channels': ['AGP'],
    'test_samples_per_plate': None,  # Remove? needed for debug purposes

    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    # Loader
    'batch_size': 1024 * 8,
    'num_data_workers': 12,
}

inp = int(sys.argv[1])
args['input_channels'] = ['GENERAL'] + args['channels'][:inp] + args['channels'][inp+1:]
args['target_channels'] = [args['channels'][inp]]

args['cols_dict'] = json.load(open(args['cols_file'], 'r'))
args['input_fields'] = sum([args['cols_dict'][k] for k in args['input_channels']], [])
args['target_fields'] = sum([args['cols_dict'][k] for k in args['target_channels']], [])
n_pth = fr"/storage/users/g-and-n/plates/01{'_'.join(args['input_channels'])}-{'_'.join(args['target_channels'])}.normsav"
args['norm_params_path'] = n_pth

args = Namespace(**args)

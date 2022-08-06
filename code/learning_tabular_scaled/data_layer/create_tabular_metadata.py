import os
import random
import sys

import pandas as pd


def sample_split_plate(plate, by_field, sample_n):
    values = set(plate[by_field].unique().tolist())
    sample = set(random.sample(values, sample_n)) if sample_n < len(values) else values.copy()
    return sample, values - sample


def create_tabular_metadata(plates_path, plates, label_field, train_labels, by_fld, sample_n):
    mt_dict = {'Plate': [], label_field: [], 'Mode': [], by_fld: [], 'Count': []}
    for plate in plates:
        plate_path = os.path.join(plates_path, f'{plate}.csv')
        df = pd.read_csv(plate_path)
        for lbl in df[label_field].unique():
            qdf = df.query(f'{label_field} == "{lbl}"')
            if lbl in train_labels:
                train_set, test_set = sample_split_plate(qdf, by_fld, sample_n)
                train_count = qdf[by_fld].isin(train_set).sum()
                test_count = qdf.shape[0] - train_count
                add_metadata(mt_dict, label_field, by_fld, plate, lbl, 'train', train_set, train_count)
            else:
                test_set = set(qdf[by_fld].unique())
                test_count = qdf.shape[0]

            if test_count:
                add_metadata(mt_dict, label_field, by_fld, plate, lbl, 'predict', test_set, test_count)
        del qdf, df

    return pd.DataFrame(mt_dict)


def add_metadata(mt_dict, label_field, by_fld, plate, lbl, mode, train_set, train_count):
    mt_dict['Plate'].append(plate)
    mt_dict[label_field].append(lbl)
    mt_dict['Mode'].append(mode)
    mt_dict[by_fld].append(train_set)
    mt_dict['Count'].append(train_count)

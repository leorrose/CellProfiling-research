import pandas
from tqdm import tqdm

from constants import *
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Preprocessing functions

# In[1] Scaling:


def fit_scaler(df, scale_method):
    """
    This function is fitting a scaler using one of two methods: STD and MinMax
    df: dataFrame to fit on
    scale_method: 'Std' or 'MinMax'
    return: scaled dataframe, according to StandardScaler or according to MinMaxScaler
    """

    if scale_method == S_STD:
        scaler = StandardScaler()
    elif scale_method == S_MinMax:
        scaler = MinMaxScaler()
    else:
        scaler = None

    if not scaler:
        return None

    return scaler.fit(df)


def scale_data(df, scaler):
    scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    scaled_df.fillna(0, inplace=True)
    return scaled_df


# In[2] Splitting:

def split_by_channel(filename, task_channel):
    """
    This function is responsible for splitting five channels into four channels as train and the remaining channel to test
    filename: file path to the cell table from a single plate
    task_channel: the current channel that we aim to predict

    Notably: In order to avoid leakage we drop all 'correlation features
    return: separated dataframes x_features and y_df.
            x_features: contains all available features excluding the features related to 'task_channel' we aim to predict
            y_df: contains all available features related to 'task_channel' only
    """

    # Data preparation
    df = pd.read_csv(filename)
    df = df.set_index([LABEL_FIELD, 'Metadata_broad_sample', 'Image_Metadata_Well', 'ImageNumber', 'ObjectNumber'])
    df.drop(['TableNumber'], inplace=True, axis=1)
    df.dropna(inplace=True)

    general_cols = [f for f in df.columns if all(c not in f for c in CHANNELS)]
    corr_cols = [f for f in df.columns if 'Correlation' in f]

    # Split columns by channel
    dict_channel_cols = {}
    for channel in CHANNELS:
        dict_channel_cols[channel] = [col for col in df.columns if channel in col and col not in corr_cols]

    not_curr_channel_cols = [col for channel in CHANNELS if channel != task_channel
                             for col in dict_channel_cols[channel]]
    cols = general_cols + not_curr_channel_cols

    x_features_df = df[cols]

    y_df = df[dict_channel_cols[task_channel]]

    return x_features_df, y_df


def split_train_test(path, csv_files, test_plate, task_channel):
    # Prepare test samples
    df_test_x, df_test_y = split_by_channel(path + test_plate, task_channel)

    # print(df_test_x.index.unique(0).tolist())
    # print(df_test_x.index.unique(1).tolist())

    df_test_mock_x = df_test_x[df_test_x.index.isin(['mock'], 0)]
    df_test_treated_x = df_test_x[df_test_x.index.isin(['treated'], 0)]
    df_test_mock_y = df_test_y.loc[df_test_mock_x.index]
    df_test_treated_y = df_test_y.loc[df_test_treated_x.index]

    # Prepare train samples - only mock
    list_x_df = []
    list_y_df = []
    for train_plate in tqdm(csv_files):
        if train_plate != test_plate:
            curr_x, curr_y = split_by_channel(path + train_plate, task_channel)
            curr_x = curr_x[curr_x.index.isin(['mock'], 0)]
            curr_y = curr_y.loc[curr_x.index]

            list_x_df.append(curr_x)
            list_y_df.append(curr_y)

    df_train_x = pd.concat(list_x_df)
    df_train_y = pd.concat(list_y_df)

    return df_test_mock_x, df_test_mock_y, df_test_treated_x, df_test_treated_y, df_train_x, df_train_y
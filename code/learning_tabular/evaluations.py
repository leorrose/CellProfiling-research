from os import path, listdir

from sklearn.metrics import mean_squared_error, mean_absolute_error

from learning_tabular.constants import ERROR_TYPE


# In[1]:

def get_error(df_predict, df_true, multioutput='uniform_average'):
    if ERROR_TYPE == 'MSE':
        return mean_squared_error(df_true, df_predict, multioutput=multioutput)
    elif ERROR_TYPE == 'RMSE':
        return mean_squared_error(df_true, df_predict, squared=False, multioutput=multioutput)
    elif ERROR_TYPE == 'MAE':
        return mean_absolute_error(df_true, df_predict, multioutput=multioutput)
    else:
        raise Exception("ERROR_TYPE NOT IN ['RMSE' | 'MSE' | 'MAE']")


# In[2]:


def print_results(plate_number, channel, family, model, _type, metric, value, inter_channel=True):
    """
    This function is creating a csv named: 'results' that contains all of the models’ performance (e.g. MSE) for each plate and each family of attributes
    plate_number: ID of plate
    channel: The channel we aim to predict
    family: features united by their charactheristics (e.g., Granularity, Texture)
    model: the model name
    _type: scaling method (e.g., MinMax Scaler or StandardScaler)
    metric: MSE/MAE
    value: value of the metric error
    """
    inter_str = '' if inter_channel else '1to1'
    results_path = f'results{inter_str}'
    file_path = path.join(results_path, 'results.csv')
    files_list = listdir(results_path)
    if 'results.csv' not in files_list:
        file1 = open(file_path, "a+")
        file1.write("Plate,Channel,Family,Model,Type,Metric,Value \n")
        file1.write(
            plate_number + "," + channel + "," + family + "," + model + "," + _type + "," + metric + "," + value + "\n")
        file1.close()
    else:
        file1 = open(file_path, "a+")
        file1.write(
            plate_number + "," + channel + "," + family + "," + model + "," + _type + "," + metric + "," + value + "\n")
        file1.close()


# In[1]:


def get_family_error(test_plate_number, task_channel, model, _type, df, channel_task_y):
    """
    This function is calculating the MSE/MAE measures for plates based on different models
    test_plate_number: ID of the examine plate
    task_channel: Channel we aim to predict
    model: model name
    _type: scaling method (e.g., MinMax Scaler or StandardScaler)
    df: prediction of any given ML model which aim to predict the channel_task_y
    channel_task_y: features corresponding to the 'task channel' (channel we aim to predict)
    """
    Families = {'Granularity': [],
                'Intensity': [],
                'Location': [],
                'RadialDistribution': [],
                'Texture': []}

    for name in channel_task_y.columns:
        if '_Granularity' in name:
            Families['Granularity'].append(name)
        elif '_Intensity' in name:
            Families['Intensity'].append(name)
        elif '_Location' in name:
            Families['Location'].append(name)
        elif '_RadialDistribution' in name:
            Families['RadialDistribution'].append(name)
        elif '_Texture' in name:
            Families['Texture'].append(name)

    for key in Families.keys():
        try:
            print_results(test_plate_number, task_channel, key, model, _type, ERROR_TYPE,
                          str(get_error(df[Families[key]], channel_task_y[Families[key]])))
        except:
            if len(Families[key]) == 0:
                print(f'empty family {key} in channel {task_channel}')
            else:
                print('problem in key')

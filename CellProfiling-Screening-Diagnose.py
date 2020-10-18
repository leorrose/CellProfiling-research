#!/usr/bin/env python
# coding: utf-8

# # Imports

# In this cell we are importing all relevant packages for our project

# In[2]:


# connections and OS
import pandas as pd
#import seaborn as sns
import os
import sqlite3
#import csv

#utils (Pandas,numpy,tqdm)
import numpy as np
import pandas as pd
from tqdm import tqdm

#visualize 
#import seaborn as sns

#preprocessing, metrices and splits 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#ML models:
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge


#tensorflow layer, callbacks and layers
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.callbacks import ModelCheckpoint, Callback


# # Constants and params

# Change directory to project directory within the department cluster (SLURM) and define to constant variables
# 

# In[3]:


#change directory to project directory within the department cluster (SLURM)
PROJECT_DIRECTORY = r'/storage/users/g-and-n/plates/'

CHANNELS = ["AGP","DNA","ER","Mito","RNA"]
LABEL_FIELD = 'Metadata_ASSAY_WELL_ROLE'
S_STD = 'Std'
S_MinMax = 'MinMax'


# In[4]:


os.chdir(PROJECT_DIRECTORY)


# # Preprocessing functions

# In[5]:


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


# In[6]:


def scale_data(df, scaler):
    scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    scaled_df.fillna(0,inplace=True)
    return scaled_df


# In[7]:


def split_channels_x_and_y(filename, task_channel):
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


# # Create Models

# In the following three cells we are creating three ML models

# In[35]:


def create_LR(df_train_X, df_train_Y):  
    """
    In this cell we are creating and training a linear regression model        
    df_train_X: contains all available features excluding the features related to 'task_channel' we aim to predict (train)
    df_train_Y: contains all available features related to 'task_channel' only for the train
    
    
    return: trained linear regression model
    """
    LR_model = LinearRegression()    
    LR_model.fit(df_train_X.values,df_train_Y.values)
    return LR_model
    
    


# In[36]:


def create_Ridge(df_train_X, df_train_Y):
    """    
    In this cell we are creating and training a ridge regression model    
    
    
    df_train_X: contains all available features excluding the features related to 'task_channel' we aim to predict (train)
    df_train_Y: contains all available features related to 'task_channel' only for the train    
    
    return: trained ridge regression model
    """
    Ridge_model = Ridge()    
    Ridge_model.fit(X=df_train_X.values,y=df_train_Y.values)    
    return Ridge_model


# In[37]:


def create_model_dnn(task_channel,df_train_X, df_train_Y,test_plate):
    """    
    In this cell we are creating and training a multi layer perceptron (we refer to it as deep neural network, DNN) model
    
    task_channel: the current channel that we aim to predict
    df_train_X: contains all available features excluding the features related to 'task_channel' we aim to predict (train)
    df_train_Y: contains all available features related to 'task_channel' only for the train
    test_plate: the ID of a given plate. This information assist us while printing the results.
    
    return: trained dnn model
    """
    # Stracture of the network#
    inputs = Input(shape=(df_train_X.shape[1],))
    dense1 = Dense(512,activation = 'relu')(inputs)
    dense2 = Dense(256,activation = 'relu')(dense1)
    dense3 = Dense(128,activation = 'relu')(dense2)    
    dense4 = Dense(100,activation = 'relu')(dense3)
    dense5 = Dense(50,activation = 'relu')(dense4)
    dense6 = Dense(25,activation = 'relu')(dense5)
    dense7 = Dense(10,activation = 'relu')(dense6)
    predictions = Dense(df_train_Y.shape[1],activation='sigmoid')(dense7)
    
    #model compiliation
    model = Model(inputs=inputs,outputs = predictions)
    model.compile(optimizer='adam',loss='mse')
    
    #model training    
    test_plate_number = test_plate[:5]
    folder = os.path.join(PROJECT_DIRECTORY, 'Models')
    filepath = os.path.join(folder, f'{test_plate_number}_{task_channel}.h5')
    my_callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)]
    model.fit(df_train_X,df_train_Y,epochs = 5,batch_size=1024*8,verbose=0,shuffle=True,validation_split=0.2,callbacks=my_callbacks)
    return model
    
    


# In[38]:


#    print_results(test_plate_number, task_channel, "Overall", "DNN", "None", "MSE", str(mean_squared_error(model_pred,channel_task_y)))
def print_results(plate_number, channel, family, model, _type, metric, value):
    """
    This function is creating a csv named: 'results' that contains all of the models’ performance (e.g. MSE) for each plate and each family of attributes
    plate_number: ID of palte
    channel: The channel we aim to predict
    family: features united by their charactheristics (e.g., Granularity, Texture)
    model: the model name
    _type: scaling method (e.g., MinMax Scaler or StandardScaler)
    metric: MSE/MAE
    value: value of the metric error    
    """
    results_path = os.path.join(PROJECT_DIRECTORY, 'Results')
    file_path = os.path.join(results_path, 'results.csv')
    files_list = os.listdir(results_path)
    if 'results.csv' not in files_list:
        file1 = open(file_path,"a+")     
        file1.write("Plate,Channel,Family,Model,Type,Metric,Value \n")
        file1.write(plate_number+","+channel+","+family+","+model+","+_type+","+metric+","+value+"\n")
        file1.close()
    else:
        file1 = open(file_path, "a+")
        file1.write(plate_number+","+channel+","+family+","+model+","+_type+","+metric+","+value+"\n")
        file1.close()


# In[39]:


def get_family_MSE(test_plate_number, task_channel, model, _type, df, channel_task_y):    
    """
    This function is calculating the MSE/MAE measures for plates based on different models
    test_plate_number: ID of the examine plate
    task_channel: Channel we aim to predict
    model: model name
    _type: scaling method (e.g., MinMax Scaler or StandardScaler)
    df: prediction of any given ML model which aim to predict the channel_task_y
    channel_task_y: features corresponding to the 'task channel' (channel we aim to predict)    
    """
    Families = {'Granularity':[],
               'Intensity':[],
               'Location':[],
               'RadialDistribution':[],
               'Texture':[]}

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
            print_results(test_plate_number, task_channel, key, model, _type, "MSE", str(mean_squared_error(df[Families[key]],channel_task_y[Families[key]])))
        except:
            if len(Families[key]) == 0:
                print('empty family {}'.format(key))
            else:
                print('problem in mse key')
            


# In[40]:


def get_family_MAE(test_plate_number, task_channel, model, _type, df, channel_task_y):
    
    """
    This function is calculating the MSE/MAE measures for plates based on different models
    test_plate_number: ID of the examine plate
    task_channel: Channel we aim to predict
    model: model name
    _type: scaling method (e.g., MinMax Scaler or StandardScaler)
    df: prediction of any given ML model which aim to predict the channel_task_y
    channel_task_y: features corresponding to the 'task channel' (channel we aim to predict)    
    """
    
    Families = {'Granularity':[],
               'Intensity':[],
               'Location':[],
               'RadialDistribution':[],
               'Texture':[]}

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
            print_results(test_plate_number, task_channel, key, model, _type, "MAE", str(mean_absolute_error(df[Families[key]],channel_task_y[Families[key]])))
        except:
            if len(Families[key]) == 0:
                print('empty family {}'.format(key))
            else:
                print('problem in mae key')
        


# In[ ]:


def extract_dataframes(path, csv_files, test_plate, task_channel):
    # Prepare test samples
    df_test_x, df_test_y = split_channels_x_and_y(path + test_plate, task_channel)
    print(df_test_x.index.unique(0).tolist())
    print(df_test_x.index.unique(1).tolist())
    df_test_mock_x = df_test_x[df_test_x.index.isin(['mock'], 0)]
    df_test_treated_x = df_test_x[df_test_x.index.isin(['treated'], 0)]
    df_test_mock_y = df_test_y.loc[df_test_mock_x.index]
    df_test_treated_y = df_test_y.loc[df_test_treated_x.index]

    # Prepare train samples - only mock
    list_x_df = []
    list_y_df = []
    for train_plate in tqdm(csv_files):
                    if train_plate!=test_plate:
                        curr_x, curr_y = split_channels_x_and_y(path + train_plate, task_channel)
                        curr_x = curr_x[curr_x.index.isin(['mock'], 0)]
                        curr_y = curr_y.loc[curr_x.index]

                        list_x_df.append(curr_x)
                        list_y_df.append(curr_y)

    df_train_x = pd.concat(list_x_df)
    df_train_y = pd.concat(list_y_df)

    return df_test_mock_x, df_test_mock_y, df_test_treated_x, df_test_treated_y, df_train_x, df_train_y


# In[41]:


def main(path, scale_method):
    """
    This is the main function of the preprocessing steps.
    This function will iterate all over the sqlite files and do the following:
    1) prepate train + test files
    2) scale train + test files (x + y values separately)
    3) return: 
        task_channel -> string, reflect the relevant channel for test. For example, 'AGP'
        df_train_X -> DataFrame, (instances,features) for the train set
        df_train_Y -> DataFrame, (instances,labels) for the train set
        channel_task_x -> DataFrame, (instances,features) for the test set
        channel_task_y -> DataFrame, (instances,labels) for the test set
    """

    csv_files= [_ for _ in os.listdir(path) if _.endswith(".csv")]

    treatments = [[],[],[]]
    controls = [[],[],[]]
    for test_plate in csv_files:
        # This is the current file that we will predict
        curr_treatments = [[],[],[]]
        curr_controls = [[],[],[]]
        for task_channel in tqdm(CHANNELS):
            print(test_plate)

            df_test_mock_x, df_test_mock_y, df_test_treated_x, df_test_treated_y, df_train_x, df_train_y =                 extract_dataframes(path,csv_files,test_plate,task_channel)

            # Scale for training set
            x_scaler = fit_scaler(df_train_x, scale_method)
            y_scaler = fit_scaler(df_train_y, scale_method)

            df_train_x_scaled = scale_data(df_train_x, x_scaler)
            df_train_y_scaled = scale_data(df_train_y, y_scaler)

            df_test_treated_x_scaled = scale_data(df_test_treated_x, x_scaler)
            df_test_treated_y_scaled = scale_data(df_test_treated_y, y_scaler)
            df_test_mock_x_scaled = scale_data(df_test_mock_x, x_scaler)
            df_test_mock_y_scaled = scale_data(df_test_mock_y, y_scaler)

            # #### Output
            # os.makedirs('DATA', exist_ok=True)
            # plate_number = test_plate.split('.')[0]
            # df_train_x.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_train_x.csv'))
            # df_train_y.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_train_y.csv'))
            # channel_task_x_mock.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_test_x_mock.csv'))
            # channel_task_x_treated.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_test_x_treated.csv'))
            # channel_task_y_mock.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_test_y_mock.csv'))
            # channel_task_y_treated.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_test_y_treated.csv'))
            #
            # df_train_x_scaled.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_train_x_scaled.csv'))
            # df_train_y_scaled.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_train_y_scaled.csv'))
            # df_test_mock_x_scaled.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_test_x_mock_scaled.csv'))
            # df_test_treated_x_scaled.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_test_x_treated_scaled.csv'))
            # df_test_mock_y_scaled.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_test_y_mock_scaled.csv'))
            # df_test_treated_y_scaled.to_csv(os.path.join('Data', f'{plate_number}_{task_channel}_test_y_treated_scaled.csv'))
            #
            # ####
            

            # Model Creation - AVG MSE for each model:
            print(test_plate)
            print(task_channel+":")
            LR_model = create_LR(df_train_x_scaled, df_train_y_scaled)
            Ridge_model = create_Ridge(df_train_x_scaled, df_train_y_scaled)
            DNN_model = create_model_dnn(task_channel,df_train_x_scaled, df_train_y_scaled,test_plate)
#             svr_model = create_SVR(task_channel,df_train_x_scaled, df_train_y_scaled, channel_task_x, channel_task_y)
            #return task_channel,df_train_X, df_train_Y, channel_task_x, channel_task_y
    
            print('**************')
            print('LR')
            print('profile_treated:') 
            yhat_lr = pd.DataFrame(LR_model.predict(df_test_treated_x_scaled.values),
                              index=df_test_treated_x_scaled.index, columns=df_test_treated_y_scaled.columns)

            print('Linear Reg MSE: {}'.format(mean_squared_error(yhat_lr, df_test_treated_y_scaled.values)))
            print('Linear Reg MAE: {}'.format(mean_absolute_error(yhat_lr, df_test_treated_y_scaled.values)))
            
            print_results(test_plate, task_channel, 'Overall', 'Linear Regression', 'Treated', 'MSE', str(mean_squared_error(yhat_lr,df_test_treated_y_scaled.values)))
            print_results(test_plate, task_channel, 'Overall', 'Linear Regression', 'Treated', 'MAE', str(mean_absolute_error(yhat_lr,df_test_treated_y_scaled.values)))
            
            get_family_MSE(test_plate, task_channel, "Linear Regression", "Treated", yhat_lr, df_test_treated_y_scaled)
            get_family_MAE(test_plate, task_channel, "Linear Regression", "Treated", yhat_lr, df_test_treated_y_scaled)
            
            # Calculate MSE for each treatment
            joined = df_test_treated_y_scaled.join(yhat_lr, how='inner', lsuffix= '_Actual', rsuffix='_Predict')
            treats = joined.groupby('Metadata_broad_sample').apply(lambda g: pd.Series(
        {f'{task_channel}_MSE': mean_squared_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1)),
         f'{task_channel}_MAE': mean_absolute_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1))
         }))
            del joined

            treats['Plate'] = test_plate.split('.')[0]
            treats.set_index(['Plate'], inplace=True, append=True)
            curr_treatments[0].append(treats)
            del treats

            print('profile_mock:')            
            yhat_lr = pd.DataFrame(LR_model.predict(df_test_mock_x_scaled.values),
                              index=df_test_mock_x_scaled.index, columns=df_test_mock_y_scaled.columns)

            print('Linear Reg MSE: {}'.format(mean_squared_error(yhat_lr,df_test_mock_y_scaled.values)))
            print('Linear Reg MAE: {}'.format(mean_absolute_error(yhat_lr,df_test_mock_y_scaled.values)))
            
            print_results(test_plate, task_channel, 'Overall', 'Linear Regression', 'Mock', 'MSE', str(mean_squared_error(yhat_lr,df_test_mock_y_scaled.values)))
            print_results(test_plate, task_channel, 'Overall', 'Linear Regression', 'Mock', 'MAE', str(mean_absolute_error(yhat_lr,df_test_mock_y_scaled.values)))

            get_family_MSE(test_plate, task_channel, "Linear Regression", "Mock", yhat_lr,df_test_mock_y_scaled)
            get_family_MAE(test_plate, task_channel, "Linear Regression", "Mock", yhat_lr,df_test_mock_y_scaled)

            # Calculate MSE for each well control
            joined = df_test_mock_y_scaled.join(yhat_lr, how='inner', lsuffix= '_Actual', rsuffix='_Predict')

            ctrl = joined.groupby('Image_Metadata_Well').apply(lambda g: pd.Series(
        {f'{task_channel}_MSE': mean_squared_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1)),
         f'{task_channel}_MAE': mean_absolute_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1))
         }))
            del joined

            ctrl['Plate'] = test_plate.split('.')[0]
            ctrl.set_index(['Plate'], inplace=True, append=True)
            curr_controls[0].append(ctrl)
            del ctrl
                          
            print('**************')
            
            print('**************')
            print('Ridge')
            print('profile_treated:') 
            yhat_ridge = pd.DataFrame(Ridge_model.predict(df_test_treated_x_scaled.values),
                              index=df_test_treated_x_scaled.index, columns=df_test_treated_y_scaled.columns)

            print('Ridge MSE: {}'.format(mean_squared_error(yhat_ridge, df_test_treated_y_scaled.values)))
            print('Ridge MAE: {}'.format(mean_absolute_error(yhat_ridge, df_test_treated_y_scaled.values)))
                          
            print_results(test_plate, task_channel, 'Overall', 'Ridge', 'Treated', 'MSE', str(mean_squared_error(yhat_ridge,df_test_treated_y_scaled.values)))
            print_results(test_plate, task_channel, 'Overall', 'Ridge', 'Treated', 'MAE', str(mean_absolute_error(yhat_ridge,df_test_treated_y_scaled.values)))
                    
                          
            get_family_MSE(test_plate, task_channel, "Ridge", "Treated", yhat_ridge, df_test_treated_y_scaled)
            get_family_MAE(test_plate, task_channel, "Ridge", "Treated", yhat_ridge, df_test_treated_y_scaled)
                          
            # Calculate MSE for each treatment
            joined = df_test_treated_y_scaled.join(yhat_ridge, how='inner', lsuffix= '_Actual', rsuffix='_Predict')

            treats = joined.groupby('Metadata_broad_sample').apply(lambda g: pd.Series(
        {f'{task_channel}_MSE': mean_squared_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1)),
         f'{task_channel}_MAE': mean_absolute_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1))
         }))
            del joined

            treats['Plate'] = test_plate.split('.')[0]
            treats.set_index(['Plate'], inplace=True, append=True)
            curr_treatments[1].append(treats)
            del treats
            
            print('profile_mock:')            
            yhat_ridge = pd.DataFrame(Ridge_model.predict(df_test_mock_x_scaled.values),
                              index=df_test_mock_x_scaled.index, columns=df_test_mock_y_scaled.columns)

            print('Ridge MSE: {}'.format(mean_squared_error(yhat_ridge,df_test_mock_y_scaled.values)))
            print('Ridge Reg MAE: {}'.format(mean_absolute_error(yhat_ridge,df_test_mock_y_scaled.values)))

            print_results(test_plate, task_channel, 'Overall', 'Ridge', 'Mock', 'MSE', str(mean_squared_error(yhat_ridge,df_test_mock_y_scaled.values)))
            print_results(test_plate, task_channel, 'Overall', 'Ridge', 'Mock', 'MAE', str(mean_absolute_error(yhat_ridge,df_test_mock_y_scaled.values)))
                    
                          
            get_family_MSE(test_plate, task_channel, "Ridge", "Mock", yhat_ridge,df_test_mock_y_scaled)
            get_family_MAE(test_plate, task_channel, "Ridge", "Mock", yhat_ridge,df_test_mock_y_scaled)

             # Calculate MSE for each well control
            joined = df_test_mock_y_scaled.join(yhat_ridge, how='inner', lsuffix= '_Actual', rsuffix='_Predict')

            ctrl = joined.groupby('Image_Metadata_Well').apply(lambda g: pd.Series(
        {f'{task_channel}_MSE': mean_squared_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1)),
         f'{task_channel}_MAE': mean_absolute_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1))
         }))
            del joined

            ctrl['Plate'] = test_plate.split('.')[0]
            ctrl.set_index(['Plate'], inplace=True, append=True)
            curr_controls[1].append(ctrl)
            del ctrl

            print('**************')
            
            print('**************')
            print('DNN')
            print('profile_treated:')
            yhat_DNN = pd.DataFrame(DNN_model.predict(df_test_treated_x_scaled.values),
                              index=df_test_treated_x_scaled.index, columns=df_test_treated_y_scaled.columns)
            print('DNN MSE: {}'.format(mean_squared_error(yhat_DNN,df_test_treated_y_scaled.values)))
            print('DNN MAE: {}'.format(mean_absolute_error(yhat_DNN,df_test_treated_y_scaled.values)))

            print_results(test_plate, task_channel, 'Overall', 'DNN', 'Treated', 'MSE', str(mean_squared_error(yhat_DNN,df_test_treated_y_scaled.values)))
            print_results(test_plate, task_channel, 'Overall', 'DNN', 'Treated', 'MAE', str(mean_absolute_error(yhat_DNN,df_test_treated_y_scaled.values)))


            get_family_MSE(test_plate, task_channel, "DNN", "Treated", yhat_DNN,df_test_treated_y_scaled)
            get_family_MAE(test_plate, task_channel, "DNN", "Treated", yhat_DNN,df_test_treated_y_scaled)


            # Calculate MSE for each treatment
            joined = df_test_treated_y_scaled.join(yhat_DNN, how='inner', lsuffix= '_Actual', rsuffix='_Predict')

            treats = joined.groupby('Metadata_broad_sample').apply(lambda g: pd.Series(
        {f'{task_channel}_MSE': mean_squared_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1)),
         f'{task_channel}_MAE': mean_absolute_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1))
         }))
            del joined

            treats['Plate'] = test_plate.split('.')[0]
            treats.set_index(['Plate'], inplace=True, append=True)
            curr_treatments[2].append(treats)
            del treats


            print('profile_mock:')
            yhat_DNN = pd.DataFrame(DNN_model.predict(df_test_mock_x_scaled.values),
                              index=df_test_mock_x_scaled.index, columns=df_test_mock_y_scaled.columns)
            print('DNN MSE: {}'.format(mean_squared_error(yhat_DNN,df_test_mock_y_scaled.values)))
            print('DNN MAE: {}'.format(mean_absolute_error(yhat_DNN,df_test_mock_y_scaled.values)))


            print_results(test_plate, task_channel, 'Overall', 'DNN', 'Mock', 'MSE', str(mean_squared_error(yhat_DNN,df_test_mock_y_scaled.values)))
            print_results(test_plate, task_channel, 'Overall', 'DNN', 'Mock', 'MAE', str(mean_absolute_error(yhat_DNN,df_test_mock_y_scaled.values)))


            get_family_MSE(test_plate, task_channel, "DNN", "Mock", yhat_DNN,df_test_mock_y_scaled)
            get_family_MAE(test_plate, task_channel, "DNN", "Mock", yhat_DNN,df_test_mock_y_scaled)


            # Calculate MSE for each well control
            joined = df_test_mock_y_scaled.join(yhat_DNN, how='inner', lsuffix= '_Actual', rsuffix='_Predict')

            ctrl = joined.groupby('Image_Metadata_Well').apply(lambda g: pd.Series(
        {f'{task_channel}_MSE': mean_squared_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1)),
         f'{task_channel}_MAE': mean_absolute_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1))
         }))
            del joined

            ctrl['Plate'] = test_plate.split('.')[0]
            ctrl.set_index(['Plate'], inplace=True, append=True)
            curr_controls[2].append(ctrl)
            del ctrl

            print('**************')

        plate_treatments = [pd.concat(trt_per_model, axis=1) for trt_per_model in curr_treatments]
        for i in range(len(treatments)):
            treatments[i].append(plate_treatments[i])

        plate_wells = [pd.concat(well_per_model, axis=1) for well_per_model in curr_controls]
        for i in range(len(treatments)):
            controls[i].append(plate_wells[i])

    pd.concat(treatments[0]).to_csv(os.path.join('Results', 'Treats_LR.csv'))
    pd.concat(treatments[1]).to_csv(os.path.join('Results', 'Treats_Ridge.csv'))
    pd.concat(treatments[2]).to_csv(os.path.join('Results', 'Treats_DNN.csv'))

    pd.concat(controls[0]).to_csv(os.path.join('Results', 'Controls_LR.csv'))
    pd.concat(controls[1]).to_csv(os.path.join('Results', 'Controls_Ridge.csv'))
    pd.concat(controls[2]).to_csv(os.path.join('Results', 'Controls_DNN.csv'))


# # Main

# In[42]:


os.makedirs('Models', exist_ok=True)
os.makedirs('Results', exist_ok=True)
main('csvs/', S_STD)


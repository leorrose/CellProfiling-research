#!/usr/bin/env python
# coding: utf-8


# In[1] Imports:

# connections and OS
from os import chdir
from sys import argv

from preprocessing import *
from training import *
from evaluations import *


# In[2] main function:


def main(csv_folder, scale_method):
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

    csv_files = [_ for _ in listdir(csv_folder) if _.endswith(".csv")]

    # Holds error per plate
    treatments = {
            'Linear': [],
            'Ridge': [],
            'DNN': []
        }
    controls = {
            'Linear': [],
            'Ridge': [],
            'DNN': []
        }
    for test_plate in csv_files:
        # This is the current file that we will predict
        print(test_plate)

        # Holds error per channel
        curr_treatments = {
            'Linear': [],
            'Ridge': [],
            'DNN': []
        }
        curr_controls = {
            'Linear': [],
            'Ridge': [],
            'DNN': []
        }
        for task_channel in tqdm(CHANNELS):

            df_test_mock_x, df_test_mock_y, df_test_treated_x, df_test_treated_y, df_train_x, df_train_y = \
                split_train_test(csv_folder, csv_files, test_plate, task_channel)

            # Scale for training set
            x_scaler = fit_scaler(df_train_x, scale_method)
            y_scaler = fit_scaler(df_train_y, scale_method)

            df_train_x_scaled = scale_data(df_train_x, x_scaler)
            df_train_y_scaled = scale_data(df_train_y, y_scaler)

            df_test_treated_x_scaled = scale_data(df_test_treated_x, x_scaler)
            df_test_treated_y_scaled = scale_data(df_test_treated_y, y_scaler)
            df_test_mock_x_scaled = scale_data(df_test_mock_x, x_scaler)
            df_test_mock_y_scaled = scale_data(df_test_mock_y, y_scaler)

            # Model Creation - AVG error for each model:
            print(task_channel + ":")

            models = {
                'Linear': create_LR(df_train_x_scaled, df_train_y_scaled),
                'Ridge': create_Ridge(df_train_x_scaled, df_train_y_scaled),
                'DNN': create_model_dnn(task_channel, df_train_x_scaled, df_train_y_scaled, test_plate)
            }
            # svr_model = create_SVR(task_channel,df_train_x_scaled, df_train_y_scaled, channel_task_x, channel_task_y)

            for model, model_obj in models.items():
                deploy_model(model, model_obj, test_plate, task_channel, curr_controls, curr_treatments,
                             df_test_mock_x_scaled, df_test_mock_y_scaled, df_test_treated_x_scaled,
                             df_test_treated_y_scaled)

        plate_treatments = {model_type: pd.concat(trt_per_model, axis=1)
                            for model_type, trt_per_model in curr_treatments.items()}

        for model_type in treatments.keys():
            treatments[model_type].append(plate_treatments[model_type])

        plate_wells = {model_type: pd.concat(well_per_model, axis=1)
                       for model_type, well_per_model in curr_controls.items()}

        for model_type in controls.keys():
            controls[model_type].append(plate_wells[model_type])

    for model_type, plates_per_model in treatments.items():
        pd.concat(plates_per_model).to_csv(path.join('results', f'Treats_{model_type}.csv'))

    for model_type, plates_per_model in controls.items():
        pd.concat(plates_per_model).to_csv(path.join('results', f'Controls_{model_type}.csv'))


# In[3]:


def deploy_model(mode_type, model_obj, test_plate, task_channel, curr_controls, curr_treatments, df_test_mock_x_scaled,
                 df_test_mock_y_scaled, df_test_treated_x_scaled, df_test_treated_y_scaled):
    print('**************')
    print(mode_type)

    print('profile_treated:')
    y_pred = pd.DataFrame(model_obj.predict(df_test_treated_x_scaled.values),
                          index=df_test_treated_x_scaled.index, columns=df_test_treated_y_scaled.columns)

    print(f'{mode_type} {ERROR_TYPE}: {get_error(y_pred, df_test_treated_y_scaled.values)}')
    print_results(test_plate, task_channel, 'Overall', mode_type, 'Treated', ERROR_TYPE,
                  str(get_error(y_pred, df_test_treated_y_scaled.values)))

    get_family_error(test_plate, task_channel, mode_type, "Treated", y_pred, df_test_treated_y_scaled)

    # Calculate error for each treatment
    pred_result = df_test_treated_y_scaled.join(y_pred, how='inner', lsuffix='_Actual', rsuffix='_Predict')

    gb = pred_result.groupby('Metadata_broad_sample')

    treats = extract_errors_from_group_by(gb, test_plate, task_channel, y_pred.columns)
    curr_treatments[mode_type].append(treats)
    del treats

    print('profile_mock:')
    y_pred = pd.DataFrame(model_obj.predict(df_test_mock_x_scaled.values),
                          index=df_test_mock_x_scaled.index, columns=df_test_mock_y_scaled.columns)

    print(f'{mode_type} {ERROR_TYPE}: {get_error(y_pred, df_test_mock_y_scaled.values)}')
    print_results(test_plate, task_channel, 'Overall', mode_type, 'Mock', ERROR_TYPE,
                  str(get_error(y_pred, df_test_mock_y_scaled.values)))

    get_family_error(test_plate, task_channel, mode_type, "Mock", y_pred, df_test_mock_y_scaled)

    # Calculate error for each well control
    pred_result = df_test_mock_y_scaled.join(y_pred, how='inner', lsuffix='_Actual', rsuffix='_Predict')

    gb = pred_result.groupby('Image_Metadata_Well')
    del pred_result

    ctrl = extract_errors_from_group_by(gb, test_plate, task_channel, y_pred.columns)
    curr_controls[mode_type].append(ctrl)
    del ctrl

    print('**************')


def extract_errors_from_group_by(group_by, test_plate, task_channel, task_cols):
    error_by_channel = group_by.apply(lambda g: pd.Series(
        {f'{task_channel}_{ERROR_TYPE}': get_error(g.filter(regex='_Predict$', axis=1),
                                                   g.filter(regex='_Actual$', axis=1))
         }))

    error_by_feature = group_by.apply(lambda g: pd.DataFrame(
        [get_error(g.filter(regex='_Actual', axis=1), g.filter(regex='_Predict', axis=1),
                   multioutput='raw_values')], columns=task_cols)).droplevel(1)
    del group_by

    errors = pd.concat([error_by_channel, error_by_feature], axis=1)
    del error_by_channel, error_by_feature
    errors['Plate'] = test_plate.split('.')[0]
    errors.set_index(['Plate'], inplace=True, append=True)
    errors = errors.swaplevel(0, 1)
    return errors


# In[4] Main:

if __name__ == '__main__':
    if len(argv) == 2:
        print('Setting environment by input')
        PROJECT_DIRECTORY = argv[1]

    chdir(PROJECT_DIRECTORY)
    makedirs('results', exist_ok=True)
    main('csvs/', S_STD)

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


def main(csv_folder, scale_method, csv_files):
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
            print(task_channel + ":")

            # Load data
            df_test_mock_x, df_test_mock_y, df_test_treated_x, df_test_treated_y, df_train_x, df_train_y = \
                split_train_test(csv_folder, csv_files, test_plate, task_channel)

            # Scale data
            x_scaler = fit_scaler(df_train_x, scale_method)
            y_scaler = fit_scaler(df_train_y, scale_method)

            df_train_x_scaled = scale_data(df_train_x, x_scaler)
            df_train_y_scaled = scale_data(df_train_y, y_scaler)

            df_test_treated_x_scaled = scale_data(df_test_treated_x, x_scaler)
            df_test_treated_y_scaled = scale_data(df_test_treated_y, y_scaler)
            df_test_mock_x_scaled = scale_data(df_test_mock_x, x_scaler)
            df_test_mock_y_scaled = scale_data(df_test_mock_y, y_scaler)

            # Models' Creation
            models = {
                'Linear': create_LR(df_train_x_scaled, df_train_y_scaled),
                'Ridge': create_Ridge(df_train_x_scaled, df_train_y_scaled),
                'DNN': create_model_dnn(task_channel, df_train_x_scaled, df_train_y_scaled, test_plate)
            }

            # Models' Evaluation
            for model, model_obj in models.items():
                deploy_model(model, model_obj, test_plate, task_channel, curr_controls, curr_treatments,
                             df_test_mock_x_scaled, df_test_mock_y_scaled, df_test_treated_x_scaled,
                             df_test_treated_y_scaled)

        plate_treatments = {model_type: pd.concat(trt_per_model, axis=1)
                            for model_type, trt_per_model in curr_treatments.items()}

        plate_controls = {model_type: pd.concat(ctrl_per_model, axis=1)
                          for model_type, ctrl_per_model in curr_controls.items()}

        for model_type in curr_treatments.keys():
            err_path = fr'errors/{model_type}/{test_plate}'
            makedirs(fr'errors/{model_type}', exist_ok=True)
            pd.concat([plate_controls[model_type], plate_treatments[model_type]]).to_csv(err_path)


# In[3]:


def deploy_model(mode_type, model_obj, test_plate, task_channel, curr_controls, curr_treatments, df_test_mock_x_scaled,
                 df_test_mock_y_scaled, df_test_treated_x_scaled, df_test_treated_y_scaled):
    print('**************')
    print(mode_type)

    print('profile_treated:')
    y_pred_trt = pd.DataFrame(model_obj.predict(df_test_treated_x_scaled.values),
                              index=df_test_treated_x_scaled.index, columns=df_test_treated_y_scaled.columns)

    print(f'{mode_type} {ERROR_TYPE}: {get_error(y_pred_trt, df_test_treated_y_scaled.values)}')
    print_results(test_plate, task_channel, 'Overall', mode_type, 'Treated', ERROR_TYPE,
                  str(get_error(y_pred_trt, df_test_treated_y_scaled.values)))

    get_family_error(test_plate, task_channel, mode_type, "Treated", y_pred_trt, df_test_treated_y_scaled)

    # Calculate error for each treatment
    pred_result = df_test_treated_y_scaled.join(y_pred_trt, how='inner', lsuffix='_Actual', rsuffix='_Predict')

    treats = extract_errors(pred_result, task_channel, y_pred_trt.columns)
    curr_treatments[mode_type].append(treats)
    del treats

    print('profile_mock:')
    y_pred_ctrl = pd.DataFrame(model_obj.predict(df_test_mock_x_scaled.values),
                               index=df_test_mock_x_scaled.index, columns=df_test_mock_y_scaled.columns)

    print(f'{mode_type} {ERROR_TYPE}: {get_error(y_pred_ctrl, df_test_mock_y_scaled.values)}')
    print_results(test_plate, task_channel, 'Overall', mode_type, 'Mock', ERROR_TYPE,
                  str(get_error(y_pred_ctrl, df_test_mock_y_scaled.values)))

    get_family_error(test_plate, task_channel, mode_type, "Mock", y_pred_ctrl, df_test_mock_y_scaled)

    # Calculate error for each well control
    pred_result = df_test_mock_y_scaled.join(y_pred_ctrl, how='inner', lsuffix='_Actual', rsuffix='_Predict')

    ctrl = extract_errors(pred_result, task_channel, y_pred_ctrl.columns)
    curr_controls[mode_type].append(ctrl)
    del ctrl

    print('**************')


def extract_errors(joined, task_channel, task_cols):
    # error_by_channel = joined.apply(lambda row: pd.Series(
    #     {f'{task_channel}_{ERROR_TYPE}': get_error(row.filter(regex='_Predict$'),
    #                                                row.filter(regex='_Actual$'))
    #      }), axis=1)

    error_by_feature = joined.apply(lambda row: pd.Series(
        get_error([row.filter(regex='_Actual')],
                  [row.filter(regex='_Predict')],
                  multioutput='raw_values'), index=task_cols),
                                    axis=1)

    del joined

    # errors = pd.concat([error_by_channel, error_by_feature], axis=1)
    # del error_by_channel, error_by_feature

    return error_by_feature


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
    if len(argv) < 2:
        print('Usage: main.py [plates directory]')

    project_dir = argv[1]

    chdir(project_dir)
    makedirs('results', exist_ok=True)
    makedirs('errors', exist_ok=True)

    csv_fld = 'csvs/'

    csvs = [_ for _ in listdir(csv_fld) if _.endswith(".csv")]
    if len(argv) > 2:
        plates_numbers = argv[2:]
        plates_csvs = [plt + '.csv' for plt in plates_numbers]
        csvs = [plt for plt in csvs if plt in plates_csvs]

    main(csv_fld, S_STD, csvs)

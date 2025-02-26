import pandas as pd

from GPARModel import WindFarmGPAR
import pickle_helper as ph
import grapher as gr
from libraries import np
import itertools

model_history_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Modelling History 4.pkl'
models_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Models.pkl'
train_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl"
test_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl"
complete_train_data_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/Complete Training Data.pkl'
complete_test_data_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/Complete Test Data.pkl'
model_metadata_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Turbine Model Metadata.pkl'


# train_data = ph.read_pickle_as_dataframe(train_data_path)
# test_data = ph.read_pickle_as_dataframe(test_data_path)
#
# complete_train_data = ph.read_pickle_as_dataframe(complete_train_data_path)
# complete_test_data = ph.read_pickle_as_dataframe(complete_test_data_path)


def sample_complete_training_data(n=1000):
    complete_df = ph.read_pickle_as_dataframe(complete_train_data_path)
    sample_times = pd.Series(complete_df['Date.time'].unique()).sample(n)
    sample = complete_df[complete_df['Date.time'].isin(sample_times)]
    return sample


train_sample = ph.read_pickle_as_dataframe(
    "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Bigger Training Sample.pkl")
test_sample = ph.read_pickle_as_dataframe(
    "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Test Sample.pkl")

#
# models_path = [WindFarmGPAR(model_params={},
#                        existing=True,
#                        model_index=0)] * 6
#
# input_cols = ['Wind.speed.me']
# output_cols = ['Power.me']
#
# train_n = 2000
# test_n = 50
#
# train_timestamps = complete_train_data['Date.time'].sample(n=train_n)
# test_timestamps = complete_test_data['Date.time'].sample(n=test_n)

"""
USE A STRATIFIED SAMPLE INSTEAD

pandas has a command for this
"""

"""
fix bug (IT WAS NDMIN IN NP.ARRAY)
make new metadata pickle file
include calibration in plot/table
test with more turbines and different orderings (plots + graphs)
more covariates
"""
#
# train_x, train_y, test_x, test_y = [], [], [], []
# train_indices, test_indices = [], []
#
# # ------------------- SELECT TRAINING VALUES -------------------
#
# for time in train_timestamps.values:
#     current_data = complete_train_data[complete_train_data['Date.time'] == time]
#     # get the data at timestamp 'time'
#
#     x, y = [], []
#     for i in range(1, 7):
#         turbine_data = current_data[current_data['turbine'] == i]
#         # select turbine i data
#
#         x += turbine_data['Wind.speed.me'].values.tolist()
#         y += turbine_data['Power.me'].values.tolist()
#         train_indices += turbine_data['index'].values.tolist()
#         # append what we need
#         # --- the .values.tolist() cooouuullldd be overkill?
#     train_x.append(x)
#     train_y.append(y)
#
# # ------------------- SELECT TEST VALUES -------------------
#
# for time in test_timestamps.values:
#     current_data = complete_test_data[complete_test_data['Date.time'] == time]
#     x, y = [], []
#     for i in range(1, 7):
#         turbine_data = current_data[current_data['turbine'] == i]
#         x += turbine_data['Wind.speed.me'].values.tolist()
#         y += turbine_data['Power.me'].values.tolist()
#         test_indices += turbine_data['index'].values.tolist()
#     test_x.append(x)
#     test_y.append(y)
#
# train_x = np.array(train_x)
# train_y = np.array(train_y)
# test_x = np.array(test_x)
# test_y = np.array(test_y)
#
#
# model.train_model(train_x, train_y, test_x, test_y, train_indices, test_indices,
#                   input_columns=['Wind.speed.me'],
#                   output_columns=[f"Turbine {i} Power" for i in range(1, 7)])
#
input_cols = ph.read_pickle_as_dataframe("/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Prelim Analysis/Correlation Plot of Mean Covariates.pkl").columns

#     [
#     'Wind.dir.std',
#     'Wind.speed.me',
#     'Wind.speed.sd',
#     'Wind.speed.min',
#     'Wind.speed.max',
#     'Front.bearing.temp.me',
#     'Front.bearing.temp.sd',
#     'Front.bearing.temp.min',
#     'Front.bearing.temp.max',
#     'Rear.bearing.temp.me',
#     'Rear.bearing.temp.sd',
#     'Rear.bearing.temp.min',
#     'Rear.bearing.temp.max',
#     'Stator1.temp.me',
#     'Nacelle.ambient.temp.me',
#     'Nacelle.temp.me',
#     'Transformer.temp.me',
#     'Gear.oil.inlet.temp.me',
#     'Gear.oil.temp.me',
#     'Top.box.temp.me',
#     'Hub.temp.me',
#     'Conv.Amb.temp.me',
#     'Rotor.bearing.temp.me',
#     'Transformer.cell.temp.me',
#     'Motor.axis1.temp.me',
#     'Motor.axis2.temp.me',
#     'CPU.temp.me',
#     'Blade.ang.pitch.pos.A.me',
#     'Blade.ang.pitch.pos.B.me',
#     'Blade.ang.pitch.pos.C.me',
#     'Gear.oil.inlet.press.me',
#     'Gear.oil.pump.press.me',
#     'Drive.train.acceleration.me',
#     'Tower.Acceleration.x',
#     'Tower.Acceleration.y',
#     'Wind.dir.sin.me',
#     'Wind.dir.cos.me',
#     'Wind.dir.sin.min',
#     'Wind.dir.cos.min',
#     'Wind.dir.sin.max',
#     'Wind.dir.cos.max'
# ]


# train_data = complete_train_data.sample(n=200)
# test_data = complete_test_data.sample(n=30)
#
# for i in range(1, 7):
#     train_df = train_data[train_data['turbine'] == i]
#     test_df = test_data[test_data['turbine'] == i]
#
#     train_x = train_df['Wind.speed.me'].values.flatten()
#     train_y = train_df['Power.me'].values.flatten()
#     test_x = test_df['Wind.speed.me'].values.flatten()
#     test_y = test_df['Power.me'].values.flatten()
#     train_indices = train_df['index'].values.tolist()
#     test_indices = test_df['index'].values.tolist()
#     input_columns = ['Wind Speed']
#     output_columns = ['Power']
#
#     models_path[i - 1].train_model(train_x=train_x,
#                               train_y=train_y,
#                               test_x=test_x,
#                               test_y=test_y,
#                               train_indices=train_indices,
#                               test_indices=test_indices,
#                               input_columns=input_columns,
#                               output_columns=output_columns)


def generate_permutations(lst, min_length=1, max_length=6):
    if min_length > max_length:
        print("Invalid lengths")
        return None

    result = []
    max_length = max_length or len(lst)  # Default max_length to full length of list

    for length in range(min_length, max_length + 1):
        result.extend(itertools.permutations(lst, length))

    return result


turbine_perms = generate_permutations(lst=[1, 2, 3, 4, 5, 6], max_length=2)

# #
# train_df = complete_train_data
# test_df = complete_test_data
# #
# train_sample_times = train_df['Date.time'].sample(n=1000)
# test_sample_times = test_df['Date.time'].sample(n=100)
#
# train_sample = train_df[train_df['Date.time'].isin(train_sample_times)]
# test_sample = test_df[test_df['Date.time'].isin(test_sample_times)]
#
# train_sample.to_pickle("Training Sample.pkl")
# test_sample.to_pickle("Test Sample.pkl")
#
#
# train_indices = result['Training Data Indices']
# test_indices = result['Test Data Indices']

# train_df = train_data[train_data['index'].isin(train_indices)]
# test_df = test_data[test_data['index'].isin(test_indices)]


# ----------------- SINGLE TURBINE MODEL -----------------
#
# for i in turbines:
#     train_x = np.array(
#         train_sample[train_sample['turbine'] == i]['Wind.speed.me'].values,
#         ndmin=2
#     ).T
#     train_y = np.array(
#         train_sample[train_sample['turbine'] == i]['Power.me'].values,
#         ndmin=2
#     ).T
#     test_x = np.array(
#         test_sample[test_sample['turbine'] == i]['Wind.speed.me'].values,
#         ndmin=2
#     ).T
#     test_y = np.array(
#         test_sample[test_sample['turbine'] == i]['Power.me'].values,
#         ndmin=2
#     ).T
#
#     train_indices = train_sample['index'].values.tolist()
#     test_indices = test_sample['index'].values.tolist()
#     input_columns = ['Wind Speed']
#     output_columns = [f'Turbine {i} Power']
#
#     model.train_model(train_x=train_x,
#                       train_y=train_y,
#                       test_x=test_x,
#                       test_y=test_y,
#                       train_indices=train_indices,
#                       test_indices=test_indices,
#                       input_columns=input_columns,
#                       output_columns=output_columns,
#                       turbine_permutation=[i])
#
#     df_model_history = ph.read_pickle_as_dataframe(model_history_path)
#     chosen_index = len(df_model_history) - 1
#     result = df_model_history.iloc[chosen_index]
#
#     df_model_metadata = ph.read_pickle_as_dataframe(model_metadata_path)
#
#     current_metadata = df_model_metadata[df_model_metadata['Modelling History Index'] == chosen_index]
#
#     turbine_data = test_sample[test_sample['turbine'] == i]
#
#     x = turbine_data['Wind.speed.me'].values.tolist()
#     y = [turbine_data['Power.me'].values.tolist(),
#          # result['Means'][output_columns[i-turbines[0]]], have excluded means here since it just clogs up the graph
#          result['Lowers'][output_columns[0]],  # need to do -turbines[0] to offset i to be an index
#          result['Uppers'][output_columns[0]]
#          ]
#
#     gr.plot_graph(x=x,
#                   y_list=y,
#                   intervals=True,
#                   # labels=['Observation', 'Means', 'Lower', 'Upper'],
#                   labels=['Observation', 'Lower', 'Upper'],
#                   calibration=float(current_metadata['Calibration'].iloc[0]),
#                   colors=['black', 'red', 'blue'],
#                   x_label='Mean Wind Speed (metres per second)',
#                   y_label=f'Mean Power for Turbine {i}',
#                   model_history_index=str(chosen_index))  # ,
#     # save_path="/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/saved_graphs")


# ----------------- MULTI-TURBINE MODEL -----------------
input_col_names = input_cols
if True:  # left this here just so i don't run everything all over again
    for turbines in turbine_perms:
        train_x = pd.concat(
            [train_sample[train_sample['turbine'] == i][input_col_names].reset_index(drop=True) for i in turbines],
            axis=1
        ).to_numpy()
        # gather the input columns into a dataframe per turbine,
        # then join them together column-wise into one big numpy ndarray

        train_y = pd.concat(
            [train_sample[train_sample['turbine'] == i]['Power.me'].reset_index(drop=True) for i in turbines],
            axis=1
        ).to_numpy()

        test_x = pd.concat(
            [test_sample[test_sample['turbine'] == i][input_col_names].reset_index(drop=True) for i in turbines],
            axis=1
        ).to_numpy()

        test_y = pd.concat(
            [test_sample[test_sample['turbine'] == i]['Power.me'].reset_index(drop=True) for i in turbines],
            axis=1
        ).to_numpy()

        train_indices = train_sample['index'].values.tolist()
        test_indices = test_sample['index'].values.tolist()
        input_columns = input_cols
        output_columns = [f'Turbine {i} Power' for i in turbines]

        model = WindFarmGPAR(model_params={}, existing=True, model_index=0)
        # have to create a fresh model for every run, it was retraining from previous runs

        model.train_model(train_x=train_x,
                          train_y=train_y,
                          test_x=test_x,
                          test_y=test_y,
                          train_indices=train_indices,
                          test_indices=test_indices,
                          input_columns=input_columns,
                          output_columns=output_columns,
                          turbine_permutation=turbines,
                          modelling_history_path=model_history_path)
        del model
        # deleting it just to be sure it'll be fresh for the next run

    # df_model_history = ph.read_pickle_as_dataframe(model_history_path)
    # chosen_index = len(df_model_history) - 1
    # result = df_model_history.iloc[chosen_index]
    # # test_sample_indices = result['Test Data Indices']
    # #
    # # test_sample = tes[test_df['index'].isin(test_sample_indices)]
    #
    # df_model_metadata = ph.read_pickle_as_dataframe(model_metadata_path)
    #
    # current_metadata = df_model_metadata[df_model_metadata['Modelling History Index'] == chosen_index]
    #
    # for i in turbines:
    #     turbine_data = test_sample[test_sample['turbine'] == i]
    #
    #     x = turbine_data['Wind.speed.me'].values.tolist()
    #     y = [turbine_data['Power.me'].values.tolist(),
    #          # result['Means'][output_columns[i-turbines[0]]], have excluded means here since it just clogs up the graph
    #          result['Lowers'][output_columns[i - turbines[0]]],  # need to do -turbines[0] to offset i to be an index
    #          result['Uppers'][output_columns[i - turbines[0]]]
    #          ]
    #
    #     gr.plot_graph(x=x,
    #                   y_list=y,
    #                   intervals=True,
    #                   # labels=['Observation', 'Means', 'Lower', 'Upper'],
    #                   calibration=float(current_metadata['Calibration'].iloc[0]),
    #                   labels=['Observation', 'Lower', 'Upper'],
    #                   colors=['black', 'red', 'blue'],
    #                   x_label='Mean Wind Speed (metres per second)',
    #                   y_label=f'Mean Power for Turbine {i}',
    #                   model_history_index=str(chosen_index))  # ,
    #     save_path="/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/saved_graphs")

# for each ndarray, take turbine i's data (train or test) and return the x/y values (which are then reshaped to a
# normal python list
# the list comprehension returns the i x n 2-D python array
# this is put into an ndarray, and then transposed (using .T) to be n x i (the necessary shape for GPAR functions)


# for i in range(len(turbines)):
#     gr.plot_graph(x=train_x[i],
#                   y_list=[train_y[i]],
#                   model_history_index=-1,
#                   title='Training Data')
#     gr.plot_graph(x=test_x[i],
#                   y_list=[test_y[i]],
#                   model_history_index=-1,
#                   title='Test Data')

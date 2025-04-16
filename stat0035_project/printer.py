from libraries import np, pickle, pd
import pickle_helper as ph
import grapher as gr
from GPARModel import WindFarmGPAR

# Set option to display max number of columns
ph.libs.pd.set_option('display.max_columns', None)

models_path = WindFarmGPAR.models_filepath
train_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl"
test_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl"
complete_train_data_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/Complete Training Data.pkl'
complete_test_data_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/Complete Test Data.pkl'
model_metadata_path = WindFarmGPAR.turbine_model_metadata_filepath
train_sample_path = '/Users/sahmrahman/Desktop/GitHub/stat0035_project/Training Sample.pkl'
big_train_sample_path = '/Users/sahmrahman/Desktop/GitHub/stat0035_project/Big Training Sample.pkl'
bigger_train_sample_path = '/Users/sahmrahman/Desktop/GitHub/stat0035_project/Bigger Training Sample.pkl'
biggest_train_sample_path = '/Users/sahmrahman/Desktop/GitHub/stat0035_project/Biggest Training Sample.pkl'
test_sample_path = '/Users/sahmrahman/Desktop/GitHub/stat0035_project/Test Sample.pkl'

all_input_cols = [
    'Wind.dir.std',
    'Wind.speed.me',
    'Wind.speed.sd',
    'Wind.speed.min',
    'Wind.speed.max',
    'Front.bearing.temp.me',
    'Front.bearing.temp.sd',
    'Front.bearing.temp.min',
    'Front.bearing.temp.max',
    'Rear.bearing.temp.me',
    'Rear.bearing.temp.sd',
    'Rear.bearing.temp.min',
    'Rear.bearing.temp.max',
    'Stator1.temp.me',
    'Nacelle.ambient.temp.me',
    'Nacelle.temp.me',
    'Transformer.temp.me',
    'Gear.oil.inlet.temp.me',
    'Gear.oil.temp.me',
    'Top.box.temp.me',
    'Hub.temp.me',
    'Conv.Amb.temp.me',
    'Rotor.bearing.temp.me',
    'Transformer.cell.temp.me',
    'Motor.axis1.temp.me',
    'Motor.axis2.temp.me',
    'CPU.temp.me',
    'Blade.ang.pitch.pos.A.me',
    'Blade.ang.pitch.pos.B.me',
    'Blade.ang.pitch.pos.C.me',
    'Gear.oil.inlet.press.me',
    'Gear.oil.pump.press.me',
    'Drive.train.acceleration.me',
    'Tower.Acceleration.x',
    'Tower.Acceleration.y',
    'Wind.dir.sin.me',
    'Wind.dir.cos.me',
    'Wind.dir.sin.min',
    'Wind.dir.cos.min',
    'Wind.dir.sin.max',
    'Wind.dir.cos.max'
]


df = ph.read_pickle_as_dataframe(file_path=model_metadata_path)
print("...")












df = ph.read_pickle_as_dataframe("/Users/sahmrahman/Desktop/GitHub/stat0035_project/Complete n=1000 run on Wind Speed, Direction and Temperature (fixed hopefully).pkl")
gr.plot_mtgp_metadata(indices=df.index,
                      history_path="/Users/sahmrahman/Desktop/GitHub/stat0035_project/Complete n=1000 run on Wind Speed, Direction and Temperature (fixed hopefully).pkl",
                      save_path="/Users/sahmrahman/Desktop/GitHub/stat0035_project/saved_graphs/Complete Runs/n=1000/MTGP/Wind Speed, Direction and Temperature")

gpar_indices = pd.concat([ph.read_pickle_as_dataframe(
    "/Users/sahmrahman/Desktop/GitHub/stat0035_project/Complete Runs/GPAR/Complete n=1000 run on Wind Speed, Sine and Cosine of Direction, and Temperature - 1.pkl"),
                  ph.read_pickle_as_dataframe(
                      "/Users/sahmrahman/Desktop/GitHub/stat0035_project/Complete Runs/GPAR/Complete n=1000 run on Wind Speed, Sine and Cosine of Direction, and Temperature - 2.pkl")]).index
model_metadata = ph.read_pickle_as_dataframe(model_metadata_path)
model_metadata = model_metadata[model_metadata['Modelling History Index'].isin(gpar_indices)]
gr.plot_model_metadata(indices=model_metadata.index,
                       save_path="/Users/sahmrahman/Desktop/GitHub/stat0035_project/saved_graphs/Complete Runs/n=1000/GPAR/Wind Speed, Direction and Temperature")

mtgp = ph.read_pickle_as_dataframe(
    "/Users/sahmrahman/Desktop/GitHub/stat0035_project/Complete Runs/MTGP/Complete n=1000 run on Wind Speed, Direction and Temperature.pkl")
gr.plot_mtgp_metadata(indices=mtgp.index,
                      save_path="/Users/sahmrahman/Desktop/GitHub/stat0035_project/saved_graphs/Complete Runs/n=1000/MTGP/Wind Speed, Direction and Temperature")
print(...)

# df = pd.concat([
#     ph.read_pickle_as_dataframe('/Users/sahmrahman/Desktop/GitHub/stat0035_project/Complete Runs/Complete n=1000 run on Wind Speed, Sine and Cosine of Direction, and Temperature - 1.pkl'),
#     ph.read_pickle_as_dataframe('/Users/sahmrahman/Desktop/GitHub/stat0035_project/Complete Runs/Complete n=1000 run on Wind Speed, Sine and Cosine of Direction, and Temperature - 2.pkl')
# ])

# df = df[df['Input Columns'].apply(lambda x: len(x) == 4)]
# df = df[df['Output Columns'].apply(lambda x: len(x) == 6)]
print()

# model_metadata = model_metadata[model_metadata['Modelling History Index'].isin(df.index)]
# gr.print_model_metadata(indices=model_metadata.index)
# gr.plot_model_metadata(indices=model_metadata.index, save_path=)


input_cols = ['Wind.speed.me', 'Wind.dir.sin.me', 'Wind.dir.cos.me']

df = ph.read_pickle_as_dataframe("/stat0035_project/Complete Runs/GPAR/Complete n=1000 run on Wind Speed - 1.pkl")
# n1000_history = n1000_history[n1000_history['Training Data Indices'].apply(lambda x: x == n1000_indices)]
print('...')
# speed_and_dir = history[history['Input Columns'].apply(lambda x: len(x) == 3)]
# complete_speed_and_dir = speed_and_dir[speed_and_dir['Input Columns'].apply(lambda x: x == ['Wind Speed',
#                                                                                             'Sine of Wind Direction',
#                                                                                             'Cosine of Wind Direction'])]
selected_indices = history.index

# print(wind_speed_history.tail(5))

model_metadata = ph.read_pickle_as_dataframe(model_metadata_path)
selected_metadata = model_metadata[model_metadata['Modelling History Index'].isin(selected_indices)]
# print("========================================== N = 1000 ==========================================")
gr.print_model_metadata(selected_metadata.index)
# gr.plot_model_metadata(selected_metadata.index, save_path="/Users/sahmrahman/Desktop/GitHub/stat0035_project/saved_graphs/Single Turbine Model/n=1000 vs =2500 comparison/n=1000")

# test = model_metadata[model_metadata['Modelling History Index'] >= 6700]
# print(ph.get_model_history().iloc[6700]['Estimated Parameters'])
# print("========================================== N = 2500 ==========================================")
# gr.print_model_metadata(test.index)
# gr.plot_model_metadata(test.index, save_path="/Users/sahmrahman/Desktop/GitHub/stat0035_project/saved_graphs/Single Turbine Model/n=1000 vs =2500 comparison/n=2500")
# print(test)
# selected_metadata_indices = model_metadata.iloc[30120:].index
# gr.plot_model_metadata(selected_metadata_indices)
# gr.print_model_metadata(selected_metadata_indices)

# print(metadata)

# model_metadata = ph.read_pickle_as_dataframe(model_metadata_path)

ph.libs.pd.set_option('display.max_columns', None)
models = ph.read_pickle_as_dataframe(models_path)

# print(models.columns)
# print(models.tail())


test_sample = ph.read_pickle_as_dataframe(test_sample_path)

# selected_metadata = model_metadata[model_metadata['Modelling History Index'] > 6296]
# selected_indices = selected_metadata.index

# print(selected_metadata.iloc[0].)

# print(ph.get_model_history().tail(27))

# for turbine in range(1, 7):
#     data = history.iloc[turbine-1]
#     gr.plot_graph(x=test_sample[test_sample['turbine'] == turbine]['Wind.speed.me'],
#                   y_list=[test_sample[test_sample['turbine'] == turbine]['Power.me'].values,
#                           data['Lowers'][f'Turbine {turbine} Power'],
#                           data['Uppers'][f'Turbine {turbine} Power']],
#                   intervals=True,
#                   model_history_index=5400+777+turbine,
#                   calibration=model_metadata.iloc[29351+turbine]['Calibration'],
#                   title=f"Wind Speed vs Power for Turbine {turbine} (Single-output with 10000 rows of input)",
#                   save_path='/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/saved_graphs/Multi-Input Single-Turbine Model/All relevant covariates')

useful_covariates = [
    "Wind.speed.me",
    "Wind.speed.min",
    "Wind.speed.max",
    'Wind.speed.sd',
    "Transformer.temp.me",
    "Gear.oil.inlet.press.me",
    "Gear.oil.pump.press.me",
    "Drive.train.acceleration.me",
    "Tower.Acceleration.y",
    'CPU.temp.me',
    'Gear.oil.pump.press.me',
    'Nacelle.temp.me',
    'Top.box.temp.me',
    'Wind.dir.sin.me',
    'Wind.dir.cos.me',
]

# indices = [i for i in range(29352, len(model_metadata))]
# gr.plot_model_metadata(indices=selected_indices)#, save_path='/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/saved_graphs/Multi-Turbine Model')
# gr.print_model_metadata(indices=selected_indices)


# metadata_df = ph.read_pickle_as_dataframe(model_metadata_path)
# history_df = pd.concat([ph.read_pickle_as_dataframe(path) for path in [model_history_1,
#                                                                        model_history_2,
#                                                                        model_history_3]])

# -------------- SINGLE INPUT INDICES IN METADATA DATAFRAME: 311 up to 9780 --------------
# -------------- MULTI INPUT INDICES IN METADATA DATAFRAME: 9780 up to 19566 --------------


# print(df_modelling_history.tail(10))
# print("\n\n\n\nModels")
# df_models = ph.read_pickle_as_dataframe(file_path=models)
# print(df_models.head())
# print("...")
# print(df_models.tail())
# print("\n\n\n\n")

# train_indices = df_modelling_history.iloc[20]['Training Data Indices'].values.tolist()
# test_data_ = ph.read_pickle_as_dataframe(test_data__path)
# test_data_ = test_data_[test_data_['index'].isin(train_indices)]
# for turbine in range(1, 7):
#     current_turbine_data = train_sample[train_sample['turbine'] == turbine]
#     x = current_turbine_data['Wind.speed.me'].values.tolist()
#     y = [current_turbine_data['Power.me'].values.tolist()]
#     gr.plot_graph(x, y, 11,
#                   title=f"{gr.libs.datetime.now().strftime('%Y-%m-%d_%H-%M')} Turbine {turbine}")


# full_timestamps = []
# train_df = ph.read_pickle_as_dataframe(test_data__path)
# test_df = ph.read_pickle_as_dataframe(test_data_path)
# turbine_train_dfs = []
# turbine_test_dfs = []
#
# train_timestamps = []
# test_timestamps = []
#
# train_timestamps_len = 99999999
# test_timestamps_len = 99999999
#
# for turbine in range(1, 7):
#     turbine_train_dfs.append(train_df[train_df['turbine'] == turbine])
#     turbine_test_dfs.append(test_df[test_df['turbine'] == turbine])
#
#     if len(turbine_train_dfs[turbine - 1]['Date.time']) < train_timestamps_len:
#         # if we found a shorter timestamp series
#
#         train_timestamps = turbine_train_dfs[turbine - 1]['Date.time']
#         # get timestamps for current turbine
#         train_timestamps_len = len(train_timestamps)
#         # update length
#
#     if len(turbine_test_dfs[turbine - 1]['Date.time']) < test_timestamps_len:
#         test_timestamps = turbine_test_dfs[turbine - 1]['Date.time']
#         test_timestamps_len = len(test_timestamps)
#
# turbines = [1, 2, 3, 4, 5, 6]
#
# # ------------------ SELECTING COMPLETE TRAINING DATA ------------------
#
# train_indices_to_remove = []
#
# for index, time in train_timestamps.items():
#     timestamp_data = train_df[train_df['Date.time'] == time]
#     if not all(turbine in timestamp_data['turbine'].values for turbine in turbines):
#         # if the 'turbine' column doesn't have all six turbines... need to remove this timestamp
#         # incomplete data!
#         train_indices_to_remove.append(index)
#
# complete_turbine_test_data__timestamps = train_timestamps.drop(train_indices_to_remove)
# # get the timestamps with complete data
# complete_turbine_test_data_ = train_df[train_df['Date.time'].isin(complete_turbine_test_data__timestamps)]
# # select rows from the original whole training dataframe with those timestamps
#
# # ------------------ SELECTING COMPLETE TEST DATA ------------------
#
# test_indices_to_remove = []
#
# for index, time in test_timestamps.items():
#     timestamp_data = test_df[test_df['Date.time'] == time]
#     if not all(turbine in timestamp_data['turbine'].values for turbine in turbines):
#         # if the 'turbine' column doesn't have all six turbines... need to remove this timestamp
#         # incomplete data!
#         test_indices_to_remove.append(index)
#
# complete_turbine_test_data_timestamps = test_timestamps.drop(test_indices_to_remove)
# complete_turbine_test_data = test_df[test_df['Date.time'].isin(complete_turbine_test_data_timestamps)]
#
# complete_turbine_test_data_.to_pickle('Complete Training Data.pkl')
# complete_turbine_test_data.to_pickle('Complete Test Data.pkl')

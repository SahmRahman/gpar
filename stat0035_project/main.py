from GPARModel import WindFarmGPAR
import pickle_helper as ph
import grapher as gr

model_history = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Modelling History.pkl'
models = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Models.pkl'
train_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl"
test_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl"

train_data = ph.read_pickle_as_dataframe(train_data_path)
test_data = ph.read_pickle_as_dataframe(test_data_path)

model = WindFarmGPAR(model_params={},
                     existing=True,
                     model_index=0)

input_cols = ['Wind.speed.me']
output_cols = ['Power.me']

train_sample_times = ph.libs.np.random.choice(train_data['Date.time'].unique(), 50)
test_sample_times = ph.libs.np.random.choice(train_data['Date.time'].unique(), 10)

train_sample = train_data[train_data['Date.time'].isin(train_sample_times)]
test_sample = test_data[test_data['Date.time'].isin(test_sample_times)]


train_x = train_sample[train_sample['turbine'] == 1][input_cols].to_numpy()
train_y = (
    train_sample.groupby("turbine")["Power.me"]  # Group by the 'y' column, selecting 'x'
    .apply(list)          # Convert each group into a list
    .apply(ph.libs.pd.Series)     # Convert lists into separate Series
    .T                    # Transpose to make 'y' values the columns
).to_numpy()


test_x = test_sample[test_sample['turbine'] == 1][input_cols].to_numpy()
test_y = (
    train_sample.groupby("turbine")["Power.me"]  # Group by the 'y' column, selecting 'x'
    .apply(list)          # Convert each group into a list
    .apply(ph.libs.pd.Series)     # Convert lists into separate Series
    .T                    # Transpose to make 'y' values the columns
).to_numpy()


model.train_model(train_x, train_y, test_x, test_y, train_indices, test_indices, input_cols, output_cols)

# input_cols = [
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
#
# model = GPARModel.WindFarmGPAR(train_data_path=train_data_path,
#                                test_data_path=test_data_path,
#                                model_params={},
#                                existing=True,
#                                model_index=0,
#                                train_size=250,
#                                test_size=50)
# model.train_model(input_columns=['Wind.speed.me'],
#                   output_columns=['Power.me', 'Power.sd'])
#
# df = ph.read_pickle_as_dataframe(model_history)
# chosen_index = 16
#
# test_indices = df.iloc[chosen_index]['Test Data Indices']
# test_data = ph.read_pickle_as_dataframe(test_data_path)
# test_data = test_data[test_data['index'].isin(test_indices)]
#
# train_indices = df.iloc[chosen_index]['Training Data Indices']
# train_data = ph.read_pickle_as_dataframe(train_data_path)
# train_data = train_data[train_data['index'].isin(train_indices)]
#
# train_mean_power = ph.libs.np.mean(train_data['Power.me'])
# train_sd_power = ph.libs.np.std(train_data['Power.me'])
#
# test_mean_power = ph.libs.np.mean(test_data['Power.me'])
# test_sd_power = ph.libs.np.std(test_data['Power.me'])
#
# print(f"Training Data distribution for Power: N({round(float(train_mean_power), 2)}, {round(float(train_sd_power), 2)}^2)")
# print(f"Test Data distribution for Power: N({round(float(test_mean_power), 2)}, {round(float(test_sd_power), 2)}^2)")
#

# for col in ('Power.me', 'Power.sd'):
#     x = test_data['Wind.speed.me'].values.tolist()
#     y = [test_data[col].values.tolist(),
#          df.iloc[chosen_index]['Means'][col],
#          df.iloc[chosen_index]['Lowers'][col],
#          df.iloc[chosen_index]['Uppers'][col]
#          ]
#
#     gr.plot_graph(x=x,
#                   y_list=y,
#                   labels=['Observation', 'Means', 'Lower', 'Upper'],
#                   colors=['black', 'green', 'red', 'blue'],
#                   x_label='Mean Wind Speed (metres per second)',
#                   y_label=col,
#                   model_history_index=chosen_index,
#                   save_path="/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/saved_graphs")

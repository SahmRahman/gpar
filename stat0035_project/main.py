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

# input_cols = ['Wind.speed.me']
# output_cols = ['Power.me']

train_n = 100
test_n = 25

train_x = ph.libs.np.full(
    (train_n, 6),
    ph.libs.np.nan
)
train_y = ph.libs.np.full(
    (train_n, 6),
    ph.libs.np.nan
)
test_x = ph.libs.np.full(
    (test_n, 6),
    ph.libs.np.nan
)
test_y = ph.libs.np.full(
    (test_n, 6),
    ph.libs.np.nan
)

train_indices, test_indices = [], []

for turbine in train_data['turbine'].unique():
    train_df = train_data[train_data['turbine'] == turbine]
    test_df = test_data[test_data['turbine'] == turbine]

    train_sample = train_df.sample(train_n)
    test_sample = test_df.sample(test_n)

    train_indices += train_sample['index'].values.tolist()
    test_indices += test_sample['index'].values.tolist()

    for i in range(train_n):
        train_x[i, turbine - 1] = train_df.iloc[i]['Wind.speed.me']
        train_y[i, turbine - 1] = train_df.iloc[i]['Power.me']
        # have to do -1 for indexing

    i = 0

    for i in range(test_n):
        test_x[i, turbine - 1] = test_df.iloc[i]['Wind.speed.me']
        test_y[i, turbine - 1] = test_df.iloc[i]['Power.me']

model.train_model(train_x, train_y, test_x, test_y, train_indices, test_indices,
                  input_columns=['Wind.speed.me'],
                  output_columns=[f"Turbine {i}" for i in range(1, 7)])

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
df = ph.read_pickle_as_dataframe(model_history)
chosen_index = len(df) - 1

result = df.iloc[chosen_index]
test_indices = result['Test Data Indices']
# print(result['Means'])
#
# test_indices = df.iloc[chosen_index]['Test Data Indices'].values
# test_data = test_data[test_data['index'].isin(test_indices)]
#
# train_indices = df.iloc[chosen_index]['Training Data Indices']
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

test_data = test_data[test_data['index'].isin(test_indices)]
for i in range(1, 7):

    x = test_data[test_data['turbine'] == i]['Wind.speed.me'].values.tolist()
    y = [test_data[test_data['turbine'] == i]['Power.me'].values.tolist(),
         result['Means'][f"Turbine {i}"],
         result['Lowers'][f"Turbine {i}"],
         result['Uppers'][f"Turbine {i}"]
         ]

    gr.plot_graph(x=x,
                  y_list=y,
                  labels=['Observation', 'Means', 'Lower', 'Upper'],
                  colors=['black', 'green', 'red', 'blue'],
                  x_label='Mean Wind Speed (metres per second)',
                  y_label='Mean Power',
                  title=f"Turbine {i} {result['Timestamp']}",
                  model_history_index=chosen_index,
                  save_path="/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/saved_graphs")

import pickle_helper as ph
import grapher as gr

# Set option to display all columns
ph.pd.set_option('display.max_columns', None)

model_history = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Modelling History.pkl'
models = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Models.pkl'
train_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl"
test_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl"


# for filepath in (models, model_history):
#     df = ph.read_pickle_as_dataframe(file_path=filepath)
#     print(df.tail())

print("Model History")
df_model_history = ph.read_pickle_as_dataframe(file_path=model_history)
print(df_model_history.tail())
print("\n\n\n\nModels")
print(ph.read_pickle_as_dataframe(file_path=models).tail())

print("\n\n\n\n")

chosen_index = 8

print(df_model_history.iloc[chosen_index])

train_data = ph.read_pickle_as_dataframe(train_data_path)
test_data = ph.read_pickle_as_dataframe(test_data_path)

means =  df_model_history.iloc[chosen_index]['Means']['Power.me']
lowers = df_model_history.iloc[chosen_index]['Lowers']['Power.me']
uppers = df_model_history.iloc[chosen_index]['Uppers']['Power.me']

test_data_indices = df_model_history.iloc[chosen_index]['Test Data Indices']

input_cols = [
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

for col in input_cols:
    x = test_data[col].iloc[test_data_indices]
    y = [test_data['Power.me'].iloc[test_data_indices],
         means,
         lowers,
         uppers]

    gr.plot_graph(
        x=x,
        y_list=y,
        x_label='Mean Wind Speed (m/s)',
        y_label='Mean Power',
        title=f'{col} vs Power',
        colors=('black', 'green', 'red', 'blue'),
        labels=['Observed', 'Means', 'Lower', 'Upper'],
        save_path="/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/saved_graphs"
    )

# df = ph.read_pickle_as_dataframe(file_path=test_data_path)
# print(df.columns)
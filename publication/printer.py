from libraries import np, pickle, pd, sys
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
train_sample_path = '/Users/sahmrahman/Desktop/GitHub2/publication/Training Sample.pkl'
big_train_sample_path = '/Users/sahmrahman/Desktop/GitHub2/publication/Big Training Sample.pkl'
bigger_train_sample_path = '/Users/sahmrahman/Desktop/GitHub2/publication/Bigger Training Sample.pkl'
biggest_train_sample_path = '/Users/sahmrahman/Desktop/GitHub2/publication/Biggest Training Sample.pkl'
test_sample_path = '/Users/sahmrahman/Desktop/GitHub2/publication/Test Sample.pkl'
biggest_test_sample_path = '/Users/sahmrahman/Desktop/GitHub2/publication/Biggest Test Sample.pkl'

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

metadata = ph.read_pickle_as_dataframe(model_metadata_path).tail(2)

# keeping the below for later logging + csv gen'ing

sys.exit(0)

log = ph.read_pickle_as_dataframe("/Users/sahmrahman/Desktop/GitHub2/publication/Complete Runs/GPAR/Best Calibration/10k/History Log 10k.pkl")

big_train = ph.read_pickle_as_dataframe(biggest_train_sample_path)
big_test = ph.read_pickle_as_dataframe(biggest_test_sample_path)

input_df = big_train.loc[:, ['index', 'Date.time'] + log.loc['Input Columns', 10693] + ['Power.me', 'turbine']]
output_df = big_test.loc[:, ['index', 'Date.time', 'Power.me', 'turbine']]

dfs = []
for turbine_str in log.loc['Output Columns', 10693]:
    df = pd.concat(
        [pd.DataFrame(log.loc['Means', 10693][turbine_str], columns=['Means']),
         pd.DataFrame(log.loc['Lowers', 10693][turbine_str], columns=['Lowers']),
         pd.DataFrame(log.loc['Uppers', 10693][turbine_str], columns=['Uppers']),
         pd.DataFrame(log.loc['Error', 10693][turbine_str]['Squared Error'], columns=['Squared Error']),
         pd.DataFrame(log.loc['Error', 10693][turbine_str]['Absolute Error'], columns=['Absolute Error'])],
         axis=1
    )


    df[['index', 'Date.time', 'Power.me', 'turbine']] = output_df[output_df['turbine'] == int(turbine_str.split(" ")[1])].values
    df = df[['index', 'Date.time', 'Power.me', 'Means', 'Lowers',
             'Uppers', 'Squared Error', 'Absolute Error', 'turbine']]

    # df['Turbine'] = turbine_str.split(" ")[1]
    dfs.append(df)

# building out output csv, but, not sure about index/time matching!

final_df = pd.concat(dfs, axis=0)

input_df.to_csv("/Users/sahmrahman/Desktop/GitHub2/publication/Complete Runs/GPAR/Best Calibration/10k/Input Data.csv")
final_df.to_csv("/Users/sahmrahman/Desktop/GitHub2/publication/Complete Runs/GPAR/Best Calibration/10k/Output Data.csv")

print('...')
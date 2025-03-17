import pandas as pd

from GPARModel import WindFarmGPAR
import pickle_helper as ph
import itertools

model_history_path = '/Users/sahmrahman/Desktop/GitHub/stat0035_project/Modelling History 8.pkl'
models_path = WindFarmGPAR.models_filepath
train_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl"
test_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl"
complete_train_data_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/Complete Training Data.pkl'
complete_test_data_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/Complete Test Data.pkl'
model_metadata_path = WindFarmGPAR.turbine_model_metadata_filepath


# train_data = ph.read_pickle_as_dataframe(train_data_path)
# test_data = ph.read_pickle_as_dataframe(test_data_path)
#
# complete_train_data = ph.read_pickle_as_dataframe(complete_train_data_path)
# complete_test_data = ph.read_pickle_as_dataframe(complete_test_data_path)

def get_season(date):
    month, day = date.month, date.day

    if (month == 12 and day >= 21) or month in [1, 2] or (month == 3 and day < 21):
        return "Winter"
    elif (month == 3 and day >= 21) or month in [4, 5] or (month == 6 and day < 21):
        return "Spring"
    elif (month == 6 and day >= 21) or month in [7, 8] or (month == 9 and day < 21):
        return "Summer"
    else:
        return "Fall"


complete_data = ph.read_pickle_as_dataframe(complete_train_data_path)
train_sample = ph.read_pickle_as_dataframe(
    "/Users/sahmrahman/Desktop/GitHub/stat0035_project/Training Sample.pkl")
complete_data['Season'] = complete_data['Date.time'].map(get_season)
complete_season_dfs = [complete_data[complete_data["Season"] == season] for season in
                       ['Winter', 'Spring', 'Summer', 'Fall']]
train_sample['Season'] = train_sample['Date.time'].map(get_season)
train_season_dfs = [train_sample[train_sample["Season"] == season] for season in ['Winter', 'Spring', 'Summer', 'Fall']]


def sample_complete_training_data(n=1000):
    # get complete data by turbine
    df = ph.read_pickle_as_dataframe(complete_train_data_path)

    # Apply function to get season
    df["season"] = df["Date.time"].map(get_season)

    # Create separate DataFrames for each season
    season_dfs = [df[df["season"] == season] for season in ['Winter', 'Spring', 'Summer', 'Fall']]

    # sample n/4 times from each seasonal dataframe
    sample_times = [pd.Series(df_['Date.time'].unique()).sample(n // 4) for df_ in season_dfs]

    # combine inputs that have these times all into one dataframe
    sample = pd.concat([
        df[df['Date.time'].isin(sample_times[i])] for i in range(4)
    ],
        ignore_index=False)  # want to keep the indices since we store them for training
    return sample


train_sample = ph.read_pickle_as_dataframe(
    "/Users/sahmrahman/Desktop/GitHub/stat0035_project/Training Sample.pkl")
test_sample = ph.read_pickle_as_dataframe(
    "/Users/sahmrahman/Desktop/GitHub/stat0035_project/Test Sample.pkl")

input_cols = ['Wind.speed.me']

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


# all_covariates = [
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


def generate_permutations(lst=[1, 2, 3, 4, 5, 6], min_length=1, max_length=6):
    if min_length > max_length:
        print("Invalid lengths")
        return None

    result = []
    max_length = max_length or len(lst)  # Default max_length to full length of list

    for length in range(min_length, max_length + 1):
        result.extend(itertools.permutations(lst, length))

    return result


turbine_perms = generate_permutations(min_length=6)[256:]
''' ============= HAD TO SKIP FOLLOWING PERMUTATIONS DUE TO NUMERICAL INSTABILITY =============
(1,3,4,6,2,5)
(1,3,5,6,4,2)
'''
input_col_names = ['Wind.speed.me', "Wind.dir.sin.me", 'Wind.dir.cos.me',
                   'Nacelle.ambient.temp.me']  # useful_covariates
failed_perms = []
if True:  # left this here just so I don't run everything all over again
    for turbines in turbine_perms:
        train_x = pd.concat(
            [train_sample[train_sample['turbine'] == i][input_col_names].reset_index(drop=True) for i in turbines],
            axis=1
        ).to_numpy()
        # gather the input columns into a dataframe per turbine,
        # then append them together column-wise and convert into one big numpy ndarray

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
        input_columns = input_col_names
        output_columns = [f'Turbine {i} Power' for i in turbines]

        model = WindFarmGPAR(model_params={}, existing=True, model_index=0)
        # have to create a fresh model for every run, it was retraining from previous runs
        try:
            model.train_model(train_x=train_x,
                              train_y=train_y,
                              test_x=test_x,
                              test_y=test_y,
                              train_indices=train_indices,
                              test_indices=test_indices,
                              input_columns=input_columns,
                              output_columns=output_columns,
                              turbine_permutation=turbines,
                              modelling_history_path=model_history_path,
                              store_posterior=True
                              )
        except:
            print(f"Permutation {turbines} failed.")
            failed_perms.append(turbines)
            pass

        print()

        del model
        # deleting it just to be sure it'll be fresh for the next run

print("FAILED PERMUTATIONS")
for perm in failed_perms:
    print(perm)

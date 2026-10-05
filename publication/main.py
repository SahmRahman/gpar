import os
import sys

import pandas as pd
import numpy as np

from GPARModel import WindFarmGPAR
import pickle_helper as ph
import itertools

model_history_path = '/Users/sahmrahman/Desktop/GitHub2/publication/Modelling History 8.pkl'
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

''' some season stuff
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
    "/Users/sahmrahman/Desktop/GitHub2/publication/Training Sample.pkl")
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
'''


# train_sample = ph.read_pickle_as_dataframe(
#     "/Users/sahmrahman/Desktop/GitHub2/publication/Biggest Training Sample.pkl")
# test_sample = ph.read_pickle_as_dataframe(
#     "/Users/sahmrahman/Desktop/GitHub2/publication/Biggest Test Sample.pkl")

input_cols = ['Wind.speed.me', 'Wind.dir.sin.me', 'Wind.dir.cos.me', 'Nacelle.temp.me']



def sample_complete_training_data(n=1000):
    complete_df = ph.read_pickle_as_dataframe(complete_train_data_path)
    sample_times = pd.Series(complete_df['Date.time'].unique()).sample(n)
    sample = complete_df[complete_df['Date.time'].isin(sample_times)]
    return sample

def sample_complete_test_data(n=1000):
    complete_df = ph.read_pickle_as_dataframe(complete_test_data_path)
    sample_times = pd.Series(complete_df['Date.time'].unique()).sample(n)
    sample = complete_df[complete_df['Date.time'].isin(sample_times)]
    return sample

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


# turbine_perms = generate_permutations(min_length=6)[256:]
''' ============= HAD TO SKIP FOLLOWING PERMUTATIONS DUE TO NUMERICAL INSTABILITY =============
(1,3,4,6,2,5)
(1,3,5,6,4,2)
'''
input_col_names = ['Wind.speed.me', "Wind.dir.sin.me", 'Wind.dir.cos.me',
                   'Nacelle.ambient.temp.me']  # useful_covariates

turbines = [3, 4, 5, 1, 2, 6]
''' BEST PERM BY CALIBRATION '''


input_columns = input_col_names
# output_columns = [f'Turbine {i} Power' for i in range(1, 7)]
output_columns = ['Wind Farm Power']  # MUST USE 'Wind Farm Power', I've hardcoded this...!

# =============================================================================
# 1K vs 10K training-set comparison, trained from scratch, both scored on the
# SAME fixed 1,000-point held-out test set so the two are directly comparable.
#
# NOTE on modelling_history_path: WindFarmGPAR.store_posterior_model() and
# log_results() both derive "Modelling History Index" from
# pickle_helper.get_model_history(), which hardcodes a concatenation of
# "Modelling History 1..8.pkl" and ignores whatever path is passed in as
# modelling_history_path. Pointing this at a brand-new file (e.g. a fresh
# "Modelling History 9.pkl") would silently desync that bookkeeping, so this
# keeps using the same model_history_path ("...8.pkl") main.py already used.
# =============================================================================

# --- dummy paths: replace with the real files once confirmed ---
train_pool_1k_path = "/Users/sahmrahman/Desktop/GitHub2/publication/Training Sample.pkl"
train_pool_10k_path = "/Users/sahmrahman/Desktop/GitHub2/publication/Biggest Training Sample.pkl"
test_pool_1k_path = "/Users/sahmrahman/Desktop/GitHub2/publication/Biggest Test Sample.pkl"

N_REPS_1K = 10          # drop to 3 if 10 full GP fits is too slow
TRAIN_SIZE_1K = 990
N_RUNS_10K = 1          # >1 only useful if fitting itself is stochastic


def pivot_inputs_outputs(df, turbine_order, cols):
    """Pivot a long (timestamp x turbine) DataFrame into the wide arrays
    train_model expects: one row per timestamp, one block of columns per
    turbine. Factored out of the single pivot_table calls above so it can
    be reused for every resample below."""
    x = pd.pivot_table(df, values=cols, index=['Date.time'], columns=['turbine']
                       ).reset_index(drop=True).to_numpy()
    y = pd.pivot_table(df, values=['Power.me'], index=['Date.time'], columns=['turbine']
                       ).loc[:, [('Power.me', i) for i in turbine_order]].reset_index(drop=True).to_numpy()
    return x, y


def sample_n_unique_timestamps(pool_df, n, seed):
    """Randomly sample n unique Date.time values (i.e. n complete
    across-turbine rows) from a larger pool, without replacement."""
    rng = np.random.default_rng(seed)
    unique_times = pool_df['Date.time'].unique()
    if n > len(unique_times):
        raise ValueError(f"Requested {n} timestamps but pool only has {len(unique_times)}.")
    sampled_times = rng.choice(unique_times, size=n, replace=False)
    return pool_df[pool_df['Date.time'].isin(sampled_times)].copy()


def train_and_score(train_x, train_y, test_x, test_y, train_indices, test_indices):
    """Train a fresh model from scratch and read back its logged
    MSE / MAE / Calibration. sum_outputs=True with a single output column
    means train_model appends exactly one metadata row per call, so the
    last row is always the one this call just wrote."""
    model = WindFarmGPAR(model_params={}, existing=False, model_index=-1)

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
                      store_posterior=True,
                      predict_only=False,
                      sum_outputs=True
                      )

    last_row = ph.read_pickle_as_dataframe(model_metadata_path).iloc[-1]

    # store_posterior_model() appends exactly one new row to Models.pkl per
    # call, at the end -- record its index now, since nothing else tells you
    # this after the fact. Useful later if you want to re-score this exact
    # model without retraining.
    stored_model_index = len(ph.read_pickle_as_dataframe(models_path)) - 1

    return {
        'MSE': last_row['MSE'],
        'MAE': last_row['MAE'],
        'Calibration': last_row['Calibration'],
        'stored_model_index': stored_model_index,
    }


def summarise(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    return {'mean': mean, 'se': se, 'n': n}


# --- fixed test set, loaded once, used for BOTH setups ---
test_pool_df = ph.read_pickle_as_dataframe(test_pool_1k_path)
test_x_fixed, test_y_fixed = pivot_inputs_outputs(test_pool_df, turbines, input_col_names)
test_indices_fixed = test_pool_df['index'].values.tolist()

# =============================================================================
# 1K setup: resample 1,000 training points N_REPS_1K times, same fixed test set
# =============================================================================
train_pool_1k_df = ph.read_pickle_as_dataframe(train_pool_1k_path)

results_1k = []
for rep in range(N_REPS_1K):
    print(f"[1K rep {rep}] sampling and training...")
    try:
        train_df = sample_n_unique_timestamps(train_pool_1k_df, TRAIN_SIZE_1K, seed=rep)
        train_x, train_y = pivot_inputs_outputs(train_df, turbines, input_col_names)
        train_indices = train_df['index'].values.tolist()

        metrics = train_and_score(train_x, train_y, test_x_fixed, test_y_fixed,
                                  train_indices, test_indices_fixed)
        metrics['rep'] = rep
        results_1k.append(metrics)
    except Exception as e:
        print(f"[1K rep {rep}] Failed with Exception {e}.")

results_1k_df = pd.DataFrame(results_1k)
summary_1k = {metric: summarise(results_1k_df[metric]) for metric in ['MSE', 'MAE', 'Calibration']}

print("\n1K setup -- per-repetition results:")
print(results_1k_df)
print("\n1K setup -- mean +/- standard error across repetitions:")
for metric, stats in summary_1k.items():
    print(f"  {metric}: {stats['mean']:.4f} +/- {stats['se']:.4f}  (n={stats['n']})")

# =============================================================================
# 10K setup: full training pool, used as-is, scored on the SAME fixed test set
# =============================================================================
train_pool_10k_df = ph.read_pickle_as_dataframe(train_pool_10k_path)
train_x_10k, train_y_10k = pivot_inputs_outputs(train_pool_10k_df, turbines, input_col_names)
train_indices_10k = train_pool_10k_df['index'].values.tolist()

results_10k = []
for run in range(N_RUNS_10K):
    print(f"[10K run {run}] training from scratch...")
    try:
        metrics = train_and_score(train_x_10k, train_y_10k, test_x_fixed, test_y_fixed,
                                  train_indices_10k, test_indices_fixed)
        metrics['run'] = run
        results_10k.append(metrics)
    except Exception as e:
        print(f"[10K run {run}] Failed with Exception {e}.")

results_10k_df = pd.DataFrame(results_10k)
print("\n10K setup -- results:")
print(results_10k_df)

if N_RUNS_10K > 1:
    summary_10k = {metric: summarise(results_10k_df[metric]) for metric in ['MSE', 'MAE', 'Calibration']}
    print("\n10K setup -- mean +/- standard error across runs:")
    for metric, stats in summary_10k.items():
        print(f"  {metric}: {stats['mean']:.4f} +/- {stats['se']:.4f}  (n={stats['n']})")
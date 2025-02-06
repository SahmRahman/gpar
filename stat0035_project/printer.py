from libraries import np, pickle, pd
import pickle_helper as ph
import grapher as gr

# Set option to display max number of columns
ph.libs.pd.set_option('display.max_columns', None)

model_history = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Modelling History.pkl'
models = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Models.pkl'
train_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl"
test_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl"
complete_train_data_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/Complete Training Data.pkl'
complete_test_data_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/Complete Test Data.pkl'
model_metadata_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Turbine Model Metadata.pkl'


#print("Modelling History")
df_modelling_history = ph.read_pickle_as_dataframe(file_path=model_history)

# print(df_modelling_history.tail(15))

# ===================================== KEEP THIS =======================================
# chosen_indices = [82, 83, 84]
# turbines = [5, 6]
#
# for chosen_index in chosen_indices:
#     i = chosen_index - 77
#
#     print(f"Chosen Index: {chosen_index}")
#
#     result = df_modelling_history.iloc[chosen_index]
#
#     if chosen_index == chosen_indices[-1]:
#         print('Double-turbine model')
#
#         for t in turbines:
#             error_dict = result['Error'][f'Turbine {t} Power']
#
#             MSE = np.sqrt(np.mean(error_dict['Squared Error']))
#             MAE = np.mean(error_dict['Absolute Error'])
#
#             print(f"\tTurbine {t} MSE: {MSE}")
#             print(f"\tTurbine {t} MAE: {MAE}\n")
#
#             ph.append_to_pickle(file_path=turbine_metadata_path,
#                                 new_row={"Turbine": t,
#                                          "Model Type": 'Double',
#                                          "History Index": chosen_index,
#                                          "MSE": MSE,
#                                          'MAE': MAE})
#
#     else:
#         print(f'Turbine {i} Data; Single-turbine model')
#
#         error_dict = result['Error'][f'Turbine {i} Power']
#
#         MSE = np.sqrt(np.mean(error_dict['Squared Error']))
#         MAE = np.mean(error_dict['Absolute Error'])
#
#         print(f"\tTurbine {i} MSE: {MSE}")
#         print(f"\tTurbine {i} MAE: {MAE}\n")
#
#         ph.append_to_pickle(file_path=turbine_metadata_path,
#                             new_row={"Turbine": i,
#                                      "Model Type": 'Single',
#                                      "History Index": chosen_index,
#                                      "MSE": MSE,
#                                      'MAE': MAE})
# ==========================================================================================

model_metadata = ph.read_pickle_as_dataframe(model_metadata_path)
print(model_metadata.tail(20))

indices = [i for i in range(4, len(model_metadata))]
#gr.plot_model_metadata(indices, save_path='/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/saved_graphs/Multi-Turbine Model')

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

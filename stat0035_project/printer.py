import pickle_helper as ph

# Set option to display max number of columns
ph.libs.pd.set_option('display.max_columns', None)

model_history = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Modelling History.pkl'
models = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Models.pkl'
train_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl"
test_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl"

# print("Modelling History")
# df_modelling_history = ph.read_pickle_as_dataframe(file_path=model_history)
# print(df_modelling_history.head())
# print("...")
# print(df_modelling_history.tail())
# print("\n\n\n\nModels")
# df_models = ph.read_pickle_as_dataframe(file_path=models)
# print(df_models.head())
# print("...")
# print(df_models.tail())
# print("\n\n\n\n")

test_data = ph.read_pickle_as_dataframe(test_data_path)
time = test_data.iloc[0]['Date.time']
print(test_data[test_data['Date.time'] == time]['turbine'])

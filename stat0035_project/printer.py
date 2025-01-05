import pickle_helper as ph

# Set option to display all columns
ph.libs.pd.set_option('display.max_columns', None)

model_history = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Modelling History.pkl'
models = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Models.pkl'
train_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl"
test_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl"


print("Model History")
df_model_history = ph.read_pickle_as_dataframe(file_path=model_history)
print(df_model_history.tail())
print("\n\n\n\nModels")
print(ph.read_pickle_as_dataframe(file_path=models).tail())
print("\n\n\n\n")

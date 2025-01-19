import pickle_helper as ph
import grapher as gr

# Set option to display max number of columns
ph.libs.pd.set_option('display.max_columns', 6)

model_history = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Modelling History.pkl'
models = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Models.pkl'
train_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl"
test_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl"

# print("Modelling History")
df_modelling_history = ph.read_pickle_as_dataframe(file_path=model_history)
# print(df_modelling_history.head())
# print("...")
#print(df_modelling_history.tail())
# print("\n\n\n\nModels")
# df_models = ph.read_pickle_as_dataframe(file_path=models)
# print(df_models.head())
# print("...")
# print(df_models.tail())
# print("\n\n\n\n")

# train_indices = df_modelling_history.iloc[20]['Training Data Indices'].values.tolist()
# train_data = ph.read_pickle_as_dataframe(train_data_path)
# train_data = train_data[train_data['index'].isin(train_indices)]
for turbine in range(1, 7):
    current_turbine_data = train_sample[train_sample['turbine'] == turbine]
    x = current_turbine_data['Wind.speed.me'].values.tolist()
    y = [current_turbine_data['Power.me'].values.tolist()]
    gr.plot_graph(x, y, 11,
                  title=f"{gr.libs.datetime.now().strftime('%Y-%m-%d_%H-%M')} Turbine {turbine}")
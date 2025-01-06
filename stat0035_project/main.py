import GPARModel
import pickle_helper as ph
import grapher as gr


model_history = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Modelling History.pkl'
models = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Models.pkl'
train_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl"
test_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl"

df = ph.read_pickle_as_dataframe(model_history)
chosen_index = 3

test_indices = df.iloc[chosen_index]['Test Data Indices']
test_data = ph.read_pickle_as_dataframe(test_data_path).iloc[test_indices]

train_indices = df.iloc[chosen_index]['Training Data Indices']
train_data = ph.read_pickle_as_dataframe(train_data_path).iloc[train_indices]

train_mean_power = ph.libs.np.mean(train_data['Power.me'])
train_sd_power = ph.libs.np.std(train_data['Power.me'])

test_mean_power = ph.libs.np.mean(test_data['Power.me'])
test_sd_power = ph.libs.np.std(test_data['Power.me'])

print(f"Training Data distribution for Power: N({round(float(train_mean_power), 2)}, {round(float(train_sd_power), 2)}^2)")
print(f"Test Data distribution for Power: N({round(float(test_mean_power), 2)}, {round(float(test_sd_power), 2)}^2)")

# x = test_data['Wind.speed.me'].values.tolist()
# y = [test_data['Power.me'].values.tolist(),
#      df.iloc[chosen_index]['Means']['Power.me'],
#      df.iloc[chosen_index]['Lowers']['Power.me'],
#      df.iloc[chosen_index]['Uppers']['Power.me']
#      ]


# gr.plot_graph(x=x,
#               y_list=y,
#               labels=['Observation', 'Means', 'Lower', 'Upper'],
#               colors=['black', 'green', 'red', 'blue'],
#               x_label='Mean Wind Speed (m/s)',
#               y_label='Mean Power',
#               title=f'Wind Speed vs Power {gr.libs.datetime.now().strftime("%Y-%m-%d_%H-%M")}',
#               save_path='/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/saved_graphs')

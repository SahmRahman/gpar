import pickle_helper as ph

model_history = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Modelling History.pkl'
models = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Models.pkl'
train_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl"
test_data_path = "/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl"

# for filepath in (models, model_history):
#     df = ph.read_pickle_as_dataframe(file_path=filepath)
#     print(df.tail())

data = ph.read_pickle_as_dataframe(test_data_path)
# for col in data.columns:
#     print(col)

selected_columns = [
    "Avg. wind speed",
    "Stdev. wind speed",
    "Min. wind speed",
    "Max. wind speed",
    "Avg. rear bearing temp.",
    "Stdev. rear bearing temp.",
    "Min. rear bearing temp.",
    "Max. rear bearing temp.",
    "Avg. transformer temp.",
    "Avg. gear oil inlet temp.",
    "Avg. top box temp.",
    "Avg. conv. ambient temp.",
    "Avg. motor axis1 temp.",
    "Avg. CPU temp.",
    "Avg. blade angle pitch B",
    "Avg. gear oil inlet press",
    "Tower acceleration x",
    "Sine avg. wind speed direction",
    "Cosine avg. wind speed direction",
    "Sine min. wind speed direction",
    "Cosine min. wind speed direction",
    "Avg. front bearing temp.",
    "Stdev. front bearing temp.",
    "Min. front bearing temp.",
    "Max. front bearing temp.",
    "Avg. rotor bearing temp.",
    "Avg. stator1 temp.",
    "Avg. nacelle ambient temp.",
    "Avg. nacelle temp.",
    "Avg. gear oil temp.",
    "Avg. drive train acceleration",
    "Avg. hub temp.",
    "Avg. transformer cell temp.",
    "Avg. motor axis2 temp.",
    "Avg. blade angle pitch A",
    "Avg. blade angle pitch C",
    "Avg. gear oil pump press",
    "Tower acceleration y",
    "Sine max. wind speed direction",
    "Cosine max. wind speed direction",
    "Stdev. wind speed direction"
]
#
# selected_columns = [col.replace(" ", ".") for col in selected_columns]
# selected_columns = [col.replace("..", ".") for col in selected_columns]
#
# adjusted_columns = []
#
# stat_keywords = {'Avg.': '.me', 'Min.': '.min', 'Max.': '.max'}
#
# for col in selected_columns:
#     if col[:4] in stat_keywords.keys():
#         col = col[4:] + stat_keywords[col[:4]]
#     col = col[0].upper() + col[1:]
#     adjusted_columns.append(col)
#
# adjusted_columns = [col.replace("..", ".") for col in adjusted_columns]
#
# print("Need to choose:\n", adjusted_columns)
#
df_columns = data.columns.tolist()
# print("From:\n", df_columns, "\n")
#
# # Assuming adjusted_columns and df_columns are your two lists
# shared_elements = list(set(adjusted_columns) & set(df_columns))
#
# print("Shared elements:", shared_elements)
# print("Num of shared elements:", len(shared_elements))

selected_column_indices = [8,9,10,11,20,21,22,23,36,40,56,64,76,88,97,105,108,112,113,114,115,116,117,2,16,17,18,19,68,24,28,32,52,107,60,72,80,93,101,106,109]
selected_column_indices.sort()

selected_data = data.iloc[:, selected_column_indices]

print("\n".join(selected_data.columns.tolist()))
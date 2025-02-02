import libraries as libs


class PickleFileError(Exception):
    def __init__(self, error_message):
        print(error_message)
        libs.sys.exit()


class DataFrameNotFoundError(PickleFileError):
    def __init__(self, error_message):
        print(error_message)
        libs.sys.exit()


def read_pickle_as_dataframe(file_path):
    """
    Read all rows from a pickle file into a DataFrame. If the file doesn't exist, raises an exception.

    Parameters:
    - file_path (str): Path to the pickle file.

    Returns:
    - libs.pd.DataFrame: DataFrame constructed from the pickle file.

    Raises:
    - DataFrameNotFoundError: If the pickle file doesn't exist.
    """
    if not libs.os.path.exists(file_path):
        raise DataFrameNotFoundError(f"Pickle file not found: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            try:
                data = libs.pickle.load(f)
            except EOFError:
                data = []  # Return empty list if the file is empty
    except Exception as e:
        raise PickleFileError(f"Error reading the pickle file: {e}")

    return libs.pd.DataFrame(data)


# Function to validate a row against DataFrame column data types
def validate_row(df, df_to_append):
    # # Check if the row length matches the number of columns
    # if len(df_to_append.columns) != len(df.columns):
    #     # raise ValueError(
    #     #     f"Row length {len(df_to_append)} does not match the number of DataFrame columns {len(df.columns)}."
    #     # )
    #     pass

    # for col, value in zip(df.columns, df_to_append.values()):
    #     expected_dtype = df[col].dtype
    #     actual_dtype = libs.pd.Series([value]).dtype
    #
    #     # Check if the value matches the expected column dtype
    #     if not libs.pd.api.types.is_dtype_equal(expected_dtype, actual_dtype):
    #         raise ValueError(
    #             f"Value {value} in column '{col}' doesn't match expected type {expected_dtype}."
    #         )

    for col, value in zip(df_to_append.columns.to_list(), df_to_append.values.flatten().tolist()):
        if col in df.columns:
            expected_dtype = df[col].dtype
            actual_dtype = libs.pd.Series([value]).dtype

            # Check if the value matches the expected column dtype
            if not libs.pd.api.types.is_dtype_equal(expected_dtype, actual_dtype) and expected_dtype != 'object':
                raise ValueError(
                    f"Value {value} in column '{col}' doesn't match expected type {expected_dtype}."
                )
        else:
            raise ValueError(
                f'New Column "{col}" not found in DataFrame.'
            )


def append_to_pickle(file_path, new_row):
    """
    Append a single row to a pickle file. If the file doesn't exist, raises an exception.

    Parameters:
    - file_path (str): Path to the pickle file.
    - new_row (dict): New row data to append as a dictionary.

    Raises:
    - DataFrameNotFoundError: If the pickle file or DataFrame doesn't exist.
    - PickleFileError: If there's an issue with serializing or saving the pickle file.
    """
    if not libs.os.path.exists(file_path):
        raise DataFrameNotFoundError(f"Pickle file not found: {file_path}")

    data = read_pickle_as_dataframe(file_path)

    if type(new_row) == dict:
        new_row = libs.pd.DataFrame.from_dict([new_row])
        # need to wrap new_row in a list so pandas knows to make a single row dataframe

    validate_row(data, new_row)  # check incoming row matches values of dataframe

    data = libs.pd.concat([data, new_row], ignore_index=True)
    # if we got to here, then we can append the row
    # ignore_index will make sure the concatenated dataframe's indices are successive and
    # not affected by either of the original dataframes

    try:
        # Save updated data back to the pickle file
        with open(file_path, 'wb') as f:
            libs.pickle.dump(data, f)
    except Exception as e:
        raise PickleFileError(f"Error saving to pickle file: {e}")

# just keeping this if i never need to reset Models.pkl

# data_dict = {
#     'replace': libs.pd.Series(dtype='bool'),
#     'impute': libs.pd.Series(dtype='bool'),
#     'scale': libs.pd.Series(dtype='object'),  # Using 'object' for tensors
#     'scale_tie': libs.pd.Series(dtype='bool'),
#     'per': libs.pd.Series(dtype='bool'),
#     'per_period': libs.pd.Series(dtype='object'),  # Using 'object' for tensors
#     'per_scale': libs.pd.Series(dtype='object'),  # Using 'object' for tensors
#     'per_decay': libs.pd.Series(dtype='object'),  # Using 'object' for tensors
#     'input_linear': libs.pd.Series(dtype='bool'),
#     'input_linear_scale': libs.pd.Series(dtype='object'),  # Using 'object' for tensors
#     'linear': libs.pd.Series(dtype='bool'),
#     'linear_scale': libs.pd.Series(dtype='object'),  # Using 'object' for tensors
#     'nonlinear': libs.pd.Series(dtype='bool'),
#     'nonlinear_scale': libs.pd.Series(dtype='object'),  # Using 'object' for tensors
#     'rq': libs.pd.Series(dtype='bool'),
#     'markov': libs.pd.Series(dtype='int'),  # Using 'int' for Markov order
#     'noise': libs.pd.Series(dtype='object'),  # Using 'object' for tensors
#     'x_ind': libs.pd.Series(dtype='object'),  # Using 'object' for tensors
#     'normalise_y': libs.pd.Series(dtype='bool'),
#     'transform_y': libs.pd.Series(dtype='object')  # Using 'object' for tuples
# }

# turbine model metadata columns
# ['Turbine Count', 'Turbine Permutation', 'Modelling History Index', 'Model Index', 'Calibration', 'MSE', 'MAE']

import pickle
import os
import pandas as pd
import sys


class PickleFileError(Exception):
    def __init__(self, error_message):
        print(error_message)
        sys.exit()


class DataFrameNotFoundError(PickleFileError):
    def __init__(self, error_message):
        print(error_message)
        sys.exit()


def read_pickle_as_dataframe(file_path):
    """
    Read all rows from a pickle file into a DataFrame. If the file doesn't exist, raises an exception.

    Parameters:
    - file_path (str): Path to the pickle file.

    Returns:
    - pd.DataFrame: DataFrame constructed from the pickle file.

    Raises:
    - DataFrameNotFoundError: If the pickle file doesn't exist.
    """
    if not os.path.exists(file_path):
        raise DataFrameNotFoundError(f"Pickle file not found: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            try:
                data = pickle.load(f)
            except EOFError:
                data = []  # Return empty list if the file is empty
    except Exception as e:
        raise PickleFileError(f"Error reading the pickle file: {e}")

    return pd.DataFrame(data)


# Function to validate a row against DataFrame column data types
def validate_row(df, row):
    # Check if the row length matches the number of columns
    if len(row) != len(df.columns):
        raise ValueError(
            f"Row length {len(row)} does not match the number of DataFrame columns {len(df.columns)}."
        )

    for col, value in zip(df.columns, row):
        expected_dtype = df[col].dtype
        actual_dtype = pd.Series([value]).dtype

        # Check if the value matches the expected column dtype
        if not pd.api.types.is_dtype_equal(expected_dtype, actual_dtype):
            raise ValueError(
                f"Value {value} in column '{col}' doesn't match expected type {expected_dtype}."
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
    if not os.path.exists(file_path):
        raise DataFrameNotFoundError(f"Pickle file not found: {file_path}")

    data = read_pickle_as_dataframe(file_path)

    validate_row(data, new_row)  # check incoming row matches values of dataframe

    data[len(data)] = new_row  # if we got to here, then we can append the row

    try:
        # Save updated data back to the pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        raise PickleFileError(f"Error saving to pickle file: {e}")



# Example usage
# pickle_file_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/dummy test pickle.pkl'
#
# try:
#     append_to_pickle(pickle_file_path, new_row_data)
#     print("Row appended successfully.")
#
#     # Read back the pickle file to verify
#     df = read_pickle_as_dataframe(pickle_file_path)
#     print("Data in pickle file as DataFrame:\n", df)
# except (DataFrameNotFoundError, PickleFileError) as e:
#     print(f"Error: {e}")

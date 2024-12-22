import pickle
import os
import pandas as pd


class PickleFileError(Exception):
    """Custom exception for errors related to pickle file handling."""
    pass


class DataFrameNotFoundError(PickleFileError):
    """Exception raised when the DataFrame file does not exist."""
    pass


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

    try:
        # Load existing data
        with open(file_path, 'rb') as f:
            try:
                data = pickle.load(f)

                # Append the new row
                data.loc[len(data)] = new_row
            except EOFError:
                data = []  # In case the file is empty
    except Exception as e:
        raise PickleFileError(f"Error reading the pickle file: {e}")

    try:
        # Save updated data back to the pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        raise PickleFileError(f"Error saving to pickle file: {e}")


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


# Example usage
pickle_file_path = 'example.pkl'
new_row_data = {'Column1': 'Value1', 'Column2': 'Value2'}

try:
    append_to_pickle(pickle_file_path, new_row_data)
    print("Row appended successfully.")

    # Read back the pickle file to verify
    df = read_pickle_as_dataframe(pickle_file_path)
    print("Data in pickle file as DataFrame:\n", df)
except (DataFrameNotFoundError, PickleFileError) as e:
    print(f"Error: {e}")

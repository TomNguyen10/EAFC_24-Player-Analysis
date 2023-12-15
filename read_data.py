import pandas as pd


def read_soccer_data(file_path):
    """
    Read soccer data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the soccer data.
    """
    try:
        df = pd.read_csv(file_path)
        filtered_df = df[df['fifa_version'] == 24.0]
        print("Data loaded successfully.")
        return filtered_df
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

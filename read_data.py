import pandas as pd


def read_soccer_data(file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Filter columns to include only numerical values (excluding 'short_name' and 'player_positions')
        numerical_columns = df.select_dtypes(include=['number']).columns
        relevant_columns = list(numerical_columns) + \
            ['short_name', 'player_positions']

        filtered_df = df[relevant_columns]

        # Filter the DataFrame to include only fifa_version 24.0
        filtered_df = filtered_df[filtered_df['fifa_version'] == 24.0]

        # Print success message and return the filtered DataFrame
        print("Data loaded successfully.")
        return filtered_df
    except FileNotFoundError:
        # Handle file not found exception
        print(f"File not found at path: {file_path}")
        return None
    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred while loading the data: {e}")
        return None

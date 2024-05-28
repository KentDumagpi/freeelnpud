import pandas as pd

# File path of the dataset
file_path = r"C:\Users\KentDumagpi\PycharmProjects\pythonFreeEl3\vgsales.csv"

# Load the dataset into a pandas DataFrame
df = pd.read_csv(file_path)

# Increase display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Print the entire DataFrame
print(df)

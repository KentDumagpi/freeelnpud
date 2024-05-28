import pandas as pd

# File path of the dataset
file_path = r"C:\Users\KentDumagpi\PycharmProjects\pythonFreeEl3\vgsales.csv"

# Load the dataset into a pandas DataFrame
df = pd.read_csv(file_path)

# Increase display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Display the first 10 rows of the DataFrame
print("First 10 rows of the dataset:")
print(df.head(10))

# Display the summary statistics of the DataFrame
print("\nSummary statistics of the dataset:")
print(df.describe(include='all'))

# Display the information about the DataFrame
print("\nInformation about the dataset:")
print(df.info())

# Checking for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Drop rows with any missing values (if appropriate)
df = df.dropna()

# Checking for duplicates
print("\nNumber of duplicate rows in the dataset:")
print(df.duplicated().sum())

# Removing duplicate rows
df = df.drop_duplicates()

# Ensure correct data types (convert columns to appropriate types if necessary)
# For example, if the 'Year' column is not in integer format, convert it
if df['Year'].dtype != 'int64':
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').dropna().astype(int)

# Check data types again
print("\nData types after conversion:")
print(df.dtypes)

# Handling outliers
# For numerical columns, you can use various methods to detect and handle outliers
# For simplicity, here we use the IQR method to filter out outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

# Apply the function to numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for column in numerical_columns:
    df = remove_outliers(df, column)

# Display the cleaned DataFrame
print("\nCleaned dataset:")
print(df.head(10))

# Save the cleaned DataFrame to a new CSV file
cleaned_file_path = r"C:\Users\KentDumagpi\PycharmProjects\pythonFreeEl3\vgsales_cleaned.csv"
df.to_csv(cleaned_file_path, index=False)

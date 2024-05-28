import pandas as pd
import numpy as np

# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the dataset
file_path = 'vgsales_cleaned.csv'
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Basic Information about the dataset:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(df.head(8))

# Calculate and display descriptive statistics
print("\nDescriptive Statistics:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
descriptive_stats = df[numeric_cols].describe(include='all')

# Print the descriptive statistics
print(descriptive_stats)

# Calculate skewness and kurtosis for numeric columns
skewness = df[numeric_cols].skew()
kurtosis = df[numeric_cols].kurtosis()

# Print skewness and kurtosis
print("\nSkewness of numerical columns:")
print(skewness)

print("\nKurtosis of numerical columns:")
print(kurtosis)

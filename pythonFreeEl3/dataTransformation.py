import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'vgsales_cleaned.csv'
data = pd.read_csv(file_path)

# Select numerical features for standardization
numerical_features = ['Rank', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply standardization to the numerical features
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Set options to display all columns and rows without truncation
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows

# Display the first few rows of the standardized data
print(data.head(8))  # Adjust to show more rows if needed

# Save the standardized data to a new CSV file (optional)
data.to_csv('standardized_vgsales.csv', index=False)

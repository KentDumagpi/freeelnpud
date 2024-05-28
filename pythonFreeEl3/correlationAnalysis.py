import pandas as pd

# Load the dataset
file_path = "vgsales_cleaned.csv"
df = pd.read_csv(file_path)

# Exclude non-numeric columns (e.g., game titles) from correlation analysis
numeric_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Sort the correlation values with the target variable (Global_Sales)
correlation_with_sales = correlation_matrix['Global_Sales'].sort_values(ascending=False)

# Display the correlation values
print("Correlation with Global Sales:")
print(correlation_with_sales)

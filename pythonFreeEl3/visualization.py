import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = "vgsales_cleaned.csv"
df = pd.read_csv(file_path)

# Sort the DataFrame by Global Sales in descending order and select the top 10 games
top_games = df.sort_values(by='Global_Sales', ascending=False).head(10)

# Histogram of Global Sales with Labels
plt.figure(figsize=(10, 6))
hist_plot = sns.histplot(df['Global_Sales'], bins=30, kde=True)
plt.title('Histogram of Global Sales')
plt.xlabel('Global Sales (in millions)')
plt.ylabel('Frequency')


plt.show()

# Scatter Plot of Year vs Global Sales
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Year', y='Global_Sales', data=df)
plt.title('Scatter Plot of Year vs Global Sales')
plt.xlabel('Year')
plt.ylabel('Global Sales (in millions)')
plt.show()

# Box Plot of Global Sales by Genre
plt.figure(figsize=(14, 8))
sns.boxplot(x='Genre', y='Global_Sales', data=df)
plt.title('Box Plot of Global Sales by Genre')
plt.xlabel('Genre')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=45)
plt.show()

# Bar Plot of Total Sales by Platform
platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(14, 8))
platform_sales.plot(kind='bar')
plt.title('Total Global Sales by Platform')
plt.xlabel('Platform')
plt.ylabel('Total Global Sales (in millions)')
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
# Select only numeric columns for the correlation matrix
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()
